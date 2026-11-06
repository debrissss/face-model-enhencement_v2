"""完整迁自 TRELLIS automatic.py；算法参数与候选顺序保持原样。"""
import copy
import numpy as np
import open3d as o3d
from .quality import _head_anchor, _head_metrics, _color_metrics, face_core_metrics, face_core_passes, nose_distance_to_model
from ...utils.logging import get_logger

def nominal_rotation(view: str) -> np.ndarray:
    from .geometry import _rot_x, _rot_y
    angle = np.pi / 4
    rotations = {"up": np.eye(3), "down": _rot_x(angle),
                 "left": _rot_y(angle) @ np.diag([-1., -1., 1.]),
                 "right": _rot_y(-angle)}
    if view not in rotations:
        raise ValueError(f"该扫描器布局没有视角定义：{view}")
    return rotations[view]


def _rotation_difference(rotation, nominal):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation @ nominal.T) - 1) / 2, -1, 1))))


def _icp(source, target, initial, thresholds, iterations=60):
    matrix = initial.copy()
    for threshold in thresholds:
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, matrix,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations))
        matrix = np.asarray(result.transformation)
    return matrix


def _projection_image(raw, camera):
    k = camera["Cointrinsics"]
    points = np.asarray(raw.points)
    mask = points[:, 2] < -1
    points, colors = points[mask], np.asarray(raw.colors)[mask]
    uv = np.rint(points[:, :2] / points[:, 2:3] * [-k["fx"], k["fy"]] + [k["ppx"], k["ppy"]]).astype(int)
    width, height = k["width"], k["height"]
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, colors, points = uv[valid], colors[valid], points[valid]
    order = np.argsort(points[:, 2])  # nearest sample overwrites farthest
    canvas = np.full((height, width, 3), 225, dtype=np.uint8)
    canvas[uv[order, 1], uv[order, 0]] = np.rint(colors[order] * 255).astype(np.uint8)
    return canvas


def estimate_view(raw, vertices, surface, view: str, camera: dict, rgb, locator):
    """根据当前源点、目标顶点、原生 RGB 与相机估计刚体位姿。"""
    from .geometry import _preprocess, _dense_anchors, _coarse_rank, _sample_cloud
    logger = get_logger()
    anchor_model = _head_anchor(np.asarray(vertices.points))
    target = _preprocess(vertices, 2.5, remove_outliers=False)
    points = np.asarray(raw.points)
    colors = np.asarray(raw.colors) if raw.has_colors() else None
    rotation = nominal_rotation(view)
    face_info = {"status": "disabled"}
    face_anchor = None
    overlay = None
    if locator is not None:
        if rgb is None and raw.has_colors():
            rgb = _projection_image(raw, camera)
        if rgb is not None:
            face_anchor, face_info, overlay = locator.locate(rgb, points, camera, view)
            if colors is not None:
                k = camera["Cointrinsics"]
                image = rgb.astype(np.float64) / 255
                selected = np.arange(0, len(points), max(1, len(points) // 30000))
                p = points[selected]
                uv = np.rint(p[:, :2] / p[:, 2:3] * [-k["fx"], k["fy"]] + [k["ppx"], k["ppy"]]).astype(int)
                valid = (uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])
                residual = np.abs(colors[selected][valid] - image[uv[valid, 1], uv[valid, 0]])
                face_info["native_projection_rgb_mae"] = float(residual.mean()) if len(residual) else None
                face_info["native_projection_formula"] = "u=ppx-fx*x/z; v=ppy+fy*y/z"

    # Depth envelope limits background faces without selecting a historical
    # pose. Multiple skin/depth anchors remain available if 2D detection fails.
    plausible = (points[:, 2] > -900) & (points[:, 2] < -150)
    eligible = points[plausible]
    if len(eligible) < 500:
        eligible = points
    anchors = []
    def add(value, label):
        value = np.asarray(value)
        if all(np.linalg.norm(value - prior[0]) > 25 for prior in anchors):
            anchors.append((value, label))
    if face_anchor is not None:
        add(face_anchor, "rgb_nose")
    if colors is not None:
        skin = plausible & (colors[:, 0] > colors[:, 2] + 0.035) & (colors[:, 0] > colors[:, 1] - 0.06) & (colors[:, 0] > 0.18)
        if skin.sum() > 500:
            add(np.median(points[skin], axis=0), "skin_median")
            for value in _dense_anchors(points[skin], 4):
                add(value, "skin_depth_band")
    for value in _dense_anchors(eligible, 8):
        add(value, "depth_band")
    shifts = (-220., -120., -80., -40., 0., 40., 80., 120., 220.)
    offsets = np.array([(x, y, z) for x in shifts for y in shifts for z in shifts])
    candidates = []
    logger.info("stage_id=multi_view_point_cloud_alignment view=%s face=%s regions=%s", view, face_info["status"], len(anchors))
    for region_index, (anchor, label) in enumerate(anchors):
        mask = np.all(np.abs(points - anchor) <= 180, axis=1)
        # Apply the historical collar rule in the nominal model orientation,
        # so left's upside-down source Y does not remove the forehead.
        if colors is not None:
            relative_y = ((points - anchor) @ rotation.T)[:, 1]
            collar = (colors[:, 0] <= colors[:, 2] + 10/255) & (relative_y < -30)
            mask &= ~collar
        crop = raw.select_by_index(np.flatnonzero(mask).tolist())
        if len(crop.points) < 500:
            continue
        source = _preprocess(crop, 3.0)
        starts = []
        for i, offset in enumerate(offsets):
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = anchor_model + offset - rotation @ anchor
            starts.append((f"region{region_index}_shift{i}", matrix))
        ranked = _coarse_rank(_sample_cloud(source, 1800, region_index), target, starts)
        for _, init_label, initial, _, _ in ranked[:8]:
            transform = _icp(source, target, initial, (20., 10., 5., 3.), 60)
            angle = _rotation_difference(transform[:3, :3], rotation)
            if angle > 15:
                continue
            fit = o3d.pipelines.registration.evaluate_registration(source, target, 3, transform)
            candidates.append((float(fit.fitness), transform, region_index, label, source, angle))
        logger.info("stage_id=multi_view_point_cloud_alignment view=%s region=%s/%s kind=%s candidates=%s", view, region_index+1, len(anchors), label, len(candidates))
    if not candidates:
        raise ValueError("所有区域配准都失败或偏离扫描器布局超过 15°")
    candidates.sort(key=lambda x: x[0], reverse=True)
    finalists = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate[1][:3, 3] - prior[1][:3, 3]) > 3 for prior in finalists):
            finalists.append(candidate)
        if len(finalists) >= 12:
            break
    best = None
    trial_reports = []
    fine_target = _preprocess(vertices, 1.5, remove_outliers=False)
    for fit, coarse, region_index, label, source, angle in finalists:
        moved = points @ coarse[:3, :3].T + coarse[:3, 3]
        fixed_head = np.all(np.abs(moved - anchor_model) <= 110, axis=1)
        fine_source = _preprocess(raw.select_by_index(np.flatnonzero(fixed_head).tolist()), 1.5, remove_outliers=False)
        transform = _icp(fine_source, fine_target, coarse, (5., 3., 1.), 100)
        angle = _rotation_difference(transform[:3, :3], rotation)
        aligned = copy.deepcopy(raw).transform(transform)
        geometry, head = _head_metrics(aligned, vertices, anchor_model)
        color = _color_metrics(fine_source, fine_target, transform) if raw.has_colors() and fine_target.has_colors() else None
        score = geometry["fraction_within_3mm"] - min(geometry["median_distance_mm"], 10) * 0.1
        if color:
            score -= color["rgb_mean_absolute_error"] * 0.5
        if angle > 15:
            score -= 10
        nose_distance = None
        if face_anchor is not None:
            # Nose patch consistency against the central most camera-facing
            # model vertices, independent of which source crop scored well.
            v = np.asarray(vertices.points)
            center = np.median(v, axis=0)
            central = v[(np.abs(v[:, 0] - center[0]) < 30) & (np.abs(v[:, 1] - center[1]) < 65)]
            nose = central[central[:, 2] >= np.percentile(central[:, 2], 99)].mean(axis=0)
            mapped = transform[:3, :3] @ face_anchor + transform[:3, 3]
            nose_distance = float(np.linalg.norm(mapped - nose))
            if nose_distance > 40:
                score -= 1
        trial_reports.append({"region": region_index, "anchor_kind": label, "head": geometry,
                              "color": color, "rotation_deviation_degrees": angle, "nose_distance_mm": nose_distance, "score": score})
        if best is None or score > best[0]:
            best = (score, transform, aligned, geometry, fine_source, fine_target, color, angle, nose_distance, label)
    score, transform, aligned, geometry, source, target_fine, color, angle, nose_distance, label = best
    color_report = {"status": "not_available", "accepted": False, "trials": []}
    if color is not None:
        color_report.update({"status": "evaluated", "before": color, "after": color})
        # Colors participate in both candidate ranking and local refinement.
        for weight in (0.968, 0.85):
            try:
                result = o3d.pipelines.registration.registration_colored_icp(
                    source, target_fine, 2.0, transform,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=weight),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                proposed = np.asarray(result.transformation)
                cloud = copy.deepcopy(raw).transform(proposed)
                geom, _ = _head_metrics(cloud, vertices, anchor_model)
                proposed_color = _color_metrics(source, target_fine, proposed)
                passed = (geom["median_distance_mm"] <= geometry["median_distance_mm"] + 0.03
                          and geom["p90_distance_mm"] <= geometry["p90_distance_mm"] + 0.1
                          and proposed_color["rgb_mean_absolute_error"] < color_report["after"]["rgb_mean_absolute_error"]
                          and _rotation_difference(proposed[:3, :3], rotation) <= 15)
                color_report["trials"].append({"weight": weight, "geometry": geom, "color": proposed_color, "accepted": passed})
                if passed:
                    aligned, transform, geometry = cloud, proposed, geom
                    color_report.update({"accepted": True, "after": proposed_color})
            except RuntimeError as exc:
                color_report["trials"].append({"weight": weight, "error": str(exc)})
    full = np.asarray(aligned.compute_point_cloud_distance(surface))
    angle = _rotation_difference(transform[:3, :3], rotation)
    nose_distance = nose_distance_to_model(vertices, transform, face_info)
    head_passed = (geometry["median_distance_mm"] < 2 and geometry["fraction_within_3mm"] > 0.75 and angle <= 15
               and (nose_distance is None or nose_distance < 40))
    core = face_core_metrics(raw, aligned, vertices, camera, face_info)
    core_passed = face_core_passes(core, face_info, angle, nose_distance)
    quality = head_passed or core_passed
    metrics = {
        "status": "review_candidate" if quality else "low_confidence",
        "initialization": "automatic_rgb_multiregion_scanner_layout",
        "historical_pose_files_read": False,
        "nominal_scanner_rotation": rotation.tolist(),
        "nominal_rotation_source": "scanner view layout: up=0, down=Rx45, right=Ry-45, left=Ry45@Rz180; no per-scan translation",
        "face_detection": face_info,
        "source_anchors": [{"point": anchor.tolist(), "kind": kind} for anchor, kind in anchors],
        "finalist_candidates": trial_reports,
        "selected_anchor_kind": label,
        "head_metrics": geometry, "color_refinement": color_report,
        "head_region_passed": head_passed,
        "face_core_metrics": core,
        "face_core_passed": core_passed,
        "quality_basis": "head_region" if head_passed else "rgb_face_core" if core_passed else "failed",
        "final_nose_distance_mm": nose_distance,
        "head_region": {"model_anchor_mm": anchor_model.tolist(), "half_width_mm": 100, "distance_target": "original model vertices"},
        "point_count_input": len(raw.points), "point_count_output": len(aligned.points),
        "crop_fitness_5mm": geometry["fraction_within_5mm"],
        "aligned_crop_median_distance_mm": geometry["median_distance_mm"],
        "full_cloud_median_distance_mm": float(np.median(full)),
        "full_cloud_p90_distance_mm": float(np.percentile(full, 90)),
        "transform_source_to_model": transform.tolist(),
        "review_note": "仅使用当前 RGB/点云/model 和名义扫描器布局重新求解；仍需检查叠加图",
    }
    return aligned, transform, metrics, overlay


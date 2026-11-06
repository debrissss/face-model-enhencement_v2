"""迁自 TRELLIS alignment.py；保留采样、粗排序及候选参数。"""
from ...utils.logging import get_logger
import math
import numpy as np
import open3d as o3d

COARSE_DISTANCE_MM = 25.0
REFINE_CANDIDATES = 24

def _preprocess(
    cloud: o3d.geometry.PointCloud,
    voxel_mm: float,
    *,
    remove_outliers: bool = True,
) -> o3d.geometry.PointCloud:
    result = cloud.voxel_down_sample(voxel_mm)
    if len(result.points) < 50:
        raise ValueError("体素下采样后点数不足，无法估计位姿")
    if remove_outliers and len(result.points) > 100:
        neighbours = min(20, len(result.points) - 1)
        result, _ = result.remove_statistical_outlier(
            nb_neighbors=neighbours, std_ratio=2.0
        )
    radius = max(8.0, voxel_mm * 3.0)
    result.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    return result


def _dense_anchors(points: np.ndarray, limit: int) -> list[tuple[float, float, float]]:
    """Return representative points from the densest camera-depth bands."""
    if len(points) < 20:
        return [tuple(np.median(points, axis=0).tolist())]

    z_bins = np.floor(points[:, 2] / 10.0).astype(np.int64)
    _, inverse_bins, bin_counts = np.unique(
        z_bins, return_inverse=True, return_counts=True
    )
    top_bins = np.argsort(bin_counts)[::-1][: max(limit * 2, limit)]
    anchors: list[np.ndarray] = []
    for bin_index in top_bins:
        band = points[inverse_bins == bin_index]
        if len(band) < 5:
            continue
        voxel_keys = np.floor(band / 5.0).astype(np.int64)
        _, inverse_voxels, voxel_counts = np.unique(
            voxel_keys, axis=0, return_inverse=True, return_counts=True
        )
        best_voxel = int(np.argmax(voxel_counts))
        anchor = band[inverse_voxels == best_voxel].mean(axis=0)
        if all(np.linalg.norm(anchor - prior) > 20.0 for prior in anchors):
            anchors.append(anchor)
        if len(anchors) >= limit:
            break

    if not anchors:
        anchors.append(np.median(points, axis=0))
    return [tuple(anchor.astype(float).tolist()) for anchor in anchors]


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _valid_rigid_transform(matrix: np.ndarray) -> bool:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False
    rotation = matrix[:3, :3]
    return (
        np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5)
        and abs(float(np.linalg.det(rotation)) - 1.0) < 0.08
        and float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")) < 0.12
    )


def _score_transform(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    threshold_mm: float,
) -> tuple[float, float]:
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold_mm, transform
    )
    return float(result.fitness), float(result.inlier_rmse)


def _pack_voxel_keys(voxels: np.ndarray) -> np.ndarray:
    """Pack signed 21-bit voxel coordinates into sortable int64 keys."""
    offset = 1 << 20
    shifted = voxels.astype(np.int64, copy=False) + offset
    if np.any(shifted < 0) or np.any(shifted >= (1 << 21)):
        raise ValueError("点云坐标超出粗搜索空间范围")
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


def _coarse_rank(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    candidates: list[tuple[str, np.ndarray]],
) -> list[tuple[float, str, np.ndarray, float, float]]:
    # A dilated occupancy grid cheaply narrows thousands of rotation/anchor
    # starts. Exact Open3D scoring is then used on the shortlist.
    target_points = np.asarray(target.points)
    cell_mm = 8.0
    target_voxels = np.floor(target_points / cell_mm).astype(np.int64)
    offsets = np.asarray(
        [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
        dtype=np.int64,
    )
    expanded = target_voxels[:, None, :] + offsets[None, :, :]
    nearby_target_keys = np.unique(_pack_voxel_keys(expanded.reshape(-1, 3)))
    source_points = np.asarray(source.points)

    approximate: list[tuple[float, str, np.ndarray]] = []
    for label, transform in candidates:
        if not _valid_rigid_transform(transform):
            continue
        moved = source_points @ transform[:3, :3].T + transform[:3, 3]
        source_voxels = np.floor(moved / cell_mm).astype(np.int64)
        source_keys = _pack_voxel_keys(source_voxels)
        locations = np.searchsorted(nearby_target_keys, source_keys)
        valid = locations < len(nearby_target_keys)
        hits = np.zeros(len(source_keys), dtype=bool)
        hits[valid] = nearby_target_keys[locations[valid]] == source_keys[valid]
        approximate.append((float(hits.mean()), label, transform))
    approximate.sort(key=lambda item: item[0], reverse=True)

    exact_limit = max(REFINE_CANDIDATES * 4, 96)
    shortlist = approximate[:exact_limit]
    selected_labels = {label for _, label, _ in shortlist}
    # Keep a few candidates from each initializer family even when their
    # occupancy score is lower; feature matching and the cross-session prior
    # can still converge under ICP.
    for prefix, keep_count in (("ransac_", 8), ("fast_", 2), ("prior_", 12)):
        family = [item for item in approximate if item[1].startswith(prefix)]
        for score, label, transform in family[:keep_count]:
            if label not in selected_labels:
                shortlist.append((score, label, transform))
                selected_labels.add(label)

    ranked: list[tuple[float, str, np.ndarray, float, float]] = []
    for _, label, transform in shortlist:
        try:
            fitness, rmse = _score_transform(
                source, target, transform, COARSE_DISTANCE_MM
            )
        except Exception as exc:
            get_logger().warning("candidate %s could not be scored: %s", label, exc)
            continue
        score = fitness - min(rmse / COARSE_DISTANCE_MM, 1.0) * 0.04
        ranked.append((score, label, transform, fitness, rmse))
    ranked.sort(key=lambda item: (item[0], item[3], -item[4]), reverse=True)
    return ranked


def _sample_cloud(
    cloud: o3d.geometry.PointCloud, maximum_points: int, seed: int = 0
) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    if len(points) <= maximum_points:
        return cloud
    indices = np.random.default_rng(seed).choice(
        len(points), size=maximum_points, replace=False
    )
    sample = o3d.geometry.PointCloud()
    sample.points = o3d.utility.Vector3dVector(points[indices])
    return sample


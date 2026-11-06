"""Shared head-region geometry and RGB correspondence metrics."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def _head_anchor(points: np.ndarray) -> np.ndarray:
    """Same z-band / densest-voxel rule as historical face_pipeline."""
    bins, counts = np.unique(np.floor(points[:, 2] / 10).astype(int), return_counts=True)
    peak = bins[np.argmax(counts)]
    band = points[(points[:, 2] >= peak * 10 - 30) & (points[:, 2] <= peak * 10 + 30)]
    _, inverse, counts = np.unique(np.floor(band / 5).astype(int), axis=0, return_inverse=True, return_counts=True)
    return band[inverse == np.argmax(counts)].mean(axis=0)


def _head_metrics(aligned: o3d.geometry.PointCloud, vertices: o3d.geometry.PointCloud,
                  anchor: np.ndarray) -> tuple[dict, o3d.geometry.PointCloud]:
    mask = np.all(np.abs(np.asarray(aligned.points) - anchor) <= 100, axis=1)
    head = aligned.select_by_index(np.flatnonzero(mask).tolist())
    if len(head.points) < 500:
        raise ValueError("模型头部区域内点数不足")
    distances = np.asarray(head.compute_point_cloud_distance(vertices))
    return {
        "point_count": len(head.points),
        "median_distance_mm": float(np.median(distances)),
        "p90_distance_mm": float(np.percentile(distances, 90)),
        "fraction_within_2mm": float(np.mean(distances < 2)),
        "fraction_within_3mm": float(np.mean(distances < 3)),
        "fraction_within_5mm": float(np.mean(distances < 5)),
    }, head


def _color_metrics(source, target, transform) -> dict:
    result = o3d.pipelines.registration.evaluate_registration(source, target, 2.0, transform)
    pairs = np.asarray(result.correspondence_set)
    if not len(pairs):
        raise ValueError("颜色检查没有 2mm 内对应点")
    a = np.asarray(source.colors)[pairs[:, 0]]
    b = np.asarray(target.colors)[pairs[:, 1]]
    return {
        "correspondences_within_2mm": len(pairs),
        "rgb_mean_absolute_error": float(np.mean(np.abs(a - b))),
        "rgb_median_absolute_error": float(np.median(np.mean(np.abs(a - b), axis=1))),
        "geometric_rmse_mm": float(result.inlier_rmse),
        "fitness_2mm": float(result.fitness),
    }


def face_core_metrics(raw, aligned, vertices, camera: dict, face: dict) -> dict:
    """Independent RGB landmark ROI: brows/eyes to chin, excluding hair.

    The ROI depends only on current 2D landmarks and original camera rays,
    never on point-to-model distances. No points are removed from the output.
    """
    if face.get("status") != "detected" or not face.get("anchor_pixels"):
        return {"available": False, "reason": "no_valid_rgb_landmarks"}
    pixels = np.asarray(face["anchor_pixels"])
    left, right, chin = pixels[0], pixels[3], pixels[5]
    horizontal = right - left
    vertical = chin - (left + right) / 2
    triangle = np.array([left - 0.15 * horizontal - 0.1 * vertical,
                         right + 0.15 * horizontal - 0.1 * vertical,
                         chin + 0.03 * vertical])
    first, second = triangle[1] - triangle[0], triangle[2] - triangle[0]
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    determinant = float(cross(first, second))
    if abs(determinant) < 1:
        return {"available": False, "reason": "degenerate_landmark_roi"}
    points = np.asarray(raw.points)
    k = camera["Cointrinsics"]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = points[:, :2] / points[:, 2:3] * [-k["fx"], k["fy"]] + [k["ppx"], k["ppy"]]
    relative = uv - triangle[0]
    u, v = cross(relative, second) / determinant, cross(first, relative) / determinant
    mask = (u >= 0) & (v >= 0) & (u + v <= 1) & (points[:, 2] > -900) & (points[:, 2] < -150)
    indices = np.flatnonzero(mask)
    if len(indices) < 500:
        return {"available": False, "reason": "insufficient_face_points", "point_count": len(indices)}
    region = aligned.select_by_index(indices.tolist())
    distances = np.asarray(region.compute_point_cloud_distance(vertices))
    result = {"available": True, "point_count": len(indices), "roi_triangle_pixels": triangle.tolist(),
              "roi_definition": "fixed RGB landmark triangle expanded at brows and chin; independent of alignment residuals",
              "median_distance_mm": float(np.median(distances)),
              "p90_distance_mm": float(np.percentile(distances, 90)),
              "fraction_within_2mm": float(np.mean(distances < 2)),
              "fraction_within_3mm": float(np.mean(distances < 3))}
    if region.has_colors() and vertices.has_colors():
        result["color"] = _color_metrics(region, vertices, np.eye(4))
    return result


def face_core_passes(core: dict, face: dict, angle: float, nose_distance: float | None) -> bool:
    """Stricter facial alternative when unmodelled hair spoils a head box."""
    return bool(core.get("available") and core["point_count"] >= 2000
                and core["median_distance_mm"] < 1.0
                and core["p90_distance_mm"] < 2.0
                and core["fraction_within_3mm"] > 0.95
                and core.get("color", {}).get("rgb_mean_absolute_error", 1) < 0.08
                and face.get("native_projection_rgb_mae", 1) is not None
                and face.get("native_projection_rgb_mae", 1) < 0.02
                and angle < 10 and nose_distance is not None and nose_distance < 25)


def nose_distance_to_model(vertices, transform: np.ndarray, face: dict) -> float | None:
    if face.get("status") != "detected" or not face.get("anchor_view_mm"):
        return None
    points = np.asarray(vertices.points)
    center = np.median(points, axis=0)
    central = points[(np.abs(points[:, 0] - center[0]) < 30) & (np.abs(points[:, 1] - center[1]) < 65)]
    nose = central[central[:, 2] >= np.percentile(central[:, 2], 99)].mean(axis=0)
    mapped = transform[:3, :3] @ np.asarray(face["anchor_view_mm"]) + transform[:3, 3]
    return float(np.linalg.norm(mapped - nose))

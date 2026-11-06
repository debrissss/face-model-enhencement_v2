"""以原始 RGB 的关键点定义可重叠的近似面部分区。"""
from dataclasses import replace
import numpy as np
from scipy.spatial import ConvexHull

from face_model_enhancement.common.geometry import SurfaceVisibility, sample_pixels
from face_model_enhancement.common.regions import BROW_LOOPS, EYE_LOOPS, MOUTH_LOOP, NOSE_LANDMARKS, polygon_mask
from face_model_enhancement.config import RegionDetectionConfig
from face_model_enhancement.models import BoolArray, ComparisonRegions, FloatArray, PipelineData
from face_model_enhancement.stages.region_detection import detect_landmarks
from ..config import ComparisonRegionsConfig
from .base import BaseStage

# MediaPipe 正面轮廓及解剖标记；仅定义几何近似，不推测轮廓之外的发际线。
FACE_OVAL = (10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,
             148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109)
CHIN = (17,314,405,321,375,291,397,365,379,378,400,377,152,148,176,149,150,136,172,61,146,91,181,84)
MESH_BANDS = frozenset(("forehead", "cheeks"))


def build_image_masks(points: FloatArray, size: tuple[int, int], config: ComparisonRegionsConfig) -> BoolArray:
    face = polygon_mask(points[list(FACE_OVAL)], size)
    eye_distance = np.linalg.norm(points[[33,133]].mean(0) - points[[362,263]].mean(0))
    eye_padding = int(round(eye_distance * config.eye_padding_ratio))
    mouth_padding = int(round(eye_distance * config.mouth_padding_ratio))
    nose = points[list(NOSE_LANDMARKS)]
    nose_contour = nose[ConvexHull(nose).vertices]
    nose_mask = polygon_mask(nose_contour, size)
    expanded_nose = polygon_mask(nose_contour, size, int(round(eye_distance * config.nose_padding_ratio)))
    masks = {
        "forehead": np.ones_like(face),
        "eyes": np.logical_or.reduce([polygon_mask(points[list(loop)], size, eye_padding) for loop in EYE_LOOPS]),
        "nose": expanded_nose,
        "mouth": polygon_mask(points[list(MOUTH_LOOP)], size, mouth_padding),
        "chin": polygon_mask(points[list(CHIN)], size),
        "cheeks": ~nose_mask,
    }
    return np.stack([masks[name] if name in ("forehead", "cheeks") else masks[name] & face for name in config.regions])


def compute_view_height_bounds(data: PipelineData, points: FloatArray, names: tuple[str, ...]) -> FloatArray:
    """将边界锚点投到网格，以正面视图的高度定义水平截平面。"""
    centers = np.stack((points[[33, 133]].mean(0), points[[362, 263]].mean(0)))
    across = centers[1] - centers[0]
    eye_distance = np.linalg.norm(across)
    across /= eye_distance
    downward = np.array([-across[1], across[0]])
    if (points[152] - centers.mean(0)) @ downward < 0:
        downward = -downward
    brow = points[np.array(BROW_LOOPS).ravel()]
    brow_top = brow[np.argmin(brow @ downward)]
    lower_eyes = points[[145, 153, 154, 374, 380, 381]]
    eye_bottom = lower_eyes[np.argmax(lower_eyes @ downward)]
    nose = points[[2, 98, 327]]
    nose_bottom = nose[np.argmax(nose @ downward)]
    anchors = np.stack((brow_top - .15 * eye_distance * downward,
                        eye_bottom + .04 * eye_distance * downward,
                        nose_bottom + .04 * eye_distance * downward,
                        points[17] + .3 * (points[152] - points[17])))
    camera = data.camera
    rays = np.column_stack(((anchors[:, 0] - camera.ppx) / camera.fx,
                            -(anchors[:, 1] - camera.ppy) / camera.fy, -np.ones(len(anchors))))
    rays = rays @ camera.model_to_view[:3, :3]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    length = 2 * np.linalg.norm(data.mesh.vertices - camera.center, axis=1).max()
    hits, faces, _ = SurfaceVisibility(data.mesh).intersections(
        np.broadcast_to(camera.center, rays.shape), camera.center + length * rays)
    if (faces < 0).any():
        raise ValueError("A horizontal-region boundary anchor does not intersect the mesh.")
    heights = hits @ camera.model_to_view[1, :3]
    boundaries = {"forehead": (heights[0], np.inf), "cheeks": (heights[2], heights[1]),
                  "chin": (-np.inf, heights[3])}
    return np.array([boundaries.get(name, (-np.inf, np.inf)) for name in names])


def select_surface_regions(data: PipelineData, points: FloatArray, image_masks: BoolArray,
                           names: tuple[str, ...], bounds: FloatArray) -> BoolArray:
    """额头/面颊只限制高度，向两侧网格边缘延伸；面颊仅排除内部鼻区。"""
    pixels, depth = data.camera.project(points)
    masks = np.stack([sample_pixels(pixels, mask) for mask in image_masks])
    heights = points @ data.camera.model_to_view[1, :3]
    for index, name in enumerate(names):
        if name == "forehead":
            masks[index] = True
        elif name == "cheeks":
            # 面颊 image_mask 保存鼻部排除掩码的补集；图像外也不施加左右截断。
            masks[index] = ~sample_pixels(pixels, ~image_masks[index])
        masks[index] &= (heights >= bounds[index, 0]) & (heights <= bounds[index, 1])
        if name not in MESH_BANDS:
            masks[index] &= depth > 0
    return masks


def map_mesh_regions(data: PipelineData, image_masks: BoolArray, tolerance: float,
                        names: tuple[str, ...] = (), bounds: FloatArray | None = None) -> BoolArray:
    pixels, depth = data.camera.project(data.mesh.vertices)
    projected = (select_surface_regions(data, data.mesh.vertices, image_masks, names, bounds) if bounds is not None
                 else np.stack([sample_pixels(pixels, mask) for mask in image_masks]))
    # 网格高度带在曲面上延伸，不与原 RGB 相机可见域相交。
    projected_rows = [index for index in range(len(projected))
                      if bounds is None or names[index] not in MESH_BANDS]
    if not projected_rows:
        return projected
    candidates = projected[projected_rows].any(0) & (depth > 0)
    visible = np.zeros(len(data.mesh.vertices), dtype=bool)
    visible[candidates] = SurfaceVisibility(data.mesh).visible(data.mesh.vertices[candidates], data.camera, tolerance)
    selected = projected.copy()
    selected[projected_rows] &= visible
    return selected


def validate_region_surfaces(names: tuple[str, ...], vertex_masks: BoolArray) -> None:
    for name, mask in zip(names, vertex_masks):
        if not mask.any():
            raise ValueError(f"Selected region {name} has no eligible mesh vertices.")


class ComparisonRegionsStage(BaseStage[ComparisonRegionsConfig]):
    name = "comparison_regions"

    def run(self, data: PipelineData, config: ComparisonRegionsConfig) -> PipelineData:
        points, _ = detect_landmarks(data.rgb, data.camera, RegionDetectionConfig(config.model))
        masks = build_image_masks(points, (data.camera.width, data.camera.height), config)
        bounds = compute_view_height_bounds(data, points, config.regions)
        vertices = map_mesh_regions(data, masks, config.visibility_tolerance_mm, config.regions, bounds)
        validate_region_surfaces(config.regions, vertices)
        return replace(data, comparison_regions=ComparisonRegions(config.regions, points, masks, vertices, bounds))

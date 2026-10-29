"""以源网格标定相对深度，仅沿模型 Z 迁移双调和皮肤参考面上的眉毛起伏。"""
from dataclasses import replace
from typing import cast

import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.sparse.linalg import spsolve

from .base import BaseStage
from ..common.geometry import SurfaceVisibility, extract_selected_region, get_fixed_vertices
from ..common.regions import LEFT_BROW, RIGHT_BROW
from ..config import BrowDepthConfig
from ..models import BoolArray, BrowDepthResult, Camera, DepthMap, FloatArray, IntArray, MeshData, PipelineData, Regions


def fit_robust_affine(x: FloatArray, y: FloatArray, mask: BoolArray) -> tuple[float, float]:
    """沿用参考项目的 15 轮 Huber 加权仿射标定，不使用 GT。"""
    # 1. 只用标定掩码内的样本，将预测深度居中并缩放，改善线性求解的数值尺度。
    # 预测本身没有 mm 单位；常量预测无法提供可用于标定的深度变化。
    center, spread = float(np.median(x[mask])), float(np.std(x[mask]))
    if spread < 1e-12:
        raise ValueError("Constant predicted depth: cannot calibrate.")
    matrix = np.column_stack(((x[mask] - center) / spread, np.ones(mask.sum())))
    target, weight = y[mask], np.ones(mask.sum())
    # 2. 迭代降低大残差样本的权重，减少局部预测与源表面不一致对整体标定的影响。
    # 最小二乘的残差会被平方，因此矩阵和目标乘 sqrt(weight)，才能得到所需权重。
    for _ in range(15):
        root_weight = np.sqrt(weight)
        coef = np.linalg.lstsq(matrix * root_weight[:, None], target * root_weight, rcond=None)[0]
        residual = matrix @ coef - target
        # 用残差的中位绝对偏差估计尺度；Huber 规则保留小残差权重，降低大残差权重。
        sigma = max(1e-8, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weight = np.minimum(1., 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
    # 3. 将标准化坐标下的系数换回原始预测单位，供整幅图统一转换为 mm 深度。
    slope = float(coef[0] / spread)
    return slope, float(coef[1] - slope * center)


def interpolate_skin_depth(depth: FloatArray, unknown: BoolArray) -> FloatArray:
    """复用参考项目的最小平方五点拉普拉斯延续，已知皮肤像素固定。"""
    # 1. 为裁剪图内部像素建立五点拉普拉斯算子。
    # 后续最小化其平方和，使待填区域以较小的弯曲延续周围皮肤，而不保留眉毛起伏。
    h, w = depth.shape
    grid = np.arange(h * w).reshape(h, w)
    center = grid[1:-1, 1:-1].ravel()
    n, rows = len(center), np.arange(len(center))
    columns = np.concatenate((center, center - 1, center + 1, center - w, center + w))
    lap = sparse.csr_matrix((np.concatenate((np.full(n, -4.), np.ones(4 * n))),
                             (np.tile(rows, 5), columns)), shape=(n, h * w))
    # 2. 眉毛区域和所有缺失深度都作为未知量；复制掩码和深度，避免改动调用方数据。
    invalid = ~np.isfinite(depth)
    unknown = unknown.copy() | invalid
    border = np.zeros_like(unknown)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    boundary_missing = border & invalid
    values = depth.copy()
    # 3. 裁剪边界没有完整的五点邻域；缺失边界先取最近有效深度并固定，提供边界条件。
    # 其余有效的非眉毛像素保持原值，作为皮肤参考面的已知约束。
    if boundary_missing.any():
        nearest = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        values[boundary_missing] = depth[tuple(nearest[:, boundary_missing])]
        unknown[boundary_missing] = False
    # 4. 将已知像素的贡献移到右端，只求未知像素。
    # 仅保留涉及未知量的方程；其他方程不影响解。正规方程对应最小化 ||lap @ depth||²。
    ids, known = np.flatnonzero(unknown.ravel()), np.flatnonzero(~unknown.ravel())
    matrix = lap[:, ids]
    active = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
    matrix = matrix[active]
    rhs = -(lap[active][:, known] @ values.ravel()[known])
    fitted = spsolve((matrix.T @ matrix).tocsc(), matrix.T @ rhs)
    values.ravel()[ids] = fitted
    # 5. 检查参考面是否有效。
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite brow skin continuation.")
    return values


def sample_field(field: FloatArray, pixels: FloatArray) -> FloatArray:
    """在原生像素平面双线性采样；图外或无效深度的邻域保持无效。"""
    # 投影坐标按 (x,y) 排列，数组采样要求 (行,列)，因此先交换为 (y,x)。
    return map_coordinates(field, pixels[:, ::-1].T, order=1, mode="constant", cval=np.nan)


def compute_model_ray_directions(camera: Camera, pixels: FloatArray) -> FloatArray:
    """返回每单位相机前向深度对应的模型坐标射线向量。"""
    # 图像 y 向下、相机前向为 -Z；先按内参还原方向，再用旋转矩阵换回模型坐标。
    # 此处变换的是方向，不包含平移；射线起点由调用方使用 camera.center 指定。
    return np.column_stack(((pixels[:, 0] - camera.ppx) / camera.fx,
                            -(pixels[:, 1] - camera.ppy) / camera.fy,
                            -np.ones(len(pixels)))) @ camera.model_to_view[:3, :3]


def find_surface_depths_and_face_indices(mesh: MeshData, camera: Camera, surface: SurfaceVisibility,
                  pixels: FloatArray) -> tuple[FloatArray, IntArray]:
    """用现有 CPU 定位器查询像素首交点的相机深度与原始面索引。"""
    # 1. 将像素射线归一化，以相机到最远顶点距离的两倍作为查询线段长度，覆盖网格。
    rays = compute_model_ray_directions(camera, pixels)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    length = 2 * np.linalg.norm(mesh.vertices - camera.center, axis=1).max()
    # 2. 只取相机可见的首交表面，避免用背面或被遮挡面标定图像深度。
    hits, faces, _ = surface.intersections(np.broadcast_to(camera.center, rays.shape),
                                         camera.center + length * rays)
    # 3. 交点转为相机前向深度；未命中保留为 NaN，让后续覆盖检查和延续统一处理。
    depth = camera.project(hits)[1]
    depth[faces < 0] = np.nan
    return depth, faces


def limit_displacement(delta: FloatArray, budget_mm: float) -> tuple[FloatArray, float]:
    """以统一缩放满足欧氏位移预算，保持两眉位移的相对幅度。"""
    # 以全体顶点的最大位移确定同一个比例，避免逐点截断改变眉毛起伏的相对幅度。
    maximum = float(np.linalg.norm(delta, axis=1).max())
    scale = min(1., budget_mm / maximum) if maximum > 0 else 1.
    return delta * scale, scale


def calibrate_depth(mesh: MeshData, camera: Camera, regions: Regions,
                    surface: SurfaceVisibility, raw: FloatArray) -> FloatArray:
    """以全部有效皮肤样本拟合源网格毫米深度。"""
    # 1. 每隔 2 像素查询一次源表面，用较少的射线建立覆盖图像的标定样本。
    # 预测与源深度取自相同像素，避免将不同位置的深度配对。
    yy, xx = np.mgrid[0:raw.shape[0]:2, 0:raw.shape[1]:2]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    source, faces = find_surface_depths_and_face_indices(mesh, camera, surface, pixels)
    prediction = raw[yy.ravel(), xx.ravel()]
    # 2. 只用远离未命中区域、且图像和首交面都标为普通皮肤（标签 0）的有效样本。
    # 距离变换在降采样网格上计算，乘 2 换回原生像素，再排除距未命中区域不超过 8 像素的点。
    # 双重标签筛选排除眼口眉等区域，避免其局部形状差异参与全局深度标定。
    edge = distance_transform_edt(np.isfinite(source).reshape(xx.shape)).ravel() * 2
    known = (np.isfinite(prediction) & np.isfinite(source) & (edge > 8) &
             (regions.image_labels[yy.ravel(), xx.ravel()] == 0) &
             (regions.face_labels[np.maximum(faces, 0)] == 0))
    # 3. 全部通过有效性筛选的样本参与拟合，要求至少 500 个标定样本。
    if known.sum() < 500:
        raise ValueError("Insufficient source-only depth calibration anchors.")
    # 4. 拟合统一的尺度和平移，再应用于整幅相对深度；非正前向深度作为无效值处理。
    slope, shift = fit_robust_affine(prediction, source, known)
    field = slope * raw + shift
    field[field <= 0] = np.nan
    return field


def prepare_relative_depth(depth: DepthMap, camera: Camera) -> FloatArray:
    """检查原生深度尺寸与 Z 响应，在自有副本上应用有效掩码。"""
    if abs(camera.model_to_view[2, 2]) <= 1e-6:
        raise ValueError("Brow model-Z updates require a nonzero camera R33.")
    if depth.raw.shape != (camera.height, camera.width):
        raise ValueError("Depth and camera must have matching native image dimensions.")
    relative_depth = depth.raw.copy()
    relative_depth[~depth.valid] = np.nan
    return relative_depth


def extract_movable_brow_vertex_indices(mesh: MeshData, regions: Regions, brow_label: int) -> IntArray:
    """提取眉毛面片的可移动顶点，固定共享边界及非流形边顶点。"""
    selected = regions.face_labels == brow_label
    patch_vertex_indices, patch_face_indices = extract_selected_region(mesh, selected)
    fixed_vertex_mask = get_fixed_vertices(
        mesh, selected, regions.vertex_labels == brow_label, patch_vertex_indices, patch_face_indices,
    )
    patch_edge_indices, edge_face_counts = np.unique(
        np.sort(patch_face_indices[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
        axis=0, return_counts=True,
    )
    # get_fixed_vertices 返回本次调用新建的掩码，可直接补充非流形边固定点。
    fixed_vertex_mask[np.unique(patch_edge_indices[edge_face_counts != 2])] = True
    if (~fixed_vertex_mask).sum() < 3:
        raise ValueError(f"Brow has insufficient movable vertices for label {brow_label}.")
    return patch_vertex_indices[~fixed_vertex_mask]


def compute_crop_bounds(brow_pixel_mask: BoolArray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """计算向外留 6 像素的眉毛裁剪范围，返回左、右、上、下边界。"""
    rows, columns = np.nonzero(brow_pixel_mask)
    if not len(columns):
        raise ValueError("Missing native brow label.")
    height, width = image_shape
    return (max(0, int(columns.min()) - 6), min(width, int(columns.max()) + 7),
            max(0, int(rows.min()) - 6), min(height, int(rows.max()) + 7))


def extract_brow_depth_crops(mesh: MeshData, camera: Camera, surface: SurfaceVisibility,
                     calibrated_depth: FloatArray,
                     crop_bounds: tuple[int, int, int, int]) -> dict[str, FloatArray]:
    """在相同像素网格上查询源表面并裁剪已标定预测，输入及裁剪视图均只读。"""
    left, right, top, bottom = crop_bounds
    rows, columns = np.mgrid[top:bottom, left:right]
    source_depth, _ = find_surface_depths_and_face_indices(
        mesh, camera, surface, np.column_stack((columns.ravel(), rows.ravel())),
    )
    return {
        "source": source_depth.reshape(columns.shape),
        "prediction": calibrated_depth[top:bottom, left:right],
    }


def interpolate_brow_skin_depths(source_and_predicted_depth_crops: dict[str, FloatArray],
                       brow_crop_mask: BoolArray) -> dict[str, FloatArray]:
    """检查覆盖率，分别延续源表面与预测中的眉下皮肤参考面。"""
    skin_reference_maps: dict[str, FloatArray] = {}
    for depth_source, depth_crop in source_and_predicted_depth_crops.items():
        if np.isfinite(depth_crop).mean() < .85:
            raise ValueError(f"Inadequate brow skin continuation coverage: {depth_source}.")
        skin_reference_maps[depth_source] = interpolate_skin_depth(
            depth_crop, brow_crop_mask,
        )
    return skin_reference_maps


def compute_brow_relief_depth_delta(skin_reference_maps: dict[str, FloatArray], calibrated_depth: FloatArray,
                            moving_pixels: FloatArray, source_depth: FloatArray,
                            crop_origin: tuple[int, int]) -> FloatArray:
    """在固定源投影处比较各自相对皮肤的起伏，返回深度增量。"""
    crop_pixels = moving_pixels - crop_origin
    source_skin_depth = sample_field(skin_reference_maps["source"], crop_pixels)
    predicted_skin_depth = sample_field(skin_reference_maps["prediction"], crop_pixels)
    predicted_brow_depth = sample_field(calibrated_depth, moving_pixels)
    # 前向深度越小越靠近观察者；预测更凸时增量为负，顶点向观察者移动。
    depth_delta_mm = ((source_skin_depth - source_depth)
                      - (predicted_skin_depth - predicted_brow_depth))
    if not np.isfinite(depth_delta_mm).all():
        raise ValueError("Invalid predicted brow depth or skin reference.")
    return depth_delta_mm


def compute_brow_depth_deltas(mesh: MeshData, camera: Camera, regions: Regions,
                              surface: SurfaceVisibility, calibrated_depth: FloatArray,
                              strength: float) -> FloatArray:
    """固定源投影，分别计算双眉的相机深度增量（mm），并汇总到新建的全顶点数组。"""
    vertex_pixels, source_vertex_depth = camera.project(mesh.vertices)
    vertex_depth_delta_mm = np.zeros(len(mesh.vertices))
    for brow_label in (RIGHT_BROW, LEFT_BROW):
        # 从区域检测已保存的标签图提取当前眉毛掩码，不重新检测或绘制轮廓。
        brow_pixel_mask = regions.image_labels == brow_label
        crop_bounds = compute_crop_bounds(brow_pixel_mask, (camera.height, camera.width))
        moving_vertex_indices = extract_movable_brow_vertex_indices(mesh, regions, brow_label)
        source_and_predicted_depth_crops = extract_brow_depth_crops(
            mesh, camera, surface, calibrated_depth, crop_bounds,
        )
        left, right, top, bottom = crop_bounds
        skin_reference_maps = interpolate_brow_skin_depths(
            source_and_predicted_depth_crops, brow_pixel_mask[top:bottom, left:right],
        )
        depth_delta_mm = compute_brow_relief_depth_delta(
            skin_reference_maps, calibrated_depth, vertex_pixels[moving_vertex_indices],
            source_vertex_depth[moving_vertex_indices], (left, top),
        )
        vertex_depth_delta_mm[moving_vertex_indices] = depth_delta_mm * strength
    return vertex_depth_delta_mm


def apply_z_displacement(vertices: FloatArray, vertex_depth_delta_mm: FloatArray,
                         model_z_to_view_z: float,
                         maximum_displacement_mm: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """将相机深度增量换算为模型 Z 位移，统一限幅后写入自有顶点副本。"""
    displaced = vertex_depth_delta_mm != 0
    displacements = np.zeros_like(vertices)
    # 相机前向深度为 -view_Z，故仅改模型 Z 时 Δdepth=-R33*ΔZ。
    displacements[displaced, 2] = -vertex_depth_delta_mm[displaced] / model_z_to_view_z
    displacements, budget_scale = limit_displacement(displacements, maximum_displacement_mm)
    result_vertices = vertices.copy()
    result_vertices[displaced, 2] += displacements[displaced, 2]
    return result_vertices, displacements, vertex_depth_delta_mm * budget_scale


class BrowDepthConstraintStage(BaseStage[BrowDepthConfig, BrowDepthResult]):
    """法线约束后分别迁移双眉起伏，保持其他区域及原始拓扑不变。"""
    name = "brow_depth_constraint"
    output_field = "brow_depth"

    def run(self, data: PipelineData, config: BrowDepthConfig) -> PipelineData:
        """取得输入、串联独立计算步骤，并将眉毛结果提交到新的数据包。"""
        if data.constraint is None or data.regions is None or data.depth is None:
            raise ValueError("brow_depth_constraint requires regions, normal constraint and --input_depth.")
        source = data.eye_height if data.eye_height is not None else data.constraint
        source_mesh = MeshData(source.vertices, data.mesh.faces)
        relative_depth = prepare_relative_depth(data.depth, data.camera)
        surface = SurfaceVisibility(source_mesh)
        calibrated_depth = calibrate_depth(source_mesh, data.camera, data.regions, surface, relative_depth)
        depth_delta_mm = compute_brow_depth_deltas(
            source_mesh, data.camera, data.regions, surface, calibrated_depth, config.strength,
        )
        vertices, displacements, applied_depth_delta_mm = apply_z_displacement(
            source_mesh.vertices, depth_delta_mm, data.camera.model_to_view[2, 2],
            config.maximum_displacement_mm,
        )
        result = BrowDepthResult(vertices, displacements, applied_depth_delta_mm)
        return replace(data, brow_depth=result)

    def compute(self, data: PipelineData, config: BrowDepthConfig) -> BrowDepthResult:
        """兼容基类抽象接口；实际编排及数据包提交由本类 run() 完成。"""
        return cast(BrowDepthResult, self.run(data, config).brow_depth)

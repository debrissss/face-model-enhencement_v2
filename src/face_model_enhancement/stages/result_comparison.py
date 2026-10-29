"""计算 GT 刚体对齐、公共灰模取景及全脸和双眼剖面，不执行绘图。"""
import numpy as np
import trimesh

from .base import BaseStage
from ..config import ResultComparisonConfig
from ..common.geometry import SurfaceVisibility
from ..models import ComparisonResult, FloatArray, MeshData, PipelineData
from ..common.registration import fit_gt_rigid, registration_support


def surface_points(data: PipelineData, surface: SurfaceVisibility, pixels: FloatArray) -> FloatArray:
    """将指定像素沿正面相机射线投到原始 largeV 表面。"""
    camera = data.camera
    direction = np.column_stack(((pixels[:, 0] - camera.ppx) / camera.fx,
                                 -(pixels[:, 1] - camera.ppy) / camera.fy, -np.ones(len(pixels))))
    direction = direction @ camera.model_to_view[:3, :3]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length = 2 * np.linalg.norm(data.mesh.vertices - camera.center, axis=1).max()
    hits, faces, _ = surface.intersections(np.broadcast_to(camera.center, direction.shape),
                                         camera.center + length * direction)
    if (faces < 0).any():
        raise ValueError("A profile landmark ray does not intersect the original largeV mesh.")
    return hits


def nose_root(data: PipelineData, surface: SurfaceVisibility) -> FloatArray:
    """将原始鼻根关键点投到 largeV，确定固定全脸剖面的空间原点。"""
    assert data.regions is not None
    return surface_points(data, surface, data.regions.landmarks[[168]])[0]


def eye_profiles(data: PipelineData, meshes: tuple[MeshData, MeshData, MeshData],
                 rotation: FloatArray) -> dict[str, FloatArray]:
    """以原始双眼中心确定平行纵剖面和放大范围，三组模型共享各眼切面。"""
    assert data.regions is not None
    # MediaPipe 的 362、263 对应被摄者左眼，33、133 对应被摄者右眼。
    corners = data.regions.landmarks[[362, 263, 33, 133]].reshape(2, 2, 2)
    surface = SurfaceVisibility(data.mesh)
    origins = surface_points(data, surface, corners.mean(axis=1))
    corner_points = surface_points(data, surface, corners.reshape(-1, 2)).reshape(2, 2, 3)
    widths = np.linalg.norm(corner_points[:, 1] - corner_points[:, 0], axis=1)
    result = {"eye_origins": origins, "eye_half_heights_mm": widths * .65}
    for side, origin in zip(("left", "right"), origins):
        for name, mesh in zip(("original", "enhanced", "gt"), meshes):
            # 保留完整交线，由可视化的局部坐标范围完成放大，不筛选或修改几何。
            result[f"{side}_eye_{name}_section"] = midline_section(mesh, origin, rotation)
    return result


def midline_section(mesh: MeshData, origin: FloatArray, rotation: FloatArray) -> FloatArray:
    """提取公共纵剖面与原始网格的全部真实交线。"""
    geometry = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    segments = trimesh.intersections.mesh_plane(geometry, plane_normal=rotation[0], plane_origin=origin)
    # 完整保留三角面交线，只转换坐标，不筛选、连接或平滑线段。
    if len(segments) == 0:
        raise ValueError("The common midline plane has no surface intersection.")
    coordinates = (segments - origin) @ rotation.T
    return coordinates[:, :, [2, 1]]


class ResultComparisonStage(BaseStage[ResultComparisonConfig, ComparisonResult]):
    """以原始输入为固定基准，准备三组模型的配准、公共取景和剖面对比数据。"""
    name = "result_comparison"
    output_field = "comparison"

    def compute(self, data: PipelineData, config: ResultComparisonConfig) -> ComparisonResult:
        """只变换 GT 副本，计算三组模型共享的相机框架及全脸、双眼纵剖面。"""
        if data.gt is None or data.constraint is None or data.regions is None:
            raise ValueError("result_comparison requires --input_3dMD and both preceding stages.")
        transform, report = fit_gt_rigid(data.mesh, data.gt, registration_support(data), config)
        aligned_vertices = data.gt.vertices @ transform[:3, :3].T + transform[:3, 3]
        enhanced = data.enhanced_mesh
        gt = MeshData(aligned_vertices, data.gt.faces)
        meshes = (data.mesh, enhanced, gt)
        origin = nose_root(data, SurfaceVisibility(data.mesh))
        rotation = data.camera.model_to_view[:3, :3].copy()
        sections = tuple(midline_section(mesh, origin, rotation) for mesh in meshes)
        # 三个方向、三组模型共用同一物理取景尺度；不对模型分别归一化。
        bounds = np.concatenate([mesh.vertices @ rotation.T for mesh in meshes])
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        center = ((low + high) / 2) @ rotation
        span = float((high - low).max() * 1.12)
        return ComparisonResult(
            aligned_vertices, transform, rotation, center, span, origin,
            *sections, report, config.render_resolution, config.supersampling,
            config.profile_width, config.profile_height,
            **eye_profiles(data, meshes, rotation),
        )

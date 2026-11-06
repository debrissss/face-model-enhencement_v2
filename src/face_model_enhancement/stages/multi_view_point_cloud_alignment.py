"""多区域扫描器布局搜索、几何/颜色 ICP 和独立面部质量验证。"""
from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import open3d as o3d

from threadpoolctl import threadpool_limits

from ..config import MultiViewAlignmentConfig
from ..models import AlignedView, MeshData, MultiViewAlignment, MultiViewPointClouds, MultiViewRGB, PipelineData, PointCloudData
from .base import BaseStage


def sample_model(mesh: MeshData) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """与源工具相同的固定种子均匀表面采样，仅供全点距离诊断。"""
    import open3d as o3d
    from ..io.point_clouds import to_open3d
    vertices = to_open3d(PointCloudData(mesh.vertices, mesh.colors))
    if not len(mesh.faces):
        return vertices, vertices
    target = o3d.geometry.TriangleMesh()
    target.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    target.triangles = o3d.utility.Vector3iVector(mesh.faces)
    if mesh.colors is not None:
        target.vertex_colors = o3d.utility.Vector3dVector(mesh.colors)
    o3d.utility.random.seed(20261013)
    return vertices, target.sample_points_uniformly(number_of_points=min(180_000, max(80_000, len(mesh.vertices) * 2)))


def align_views(mesh: MeshData, inputs: MultiViewPointClouds, images: MultiViewRGB | None, config: MultiViewAlignmentConfig) -> MultiViewAlignment:
    from ..common.alignment.automatic import estimate_view
    from ..common.alignment.landmarks import FaceLocator
    from ..io.point_clouds import aligned_camera, from_open3d, to_open3d
    from ..utils.logging import get_logger
    logger = get_logger()
    vertices, surface = sample_model(mesh)
    locator = FaceLocator(config.model)
    results = {}
    try:
        for name, view in inputs.views.items():
            start = perf_counter()
            rgb = images.images.get(name) if images is not None else None
            logger.info("stage_id=multi_view_point_cloud_alignment view=%s status=start", name)
            try:
                aligned, transform, metrics, overlay = estimate_view(to_open3d(view.cloud), vertices, surface, name, view.camera, rgb, locator)
            except Exception:
                logger.exception("stage_id=multi_view_point_cloud_alignment view=%s status=failed algorithm_seconds=%.6f", name, perf_counter() - start)
                raise
            metrics["removed_nonfinite_points"] = view.removed_nonfinite
            results[name] = AlignedView(from_open3d(aligned), aligned_camera(view.camera, transform), transform.copy(), metrics, overlay)
            logger.info("stage_id=multi_view_point_cloud_alignment view=%s status=%s algorithm_seconds=%.6f", name, metrics["status"], perf_counter() - start)
    finally:
        locator.close()
    return MultiViewAlignment(results)


class MultiViewPointCloudAlignmentStage(BaseStage[MultiViewAlignmentConfig, MultiViewAlignment]):
    name = "multi_view_point_cloud_alignment"
    output_field = "alignment"

    def compute(self, data: PipelineData, config: MultiViewAlignmentConfig) -> MultiViewAlignment:
        if data.point_clouds is None:
            raise ValueError("multi_view_point_cloud_alignment requires --input_point_clouds.")
        with threadpool_limits(limits=config.threads):
            return align_views(data.mesh, data.point_clouds, data.multi_view_rgb, config)

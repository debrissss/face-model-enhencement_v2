"""计算原始顶点到参考三角面的距离，不要求对应顶点或相同拓扑。"""
from dataclasses import replace
import numpy as np
from face_model_enhancement.common.geometry import SurfaceVisibility
from face_model_enhancement.models import FloatArray, MeshData, PipelineData, SurfaceDistanceResult
from ..config import SurfaceDistanceConfig
from .base import BaseStage


def compute_surface_distances(mesh: MeshData, reference: MeshData) -> FloatArray:
    closest, _ = SurfaceVisibility(reference).closest_points(mesh.vertices)
    return np.linalg.norm(mesh.vertices - closest, axis=1)


class SurfaceDistanceStage(BaseStage[SurfaceDistanceConfig]):
    name = "surface_distance"

    def run(self, data: PipelineData, config: SurfaceDistanceConfig) -> PipelineData:
        distances = compute_surface_distances(data.mesh, data.reference)
        return replace(data, surface_distance=SurfaceDistanceResult(distances))

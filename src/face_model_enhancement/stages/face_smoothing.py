"""保护五官与固定边界，直接求解曲率和位置保持能量的三维表面整形。"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull

from .base import BaseStage
from ..common.geometry import compute_face_normals, sample_pixels
from ..common.registration import FACE_OVAL
from ..common.regions import NOSE_LANDMARKS, polygon_mask
from ..config import FaceSmoothingConfig
from ..models import BoolArray, FloatArray, IntArray, PipelineData, SmoothingResult


def nose_mask(data: PipelineData) -> BoolArray:
    """用鼻梁、鼻尖、鼻翼关键点的二维凸包圈定鼻部保护区。"""
    points = data.regions.landmarks[list(NOSE_LANDMARKS)]
    return polygon_mask(points[ConvexHull(points).vertices],
                        (data.camera.width, data.camera.height))


def build_domain(data: PipelineData, transition_mm: float, protect_nose: bool = False,
                 minimum_facing_cosine: float = .5, normal_scale_mm: float = 0.,
                 domain_padding_mm: float = 0.
                 ) -> tuple[FloatArray, IntArray, IntArray, BoolArray, FloatArray]:
    """合并完全重合点，在正面轮廓内固定全部非皮肤点及共享边界。"""
    assert data.regions is not None
    mesh, camera, regions = data.mesh, data.camera, data.regions
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    edges, counts = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                              axis=0, return_counts=True)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = sparse.csr_matrix((np.r_[lengths, lengths],
                               (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                              shape=(len(vertices), len(vertices)))
    outline = polygon_mask(regions.landmarks[list(FACE_OVAL)], (camera.width, camera.height))
    pixels, depth = camera.project(mesh.vertices)
    skin_vertices = sample_pixels(pixels, outline) & (depth > 0) & (regions.vertex_labels == 0)
    centers = mesh.vertices[mesh.faces].mean(1)
    face_pixels, face_depth = camera.project(centers)
    direction = normalize(camera.center - centers)
    if normal_scale_mm > 0:
        reference = compute_reference_surface(vertices, faces, normal_scale_mm)
        normals = compute_face_normals(reference, faces)
    else:
        normals = compute_face_normals(mesh.vertices, mesh.faces)
    facing = (normals * direction).sum(1)
    skin_faces = (sample_pixels(face_pixels, outline) & (face_depth > 0)
                  & (facing > minimum_facing_cosine) & (regions.face_labels == 0))
    if domain_padding_mm > 0:
        seeds = np.unique(inverse[sample_pixels(pixels, outline) & (depth > 0)])
        distance = dijkstra(graph, directed=False, indices=seeds, min_only=True)
        expanded = distance <= domain_padding_mm
        skin_vertices = expanded[inverse] & (depth > 0) & (regions.vertex_labels == 0)
        skin_faces = (expanded[faces].all(1) & (face_depth > 0)
                      & (facing > minimum_facing_cosine) & (regions.face_labels == 0))
    if protect_nose:
        protected = nose_mask(data)
        skin_vertices &= ~sample_pixels(pixels, protected)
        skin_faces &= ~sample_pixels(face_pixels, protected)
    fixed = np.zeros(len(vertices), dtype=bool)
    fixed[inverse[~skin_vertices]] = True
    fixed[np.unique(faces[~skin_faces])] = True
    fixed[np.unique(edges[counts != 2])] = True
    fixed |= np.bincount(faces.ravel(), minlength=len(vertices)) == 0
    distance = (dijkstra(graph, directed=False, indices=np.flatnonzero(fixed), min_only=True)
                if fixed.any() else np.full(len(vertices), np.inf))
    u = np.clip(distance / transition_mm, 0, 1)
    gate = u**3 * (10 + u * (-15 + 6 * u))
    gate[fixed] = 0
    return vertices, faces, inverse, fixed, gate


def normalize(vectors: FloatArray) -> FloatArray:
    """归一化方向，使用长度下限保留零向量。"""
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def build_fairing_operator(vertices: FloatArray, faces: IntArray
                           ) -> tuple[sparse.csr_matrix, FloatArray, FloatArray]:
    """构建余切刚度、顶点面积和面积加权法线；孤立点不参与求解。"""
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    rows, cols, values = [], [], []
    area = np.zeros(len(vertices))
    normals = np.zeros_like(vertices)
    for corner in range(3):
        a, b = (corner + 1) % 3, (corner + 2) % 3
        cotangent = ((triangles[:, a] - triangles[:, corner])
                     * (triangles[:, b] - triangles[:, corner])).sum(1) / double_area
        rows.extend((faces[:, a], faces[:, b]))
        cols.extend((faces[:, b], faces[:, a]))
        values.extend((cotangent * .5, cotangent * .5))
        np.add.at(area, faces[:, corner], double_area / 6)
        np.add.at(normals, faces[:, corner], cross)
    adjacency = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(len(vertices), len(vertices)))
    stiffness = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    return stiffness.tocsr(), area, normalize(normals)


def compute_reference_surface(vertices: FloatArray, faces: IntArray, scale_mm: float) -> FloatArray:
    """以隐式扩散生成只用于朝向估计的粗表面，不作为最终输出。

    余切算子考虑连接关系，避免把欧氏距离近但不连接的表面混合。
    """
    stiffness, area, _ = build_fairing_operator(vertices, faces)
    # 孤立点的算子行为零；为其保留原坐标，不参与表面的扩散。
    mass = sparse.diags(np.where(area > 0, area, 1.))
    center = vertices.mean(0)
    return spsolve((mass + scale_mm**2 * stiffness).tocsc(), mass @ (vertices - center)) + center


def fair_vertices(vertices: FloatArray, faces: IntArray, gate: FloatArray,
                  config: FaceSmoothingConfig) -> FloatArray:
    """直接求解曲率与原始位置保持的三维二次能量，不施加位移或面片朝向约束。

    固定区位移为零，过渡区通过更强的位置惩罚减小位移。
    刚度、面积及过渡权重在求解中保持不变。
    """
    stiffness, area, _ = build_fairing_operator(vertices, faces)
    free = np.flatnonzero(gate > 0)
    if len(free) == 0:
        return vertices.copy()
    inverse_area = np.zeros_like(area)
    np.divide(1., area, out=inverse_area, where=area > 0)
    # 固定区域提供边界值；不把其离散边界曲率当作需要消除的表面起伏。
    inverse_area[gate == 0] = 0
    mass_inverse = sparse.diags(inverse_area)
    # 在原点附近计算，避免模型绝对坐标带来额外消减误差。
    strength = config.fairing_scale_mm**4
    bending = stiffness.T @ mass_inverse @ stiffness
    system = strength * bending[free][:, free] + sparse.diags(area[free] / gate[free]**2)
    target = -strength * (bending[free] @ (vertices - vertices.mean(0)))
    movement = spsolve(system.tocsc(), target)
    positions = vertices.copy()
    positions[free] += movement
    return positions


def restore_result(original: FloatArray, filtered: FloatArray, inverse: IntArray,
                   fixed: BoolArray, gate: FloatArray) -> SmoothingResult:
    """恢复原始顶点编号，记录本阶段 XYZ 增量及非负位移长度。"""
    vertices = filtered[inverse]
    displacement = vertices - original
    return SmoothingResult(vertices, displacement, np.linalg.norm(displacement, axis=1),
                           fixed[inverse], gate[inverse])


class FaceSmoothingStage(BaseStage[FaceSmoothingConfig, SmoothingResult]):
    """串联域构建、皮肤整形及原始索引恢复，通过基类写回数据包。"""
    name = "face_smoothing"
    output_field = "smoothing"

    def compute(self, data: PipelineData, config: FaceSmoothingConfig) -> SmoothingResult:
        if data.regions is None:
            raise ValueError("face_smoothing requires region_detection results.")
        vertices, faces, inverse, fixed, gate = build_domain(
            data, config.transition_mm, config.protect_nose, config.minimum_facing_cosine,
            config.normal_scale_mm, config.domain_padding_mm)
        filtered = fair_vertices(vertices, faces, gate, config)
        return restore_result(data.mesh.vertices, filtered, inverse, fixed, gate)

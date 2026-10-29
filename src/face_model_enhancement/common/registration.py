"""使用未增强的正面表面计算刚体变换，保持原始毫米尺度。"""
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ResultComparisonConfig
from .geometry import SurfaceVisibility, compute_face_normals, sample_pixels
from ..models import BoolArray, FloatArray, MeshData, Metrics, PipelineData
from .regions import polygon_mask

# MediaPipe 的完整脸部轮廓，配准只使用其内部的正面且未增强区域。
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)


def registration_support(data: PipelineData) -> BoolArray:
    """选择原始脸部轮廓内、朝向相机且不属于眼口增强区的三角面。"""
    assert data.regions is not None
    mesh, camera = data.mesh, data.camera
    outline = polygon_mask(data.regions.landmarks[list(FACE_OVAL)], (camera.width, camera.height))
    centers = mesh.vertices[mesh.faces].mean(axis=1)
    uv, depth = camera.project(centers)
    direction = camera.center - centers
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    facing = np.sum(compute_face_normals(mesh.vertices, mesh.faces) * direction, axis=1) > .5
    return sample_pixels(uv, outline) & (depth > 0) & facing & (data.regions.face_labels == 0)


def sample_surface(mesh: MeshData, selected: BoolArray, count: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """按三角面面积确定性采样表面位置及对应面法线。"""
    triangles = mesh.vertices[mesh.faces[selected]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    if len(area) == 0:
        raise ValueError("No unchanged frontal face surface is available for registration.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(triangles), count, p=area / area.sum())
    # 平方根重参数化使采样在三角面内部按面积均匀分布。
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    points = np.sum(triangles[indices] * weights[:, :, None], axis=1)
    return points, cross[indices] / area[indices, None]


def fit_gt_rigid(source: MeshData, gt: MeshData, selected: BoolArray,
                 config: ResultComparisonConfig) -> tuple[FloatArray, Metrics]:
    """以原始网格表面拟合刚体 ICP，再取逆得到仅作用于 GT 的变换。"""
    points, normals = sample_surface(source, selected, config.sample_count, config.seed)
    target = SurfaceVisibility(gt)
    target_normals = compute_face_normals(gt.vertices, gt.faces)
    rotation = np.eye(3)
    translation = (gt.vertices.min(0) + gt.vertices.max(0) - source.vertices.min(0) - source.vertices.max(0)) / 2

    def correspondences() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
        """建立当前位姿的最近表面对应，并排除法线不兼容及远距离匹配。"""
        moved = points @ rotation.T + translation
        closest, faces = target.closest_points(moved)
        normal = target_normals[faces]
        distance = np.linalg.norm(moved - closest, axis=1)
        compatible = np.abs(np.sum((normals @ rotation.T) * normal, axis=1)) > .5
        if compatible.sum() < 6:
            raise ValueError("Rigid registration has insufficient compatible surface correspondences.")
        keep = compatible & (distance <= np.quantile(distance[compatible], config.trim_fraction))
        return moved, closest, normal, distance, keep

    _, _, _, initial_distance, initial_keep = correspondences()
    for iteration in range(config.max_iterations):
        moved, closest, normal, _, keep = correspondences()
        p, q, n = moved[keep], closest[keep], normal[keep]
        center = p.mean(axis=0)
        # 以采样中心为旋转基准，避免模型原点较远时旋转和平移的数值耦合。
        system = np.column_stack((np.cross(p - center, n), n))
        residual = np.sum((p - q) * n, axis=1)
        increment, _, rank, _ = np.linalg.lstsq(system, -residual, rcond=None)
        if rank < 6:
            raise ValueError("Rigid registration surface does not constrain all six pose parameters.")
        delta_rotation = Rotation.from_rotvec(increment[:3]).as_matrix()
        delta_translation = center - delta_rotation @ center + increment[3:]
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        if np.linalg.norm(increment) < 1e-6:
            break
    _, _, _, distance, keep = correspondences()
    # 求解过程中 source 只用于计算对应；取逆后只变换 GT，不移动 largeV 或 enhanced。
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = -rotation.T @ translation
    report = {
        "sample_count": config.sample_count, "iterations": iteration + 1,
        "retained_samples": int(keep.sum()), "scale": 1.,
        "initial_trimmed_mean_mm": float(initial_distance[initial_keep].mean()),
        "trimmed_mean_mm": float(distance[keep].mean()),
        "trimmed_median_mm": float(np.median(distance[keep])),
        "trimmed_p95_mm": float(np.quantile(distance[keep], .95)),
        "all_sample_mean_mm": float(distance.mean()),
    }
    return matrix, report

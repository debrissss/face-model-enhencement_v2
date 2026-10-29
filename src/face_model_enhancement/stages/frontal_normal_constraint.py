"""按法线图求解眼口区域顶点 Z。

模型坐标和高度差使用 mm，图像坐标使用像素；保留输入数组、顶点顺序和网格拓扑。
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .base import BaseStage
from ..common.regions import RIGHT_GLOBE, RIGHT_LID, LEFT_GLOBE, LEFT_LID, MOUTH
from ..config import FrontalNormalConfig
from ..common.geometry import extract_selected_region, get_fixed_vertices, sample_normals
from ..models import BoolArray, Camera, ConstraintResult, FloatArray, IntArray, MeshData, NormalMap, PipelineData


def gradient_operators(vertices: FloatArray, faces: IntArray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """把“每个顶点的 Z 高度”与“每个三角面的 X、Y 方向坡度”建立成线性关系。"""

    # 1. 取各面三个端点的 XY，计算有向面积的两倍作为梯度公式分母。
    # xy：(F, 3, 2)，按面组织的 A、B、C 的 XY 坐标，单位 mm。
    xy = vertices[faces, :2]
    # edge_ab_xy：(F, 2)，各面从 A 指向 B 的二维边向量，单位 mm。
    # edge_ac_xy：(F, 2)，各面从 A 指向 C 的二维边向量，单位 mm。
    edge_ab_xy, edge_ac_xy = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    # determinant：(F,)，两条投影边的行列式，即 XY 有向面积的两倍，单位 mm²。
    determinant = edge_ab_xy[:, 0] * edge_ac_xy[:, 1] - edge_ab_xy[:, 1] * edge_ac_xy[:, 0]

    # 2. 计算每个端点 Z 对 X/Y 坡度的贡献系数，分子分母保持同一顶点顺序。
    # gx：(F, 3)，各面 A、B、C 的 Z 对 X 坡度的贡献系数，单位 1/mm。
    # determinant[:, None]：(F, 1)，使每个面的三个系数都除以该面的行列式。
    gx = np.stack((xy[:, 1, 1] - xy[:, 2, 1], xy[:, 2, 1] - xy[:, 0, 1], xy[:, 0, 1] - xy[:, 1, 1]), axis=1) / determinant[:, None]
    # gy：(F, 3)，各面 A、B、C 的 Z 对 Y 坡度的贡献系数，单位 1/mm。
    gy = np.stack((xy[:, 2, 0] - xy[:, 1, 0], xy[:, 0, 0] - xy[:, 2, 0], xy[:, 1, 0] - xy[:, 0, 0]), axis=1) / determinant[:, None]

    # 3. 将系数装入面×顶点稀疏矩阵。
    # rows：(3*F,)，每个系数所属的面编号；每个面编号重复三次，作为矩阵行索引。
    rows = np.repeat(np.arange(len(faces)), 3)
    # shape：长度为 2 的元组 (F, N)，指定返回矩阵的行数和列数。
    shape = len(faces), len(vertices)
    # gx.ravel()、gy.ravel()：(3*F,)，按面顺序展开的系数值。
    # faces.ravel()：(3*F,)，各系数对应的顶点编号，作为矩阵列索引。
    # 两个 CSR 矩阵形状均为 (F, N)。
    return (sparse.coo_matrix((gx.ravel(), (rows, faces.ravel())), shape=shape).tocsr(),
            sparse.coo_matrix((gy.ravel(), (rows, faces.ravel())), shape=shape).tocsr())


def fit_height_patch(vertices: FloatArray, faces: IntArray, fixed: BoolArray, normal: NormalMap,
                     camera: Camera, config: FrontalNormalConfig) -> FloatArray:
    """在固定边界下，按目标法线迭代求解顶点 Z。

    返回独立持有的新坐标；眼球高度限制由后续步骤执行。
    """
    # 1. 由固定的 XY 坐标建立梯度算子，后续迭代复用。
    gx, gy = gradient_operators(vertices, faces)

    # 2. 确定可求解顶点并复制坐标；固定点消除整体高度平移自由度。
    free = ~fixed
    if not free.any():
        raise ValueError("The selected patch has no free interior vertices.")
    result = vertices.copy()

    # 3. 迭代读取当前面中心的法线并求解高度。
    for _ in range(config.iterations):
        # 3.1 用最新坐标重新投影采样，避免 Z 改变后仍读取旧像素的方向。
        targets = sample_normals(result[faces].mean(1), normal, camera)

        # 3.2 全部面参与法线误差；由法线计算目标坡度 dz/dx=-nx/nz、dz/dy=-ny/nz。
        x, y = gx, gy
        slopes = -targets[:, :2] / targets[:, 2, None]

        # 3.3 每个三角面等权，最小化 X/Y 坡度残差的平方和。
        matrix = x.T @ x + y.T @ y
        rhs = x.T @ slopes[:, 0] + y.T @ slopes[:, 1]

        # 3.4 从右端扣除固定点贡献，只求 free 顶点的新 Z，固定点继续用输入高度。
        result[free, 2] = spsolve(matrix[free][:, free], rhs[free] - matrix[free][:, fixed] @ vertices[fixed, 2])

    # 4. 将求解变化量按 strength 混合；只写自由顶点 Z，固定点及所有 X/Y 保持不变。
    result[free, 2] = vertices[free, 2] + config.strength * (result[free, 2] - vertices[free, 2])

    return result


class FrontalNormalConstraintStage(BaseStage[FrontalNormalConfig, ConstraintResult]):
    """编排眼口法线求解，由 BaseStage.run 写回新的数据包。"""
    name = "frontal_normal_constraint"
    output_field = "constraint"

    def compute(self, data: PipelineData, config: FrontalNormalConfig) -> ConstraintResult:
        """拟合眼口高度。

        返回坐标仍沿用原顶点编号；displacement_mm 为最终模型 Z 减输入模型 Z。
        """
        # 1. 确认区域检测已完成，读取原始网格和标签。
        if data.regions is None:
            raise ValueError("frontal_normal_constraint requires region_detection results.")
        mesh, regions = data.mesh, data.regions
        if data.smoothing is not None:
            mesh = MeshData(data.smoothing.vertices, mesh.faces)

        # 2. 复制结果坐标。
        vertices = mesh.vertices.copy()

        # 3. 在本阶段合并同眼的眼球与眼睑，分别定义右眼、左眼和嘴部求解面片。
        # 同眼内部不会仅因眼球/眼睑标签不同，就将交界处固定。
        patches = ((RIGHT_GLOBE, RIGHT_LID), (LEFT_GLOBE, LEFT_LID), (MOUTH,))

        # 4. 分别确定各面片的固定边界、拟合 Z，并按原编号写回结果副本。
        for labels in patches:
            selected = np.isin(regions.face_labels, labels)
            patch_vertex_indices, patch_face_indices = extract_selected_region(mesh, selected)
            fixed = get_fixed_vertices(mesh, selected, np.isin(regions.vertex_labels, labels), patch_vertex_indices, patch_face_indices)
            # 每个面片均从原始 mesh 坐标拟合，不以前一个面片的计算结果作为输入。
            fitted = fit_height_patch(mesh.vertices[patch_vertex_indices], patch_face_indices, fixed, data.normal, data.camera, config)
            vertices[patch_vertex_indices, 2] = fitted[:, 2]

        # 5. 返回最终坐标和相对输入的模型 Z 位移，不改上游网格或标签。
        return ConstraintResult(vertices, vertices[:, 2] - mesh.vertices[:, 2])

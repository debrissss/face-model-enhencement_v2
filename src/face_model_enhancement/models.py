"""定义阶段间共享的数据结构；网格采用模型坐标系，长度单位为毫米。"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]
Metrics = dict[str, float | int]


@dataclass(frozen=True)
class MeshData:
    """保存原始顶点坐标与三角面索引，不执行顶点合并或拓扑调整。"""
    vertices: FloatArray  # 形状为 (N,3)，沿用原始顶点排列。
    faces: IntArray  # 形状为 (F,3)，沿用原始三角面索引。


@dataclass(frozen=True)
class RGBImage:
    """保存原始 RGB 图像，供关键点检测和叠加可视化使用。"""
    pixels: ByteArray  # 形状为 (H,W,3)，对应原始 RGB 像素。


@dataclass(frozen=True)
class NormalMap:
    """保存相机坐标系下的单位法线图及有效像素掩码。"""
    vectors: FloatArray  # 形状为 (H,W,3)，表示相机坐标系下的单位法线。
    valid: BoolArray  # 形状为 (H,W)，标记可参与采样的像素。


@dataclass(frozen=True)
class DepthMap:
    """原生 RGB 像素平面的相对深度与有效观测掩码，尚未标定为毫米。"""
    raw: FloatArray
    valid: BoolArray


@dataclass(frozen=True)
class BrowDepthResult:
    """法线约束后仅沿模型 Z 追加眉毛起伏；位移仅表示本阶段的增量。"""
    vertices: FloatArray
    displacement_xyz_mm: FloatArray  # 保留 XYZ 存储格式，X、Y 位移均为零。
    displacement_mm: FloatArray  # 相机前向深度增量，正值表示远离相机。


@dataclass(frozen=True)
class Camera:
    """封装原始图像内参和模型到相机视图的刚性变换。"""
    model_to_view: FloatArray  # 形状为 (4,4)，使用列向量约定的刚性变换。
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int

    @property
    def center(self) -> FloatArray:
        """通过刚性变换的逆变换计算模型坐标系中的相机中心。"""
        # 对刚性变换 q=Rp+t 求逆，相机原点在模型坐标系中为 -Rᵀt。
        return -self.model_to_view[:3, 3] @ self.model_to_view[:3, :3]

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """将模型坐标投影到原始像素平面，并按相机负 Z 方向计算前向深度。"""
        # 先应用刚性变换；数组按行存储坐标，因此旋转矩阵使用转置。
        view = points @ self.model_to_view[:3, :3].T + self.model_to_view[:3, 3]
        depth = -view[:, 2]
        # 图像纵轴向下，相机视图的 Y 轴向上，因此纵向投影带负号。
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = np.column_stack((
                self.ppx + self.fx * view[:, 0] / depth,
                self.ppy - self.fy * view[:, 1] / depth,
            ))
        return pixels, depth


@dataclass(frozen=True)
class Regions:
    """保存关键点及二维、三维区域标签，三维标签始终对应原始网格索引。"""
    landmarks: FloatArray  # 形状为 (478,2)，位于原始图像像素系。
    image_labels: ByteArray  # 0 其他；1 右眼球；2 右眼睑；3 左眼球；4 左眼睑；5 嘴部；6 右眉；7 左眉。
    vertex_labels: ByteArray  # 形状为 (N,)，对应原始顶点索引。
    face_labels: ByteArray  # 形状为 (F,)，对应原始三角面索引。
    detection_rot90: int


@dataclass(frozen=True)
class SmoothingResult:
    """面部皮肤平滑结果；所有数组均对应原始顶点编号。"""
    vertices: FloatArray
    displacement_xyz_mm: FloatArray
    displacement_mm: FloatArray  # 非负欧氏位移长度，不是带符号 Z 位移。
    fixed: BoolArray
    gate: FloatArray


@dataclass(frozen=True)
class ConstraintResult:
    """保存高度约束后的顶点坐标和本阶段带符号 Z 位移，供法线和眼球高度阶段复用。"""
    vertices: FloatArray  # 形状为 (N,3)，沿用原始排列，仅允许 Z 改变。
    displacement_mm: FloatArray  # 形状为 (N,)，表示带符号的 Z 位移，单位为毫米。


@dataclass(frozen=True)
class ComparisonResult:
    """保存 GT 刚体配准、公共取景以及全脸和双眼的原始剖面。"""
    aligned_gt_vertices: FloatArray
    gt_to_largev: FloatArray  # 列向量约定的 4×4 刚体变换，不含尺度。
    frame_rotation: FloatArray  # 模型坐标到正面视图的旋转，行向量右乘其转置。
    render_center: FloatArray
    render_span_mm: float
    profile_origin: FloatArray  # 原始 largeV 上的鼻根位置。
    original_section: FloatArray  # 形状为 (K,2,2)，每段为两个端点的深度、高度。
    enhanced_section: FloatArray
    gt_section: FloatArray
    registration: Metrics
    render_resolution: int
    supersampling: int
    profile_width: int
    profile_height: int  # 每行的图像高度，完整剖面对比图共三行。
    eye_origins: FloatArray  # 原始 largeV 上的左、右眼中心，左右按被摄者区分。
    eye_half_heights_mm: FloatArray  # 各眼局部显示范围的半高，由原始眼宽确定。
    left_eye_original_section: FloatArray
    left_eye_enhanced_section: FloatArray
    left_eye_gt_section: FloatArray
    right_eye_original_section: FloatArray
    right_eye_enhanced_section: FloatArray
    right_eye_gt_section: FloatArray


@dataclass(frozen=True)
class PipelineData:
    """统一承载阶段输入和计算结果；阶段写回新容器，上游数组按只读约定共享。"""
    mesh: MeshData
    rgb: RGBImage
    normal: NormalMap
    camera: Camera
    source_name: str = "model"
    regions: Regions | None = None
    constraint: ConstraintResult | None = None
    gt: MeshData | None = None
    comparison: ComparisonResult | None = None
    depth: DepthMap | None = None
    brow_depth: BrowDepthResult | None = None
    smoothing: SmoothingResult | None = None

    eye_height: ConstraintResult | None = None

    @property
    def enhanced_mesh(self) -> MeshData:
        """返回已完成的最后一个几何结果，包括仅执行皮肤平滑的流水线。"""
        result = self.brow_depth if self.brow_depth is not None else self.eye_height
        if result is None:
            result = self.constraint
        if result is None:
            result = self.smoothing
        if result is None:
            raise ValueError("No enhancement result is available.")
        return MeshData(result.vertices, self.mesh.faces)

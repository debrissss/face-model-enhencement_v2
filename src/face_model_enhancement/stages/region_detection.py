"""依据关键点划分眉毛、可见眼球、整圈眼睑和嘴部，并映射到原始网格。"""
import numpy as np

from .base import BaseStage
from ..common.regions import (
    BROW_LOOPS, EYE_LOOPS, MOUTH_LOOP,
    RIGHT_GLOBE, RIGHT_LID, LEFT_GLOBE, LEFT_LID, MOUTH, RIGHT_BROW, LEFT_BROW, polygon_mask,
)
from ..config import RegionDetectionConfig
from ..common.landmarks import OpenCVFaceLandmarker
from ..models import ByteArray, Camera, FloatArray, PipelineData, Regions, RGBImage


def upright_rotation(camera: Camera) -> int:
    """根据模型竖直方向在图像中的投影，确定一次性的图像转正角度。"""
    rotation = camera.model_to_view[:3, :3]
    # 模型 +Y 向上，其反方向投影到图像后应与转正前的人脸向下方向一致。
    downward = np.array([-rotation[0, 1] * camera.fx, rotation[1, 1] * camera.fy])
    axes = np.array(((0, 1), (-1, 0), (0, -1), (1, 0)))
    # 在四个直角旋转方向中选择与投影方向最接近的一档，不重复试探检测。
    return int(np.argmax(axes @ downward))


def restore_pixels(points: FloatArray, rotation: int, width: int, height: int) -> FloatArray:
    """按照四分之一圈旋转关系，将检测坐标还原到原始图像像素系。"""
    # 像素坐标从 0 开始，旋转逆变换使用宽、高减 1 作为翻转基准。
    x, y = points.T
    if rotation == 1:
        return np.column_stack((width - 1 - y, x))
    if rotation == 2:
        return np.column_stack((width - 1 - x, height - 1 - y))
    if rotation == 3:
        return np.column_stack((y, height - 1 - x))
    return points.copy()


def detect_landmarks(rgb: RGBImage, camera: Camera, config: RegionDetectionConfig) -> tuple[FloatArray, int]:
    """在相机确定的直立图像上检测一次人脸，再还原原始关键点坐标。"""
    rotation = upright_rotation(camera)
    # 相机确定一次性转正方向，推理全程使用 OpenCV CPU，不初始化图形服务。
    image = np.ascontiguousarray(np.rot90(rgb.pixels, rotation))
    points = OpenCVFaceLandmarker(config.model).detect(image)
    # 网络裁剪已还原到直立图像，此处只需逆转相机确定的整图旋转。
    return restore_pixels(points, rotation, camera.width, camera.height), rotation


def image_regions(points: FloatArray, size: tuple[int, int], config: RegionDetectionConfig) -> ByteArray:
    """独立生成眉毛、眼球、整圈眼睑和嘴部掩码，再统一写入标签图。"""
    # 1. 直接填充左右眉和嘴部轮廓，生成各自的布尔掩码，不做外扩。
    contours = ((RIGHT_BROW, BROW_LOOPS[0]), (LEFT_BROW, BROW_LOOPS[1]), (MOUTH, MOUTH_LOOP))
    masks = {label: polygon_mask(points[list(loop)], size) for label, loop in contours}

    # 2. 分别生成左右眼掩码：原轮廓内部为眼球，外扩区域减去眼球为整圈眼睑。
    for (globe_label, lid_label), loop in zip(((RIGHT_GLOBE, RIGHT_LID), (LEFT_GLOBE, LEFT_LID)), EYE_LOOPS):
        contour = points[list(loop)]
        globe = polygon_mask(contour, size)
        expanded = polygon_mask(contour, size, config.eye_padding)
        masks[globe_label] = globe
        masks[lid_label] = expanded & ~globe

    # 3. 创建与原图尺寸一致的标签图，初始值 0 表示其他区域。
    # size 使用 (宽, 高)，NumPy 图像数组使用 (高, 宽)。
    width, height = size
    labels = np.zeros((height, width), dtype=np.uint8)

    # 4. 按眉毛、右眼、左眼、嘴部的顺序统一赋值；重叠时后写入的标签覆盖前面的标签。
    for label in (RIGHT_BROW, LEFT_BROW, RIGHT_GLOBE, RIGHT_LID, LEFT_GLOBE, LEFT_LID, MOUTH):
        labels[masks[label]] = label
    return labels


def project_regions(data: PipelineData, labels: ByteArray) -> tuple[ByteArray, ByteArray]:
    """直接投影原始顶点和面中心，向下取整采样二维标签。"""
    mesh = data.mesh

    # 投影原始顶点，采样结果与原始顶点编号一一对应。
    vertex_pixels, _ = data.camera.project(mesh.vertices)
    vertex_xy = np.floor(vertex_pixels).astype(np.int64)
    vertex_labels = labels[vertex_xy[:, 1], vertex_xy[:, 0]]

    # 独立投影三角面中心，采样结果与原始面编号一一对应。
    face_pixels, _ = data.camera.project(mesh.vertices[mesh.faces].mean(1))
    face_xy = np.floor(face_pixels).astype(np.int64)
    face_labels = labels[face_xy[:, 1], face_xy[:, 0]]
    return vertex_labels, face_labels


class RegionDetectionStage(BaseStage[RegionDetectionConfig, Regions]):
    """执行正面关键点检测及眉眼口区域的二维到三维映射。"""
    name = "region_detection"
    output_field = "regions"

    def compute(self, data: PipelineData, config: RegionDetectionConfig) -> Regions:
        """依次检测关键点、构建像素掩码和投影网格标签，汇总区域检测结果。"""
        # 二维检测先还原原始像素系，再与未变换的原始网格通过相机标定关联。
        points, rotation = detect_landmarks(data.rgb, data.camera, config)
        labels = image_regions(points, (data.camera.width, data.camera.height), config)
        vertex_labels, face_labels = project_regions(data, labels)
        return Regions(points, labels, vertex_labels, face_labels, rotation)

"""使用 OpenCV 的 CPU 后端执行 MediaPipe 模型，避免创建图形服务。"""
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.special import expit

from ..models import ByteArray, FloatArray


def face_anchors() -> FloatArray:
    """构造短距离人脸检测模型使用的 896 个固定大小锚点中心。"""
    anchors = []
    # 步长 8 的网格每格两个锚点；三个步长 16 的层合并后每格六个锚点。
    for grid_size, repeats in ((16, 2), (8, 6)):
        y, x = np.mgrid[:grid_size, :grid_size]
        centers = np.column_stack(((x.ravel() + .5) / grid_size, (y.ravel() + .5) / grid_size))
        anchors.append(np.repeat(centers, repeats, axis=0))
    return np.concatenate(anchors)


def weighted_face(raw: FloatArray, scores: FloatArray, anchors: FloatArray) -> FloatArray:
    """解码锚点预测，并对最高分人脸的重叠候选执行加权合并。"""
    probabilities = expit(scores.ravel())
    selected = probabilities >= .5
    if not selected.any():
        raise ValueError("No face detected in camera-oriented frontal RGB.")
    # 每行依次表示框中心、宽高以及六个二维关键点，尺度均按 128 像素输入归一化。
    detections = raw.reshape(-1, 16)[selected].astype(np.float64) / 128.
    centers = anchors[selected]
    detections[:, :2] += centers
    detections[:, 4:] = (detections[:, 4:].reshape(-1, 6, 2) + centers[:, None]).reshape(-1, 12)
    probabilities = probabilities[selected]
    best = detections[probabilities.argmax()]
    lower = detections[:, :2] - detections[:, 2:4] / 2
    upper = detections[:, :2] + detections[:, 2:4] / 2
    intersection = np.maximum(
        np.minimum(upper, best[:2] + best[2:4] / 2) - np.maximum(lower, best[:2] - best[2:4] / 2), 0,
    ).prod(axis=1)
    union = detections[:, 2:4].prod(axis=1) + best[2:4].prod() - intersection
    # 原任务只处理一张人脸，因此只需保留最高分框对应的首个加权抑制分组。
    overlapping = intersection / union > .5
    return np.average(detections[overlapping], axis=0, weights=probabilities[overlapping])


def crop_roi(image: ByteArray, center: FloatArray, size: FloatArray, angle: float,
             edge: int, border: int) -> tuple[ByteArray, FloatArray]:
    """将旋转矩形裁剪到模型输入平面，同时保留坐标还原所需的逆变换。"""
    corners = cv2.boxPoints((tuple(center), tuple(size), float(np.degrees(angle))))
    # 沿用 MediaPipe OpenCV 前处理的矩形角点顺序和像素尺度。
    destination = np.float32(((0, edge), (0, 0), (edge, 0), (edge, edge)))
    transform = cv2.getPerspectiveTransform(corners, destination)
    crop = cv2.warpPerspective(image, transform, (edge, edge), flags=cv2.INTER_LINEAR, borderMode=border)
    return crop, np.linalg.inv(transform)


def project_crop_points(points: FloatArray, inverse: FloatArray) -> FloatArray:
    """通过裁剪逆变换将模型输入平面的二维点还原到检测图像。"""
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64)[None], inverse)[0]


class OpenCVFaceLandmarker:
    """复用模型包中的人脸检测和 478 点模型，以明确指定的 CPU 后端执行推理。"""

    def __init__(self, model: Path):
        """从现有任务模型包加载两个 TFLite 网络，并准备固定锚点。"""
        # 直接从任务包读取权重，不生成临时模型文件，也不加载 MediaPipe 运行库。
        with ZipFile(model) as archive:
            self.detector = cv2.dnn.readNetFromTFLite(np.frombuffer(archive.read("face_detector.tflite"), dtype=np.uint8))
            self.landmarker = cv2.dnn.readNetFromTFLite(np.frombuffer(archive.read("face_landmarks_detector.tflite"), dtype=np.uint8))
        for network in (self.detector, self.landmarker):
            network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.anchors = face_anchors()

    def _detect_face(self, image: ByteArray) -> tuple[FloatArray, FloatArray, float]:
        """检测最高分人脸，并依据眼部关键点确定关键点模型的旋转裁剪框。"""
        height, width = image.shape[:2]
        side = float(max(height, width))
        # 检测输入保持纵横比：把原图放入居中的正方形区域，图外以黑色填充。
        crop, inverse = crop_roi(image, np.array([width / 2, height / 2]), np.array([side, side]),
                                 0., 128, cv2.BORDER_CONSTANT)
        self.detector.setInput(cv2.dnn.blobFromImage(crop, 1 / 127.5, mean=(127.5,) * 3))
        scores, boxes = self.detector.forward(("classificators", "regressors"))
        detection = weighted_face(boxes, scores, self.anchors)
        center = project_crop_points(detection[:2][None] * 128, inverse)[0]
        eyes = project_crop_points(detection[4:8].reshape(2, 2) * 128, inverse)
        direction = eyes[1] - eyes[0]
        # 沿用原任务图的 1.5 倍框扩展；眼睛连线确定图像坐标系中的旋转角。
        return center, detection[2:4] * side * 1.5, float(np.arctan2(direction[1], direction[0]))

    def detect(self, image: ByteArray) -> FloatArray:
        """在单张直立 RGB 中检测 478 个关键点，并还原到该图像的像素坐标。"""
        center, size, angle = self._detect_face(image)
        crop, inverse = crop_roi(image, center, size, angle, 256, cv2.BORDER_REPLICATE)
        # 关键点模型使用 [0,1] RGB；OpenCV 负责从像素排列转换到网络张量排列。
        self.landmarker.setInput(cv2.dnn.blobFromImage(crop, 1 / 255.))
        landmarks, presence = self.landmarker.forward(("Identity", "Identity_1"))
        if float(expit(presence.item())) < .5:
            raise ValueError("No face landmarks detected in the selected face crop.")
        # 这里只使用二维位置；模型预测的相对 Z 不参与网格高度约束。
        return project_crop_points(landmarks.reshape(478, 3)[:, :2], inverse)

"""共享的面部区域标签、关键点轮廓和多边形掩码。"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from ..models import FloatArray

# 沿用参考项目 predictors/eyes.py 中的 MediaPipe 眼睑关键点顺序。
EYE_LOOPS = (
    (133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155),
    (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382),
)
MOUTH_LOOP = (0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37)
BROW_LOOPS = ((70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
              (300, 293, 334, 296, 336, 285, 295, 282, 283, 276))
# 鼻梁、鼻尖和两侧鼻翼，用其二维凸包定义平滑阶段的保护区。
NOSE_LANDMARKS = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
                  98, 97, 48, 64, 129, 327, 326, 278, 294, 358)
# 每个标签同时确定部位和左右，左右按被摄者自身区分。
OTHER, RIGHT_GLOBE, RIGHT_LID, LEFT_GLOBE, LEFT_LID, MOUTH, RIGHT_BROW, LEFT_BROW = range(8)
BROW_LABEL_NAMES = {RIGHT_BROW: "right_brow", LEFT_BROW: "left_brow"}


def polygon_mask(points: FloatArray, size: tuple[int, int], padding: int = 0) -> np.ndarray:
    """填充关键点多边形；padding 大于零时按欧氏距离外扩指定像素数。"""
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon([tuple(p) for p in points], fill=255)
    mask = np.asarray(image) > 0
    if padding:
        mask = distance_transform_edt(~mask) <= padding
    return mask

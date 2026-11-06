"""绘制原始 RGB 上的区域标签与关键点叠加图。"""

import numpy as np
from PIL import Image, ImageDraw

from ..models import PipelineData

# 颜色及名称按统一标签编号排列，左右同类区域共享颜色。
COLORS = np.array([(180, 180, 180),
                   (0, 220, 220), (0, 128, 255),
                   (0, 220, 220), (0, 128, 255),
                   (240, 120, 80), (255, 210, 0), (255, 210, 0)], dtype=np.uint8)
LABEL_NAMES = ["other", "right_globe", "right_lid", "left_globe", "left_lid", "mouth", "right_brow", "left_brow"]


class RegionsVisualizer:
    """返回关键点与区域标签叠加图，不导出中间数组或网格。"""
    def render(self, data: PipelineData) -> Image.Image:
        """在独立图像副本上叠加区域颜色及关键点。"""
        regions = data.regions
        assert regions is not None
        # 只对目标区域混合颜色，在独立图像副本上绘制，保留背景细节。
        selected = regions.image_labels != 0
        rgb = data.rgb.pixels.copy()
        rgb[selected] = (.55 * rgb[selected] + .45 * COLORS[regions.image_labels[selected]]).astype(np.uint8)
        picture = Image.fromarray(rgb)
        draw = ImageDraw.Draw(picture)
        # 关键点使用原始图像坐标，可直接与区域叠加图对照。
        for x, y in regions.landmarks:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 0))
        draw.text((12, 12), "cyan: globe | blue: lid | orange: mouth | yellow: brow", fill="white", stroke_fill="black", stroke_width=1)
        return picture

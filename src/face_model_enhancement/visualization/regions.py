"""绘制原始 RGB 区域叠加图，并导出保留拓扑的着色网格和标签图例。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from ..models import PipelineData

# 颜色及名称按统一标签编号排列，左右同类区域共享颜色。
COLORS = np.array([(180, 180, 180),
                   (0, 220, 220), (0, 128, 255),
                   (0, 220, 220), (0, 128, 255),
                   (240, 120, 80), (255, 210, 0), (255, 210, 0)], dtype=np.uint8)
LABEL_NAMES = ["other", "right_globe", "right_lid", "left_globe", "left_lid", "mouth", "right_brow", "left_brow"]


class RegionsVisualizer:
    """负责关键点与区域标签的图像、网格和计数展示。"""
    def __init__(self, directory: Path):
        """记录由外层创建和管理的阶段产物目录。"""
        self.directory = directory

    def render(self, data: PipelineData) -> None:
        """叠加区域颜色及关键点，导出着色网格和各标签计数。"""
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
        picture.save(self.directory / "regions.png")
        # 着色操作使用独立的渲染网格，避免改动上游几何或显示属性。
        mesh = trimesh.Trimesh(data.mesh.vertices.copy(), data.mesh.faces.copy(), process=False)
        mesh.visual.face_colors = COLORS[regions.face_labels]
        mesh.export(self.directory / "regions.ply")
        # 标签计数与颜色图例随产物保存，便于检查区域覆盖范围。
        counts = {name: {"label": i, "pixels": int((regions.image_labels == i).sum()),
                         "vertices": int((regions.vertex_labels == i).sum()),
                         "faces": int((regions.face_labels == i).sum()), "rgb": COLORS[i].tolist()}
                  for i, name in enumerate(LABEL_NAMES)}
        (self.directory / "regions.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")


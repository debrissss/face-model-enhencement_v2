"""导出约束后网格，将所指定的带符号位移绘制到原始正面 RGB 图像。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from ..models import PipelineData


class DisplacementVisualizer:
    """负责增强网格、指定元数据和正面位移图的展示。"""
    def __init__(self, directory: Path, result_field: str = "constraint", displacement_label: str = "Z",
                 metadata_fields: tuple[str, ...] = (), unsigned: bool = False):
        """记录由外层创建和管理的阶段产物目录。"""
        self.directory = directory
        self.result_field = result_field
        self.displacement_label = displacement_label
        self.metadata_fields = metadata_fields
        self.unsigned = unsigned

    def render(self, data: PipelineData) -> None:
        """保存增强网格及指标，以红蓝色绘制所指定的带符号位移。"""
        result = getattr(data, self.result_field)
        assert result is not None
        # 导出网格使用结果坐标和原始面索引，显示过程不参与核心求解。
        mesh = trimesh.Trimesh(result.vertices.copy(), data.mesh.faces.copy(), process=False)
        mesh.export(self.directory / "mesh.ply")
        if self.metadata_fields:
            metrics = {field: getattr(result, field) for field in self.metadata_fields}
            (self.directory / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        displacement = result.displacement_mm
        scale = float(np.abs(displacement).max())
        # 位移绘制在原始顶点的投影位置，便于与输入 RGB 的部位直接对应。
        pixels, _ = data.camera.project(data.mesh.vertices)
        picture = Image.fromarray(data.rgb.pixels.copy())
        draw = ImageDraw.Draw(picture)
        for index in np.flatnonzero(displacement):
            x, y = pixels[index]
            # 用本次最大绝对位移归一化颜色：正向偏红，负向偏蓝，零附近偏白。
            value = displacement[index] / scale
            color = (255, int(255 * (1 - value)), int(255 * (1 - value))) if value > 0 else (int(255 * (1 + value)), int(255 * (1 + value)), 255)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        legend = (f"white 0 | red {scale:.4f}" if self.unsigned
                  else f"blue {-scale:.4f} | white 0 | red {scale:.4f}")
        draw.text((12, 12), f"{self.displacement_label} displacement (mm): {legend}", fill="white", stroke_fill="black", stroke_width=1)
        picture.save(self.directory / "displacement.png")

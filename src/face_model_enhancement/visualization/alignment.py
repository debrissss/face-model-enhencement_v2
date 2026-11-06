"""只读绘制对齐点云、模型和原生 RGB 定位诊断。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..models import PipelineData


class AlignmentVisualizer:
    def __init__(self, directory: Path):
        self.directory = directory

    def render(self, data: PipelineData) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        vertices = data.mesh.vertices
        center = np.median(vertices, axis=0)
        for name, view in data.alignment.views.items():
            if view.overlay is not None:
                Image.fromarray(view.overlay).save(self.directory / f"{name}_face_landmarks.png")
            points = view.cloud.points
            step = max(1, len(points) // 60000)
            points = points[::step]
            colors = view.cloud.colors[::step] if view.cloud.colors is not None else "tab:orange"
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, (horizontal, vertical, title) in zip(axes, ((0, 1, "Front"), (2, 1, "Side"), (0, 2, "Top"), (0, 1, "Full cloud"))):
                ax.scatter(vertices[::3, horizontal], vertices[::3, vertical], c="gray", alpha=.25, s=.5)
                ax.scatter(points[:, horizontal], points[:, vertical], c=colors, s=.2)
                if title != "Full cloud":
                    ax.set_xlim(center[horizontal] - 115, center[horizontal] + 115)
                    ax.set_ylim(center[vertical] - 115, center[vertical] + 115)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(title)
                ax.set_xlabel("mm")
                ax.set_ylabel("mm")
            fig.suptitle(f"{name}: {view.metrics['status']} (native RGB)")
            fig.tight_layout()
            fig.savefig(self.directory / f"{name}_alignment.png", dpi=160)
            plt.close(fig)

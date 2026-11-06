"""每次绘制一对模型的全脸、左眼和右眼剖面，返回一列三行图片。"""
import numpy as np
from PIL import Image

from ..models import ComparisonResult, FloatArray, PipelineData

MODEL_NAMES = ("largeV", "largeV-enhanced", "3dMD")
PROFILE_COLORS = ("#ef842b", "#c03a87", "#4c78d8")


def profile_rows(comparison: ComparisonResult) -> tuple[tuple[FloatArray, FloatArray, FloatArray], ...]:
    """返回全脸、左眼、右眼各模型的既有交线，不重新计算几何。"""
    return (
        (comparison.original_section, comparison.enhanced_section, comparison.gt_section),
        (comparison.left_eye_original_section, comparison.left_eye_enhanced_section, comparison.left_eye_gt_section),
        (comparison.right_eye_original_section, comparison.right_eye_enhanced_section, comparison.right_eye_gt_section),
    )


def compute_profile_limits(comparison: ComparisonResult) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """计算三组比较共享的坐标范围，保持各行等宽、横纵等毫米比例。"""
    points = np.concatenate([curve.reshape(-1, 2) for curve in profile_rows(comparison)[0]])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(high - low, 2.) * .06
    low, high = low - padding, high + padding
    aspect = (high[0] - low[0]) / (high[1] - low[1])
    eye_half_span = np.array([comparison.eye_half_heights_mm.max() * aspect,
                              comparison.eye_half_heights_mm.max()])
    return ((low, high), (-eye_half_span, eye_half_span), (-eye_half_span, eye_half_span))


class ProfileComparisonVisualizer:
    """只绘制指定的一对模型；多个比较列的拼接由 Stage 负责。"""
    def __init__(self, pair: tuple[int, int], limits: tuple[tuple[FloatArray, FloatArray], ...], width: int):
        self.pair = pair
        self.limits = limits
        self.width = width

    def render(self, data: PipelineData) -> Image.Image:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure

        comparison = data.comparison
        # profile_width 沿用完整三列合成图的宽度；各列按整数边界分配像素。
        width = self.width
        height = comparison.profile_height * 3
        figure = Figure(figsize=(width / 200, height / 200), dpi=200, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots(3, 1, squeeze=False)[:, 0]
        figure.subplots_adjust(left=.28, right=.88, bottom=.045, top=.90, hspace=.35)
        first, second = self.pair
        figure.text(.5, .985, f"{MODEL_NAMES[first]} vs {MODEL_NAMES[second]}",
                    ha="center", va="top", fontsize=11, weight="bold")
        labels = ("(a) Full face", "(b) Left eye", "(c) Right eye")
        for axis, curves, (low, high), label in zip(axes, profile_rows(comparison), self.limits, labels):
            for index in self.pair:
                axis.add_collection(LineCollection(curves[index], colors=PROFILE_COLORS[index],
                                                    linewidths=.3, label=MODEL_NAMES[index]))
            axis.set_xlim(low[0], high[0])
            axis.set_ylim(low[1], high[1])
            axis.set_aspect("equal", adjustable="box")
            box = axis.get_position()
            figure.text(.06, box.y1 + .050, label, ha="left", va="top", fontsize=12)
            axis.legend(loc="upper left", bbox_to_anchor=(box.x0, box.y1 + .030),
                        bbox_transform=figure.transFigure, borderaxespad=0, borderpad=0,
                        fontsize=8, frameon=False)
            axis.set_xlabel("(mm)", fontsize=9)
            axis.set_ylabel("(mm)", fontsize=9)
            axis.tick_params(labelsize=9)
            axis.grid(color="#e8eaed", linewidth=.5)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_color("#b6b8bc")
        canvas.draw()
        picture = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
        figure.clear()
        return picture

"""在 CPU 上渲染公共取景灰模，并绘制共享尺度的真实剖面对比。"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from ..common.geometry import SurfaceVisibility
from ..models import ByteArray, ComparisonResult, FloatArray, MeshData, PipelineData

MODEL_NAMES = ("largeV", "largeV-enhanced", "3dMD")
VIEWS = (("Front", (0., 0., 1.)), ("Left", (-1., 0., 0.)), ("Right", (1., 0., 0.)))
BACKGROUND = np.array([255, 255, 255], dtype=np.uint8)
MODEL_COLOR = np.array([.80, .80, .80])
PROFILE_COLORS = ("#ef842b", "#c03a87", "#4c78d8")


def gray_panel(mesh: MeshData, surface: SurfaceVisibility, normals: FloatArray,
               comparison: ComparisonResult, direction: tuple[float, float, float]) -> ByteArray:
    """以正交首交点和插值法线生成灰模，不创建任何 OpenGL 渲染窗口。"""
    resolution = comparison.render_resolution
    size = resolution * comparison.supersampling
    forward = np.asarray(direction) @ comparison.frame_rotation
    up = comparison.frame_rotation[1]
    right = np.cross(up, forward)
    pixels = ((np.arange(size) + .5) / size - .5) * comparison.render_span_mm
    x, y = np.meshgrid(pixels, -pixels)
    plane = comparison.render_center + x.reshape(-1, 1) * right + y.reshape(-1, 1) * up
    distance = comparison.render_span_mm * 2
    _, faces, coordinates = surface.intersections(plane + distance * forward, plane - distance * forward)
    visible = faces >= 0
    # VTK 的三角面参数坐标对应第二、第三个顶点的重心权重。
    uv = coordinates[visible, :2]
    weights = np.column_stack((1 - uv.sum(axis=1), uv))
    normal = np.sum(normals[mesh.faces[faces[visible]]] * weights[:, :, None], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    # 采用固定的侧上方主光和反侧补光，只计算漫反射，避免高光造成油亮观感。
    light = forward + .6 * right + .9 * up
    light /= np.linalg.norm(light)
    fill_light = forward - .7 * right + .15 * up
    fill_light /= np.linalg.norm(fill_light)
    diffuse = np.maximum(normal @ light, 0.)
    fill_diffuse = np.maximum(normal @ fill_light, 0.)
    shade = (.28 + .57 * diffuse + .15 * fill_diffuse)[:, None] * MODEL_COLOR
    image = np.broadcast_to(BACKGROUND, (size * size, 3)).copy()
    image[visible] = np.clip(shade * 255, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(image.reshape(size, size, 3)).resize(
        (resolution, resolution), Image.Resampling.LANCZOS))


def render_models(meshes: tuple[MeshData, MeshData, MeshData], comparison: ComparisonResult) -> Image.Image:
    """按原始、增强、GT 三行和正、左、右三列拼接公共取景灰模。"""
    from matplotlib import get_data_path

    resolution = comparison.render_resolution
    header = 32
    canvas = Image.new("RGB", (resolution * 3, (resolution + header) * 3), tuple(BACKGROUND))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 18)
    for row, mesh in enumerate(meshes):
        # 独立构建显示网格，只计算插值法线，不修改输入几何、顶点颜色或纹理。
        geometry = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        normals = np.asarray(geometry.vertex_normals)
        surface = SurfaceVisibility(mesh)
        for col, (title, direction) in enumerate(VIEWS):
            panel = gray_panel(mesh, surface, normals, comparison, direction)
            x, y = col * resolution, row * (resolution + header)
            canvas.paste(Image.fromarray(panel), (x, y + header))
            draw.text((x + 10, y + 6), f"{MODEL_NAMES[row]} | {title}", font=font, fill="#202020")
    return canvas


def render_profiles(comparison: ComparisonResult, path: Path) -> None:
    """按全脸、左眼和右眼三行绘制原始交线，保持各行同列等宽和统一配色。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    rows = (
        (comparison.original_section, comparison.enhanced_section, comparison.gt_section),
        (comparison.left_eye_original_section, comparison.left_eye_enhanced_section, comparison.left_eye_gt_section),
        (comparison.right_eye_original_section, comparison.right_eye_enhanced_section, comparison.right_eye_gt_section),
    )
    points = np.concatenate([curve.reshape(-1, 2) for curve in rows[0]])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(high - low, 2.) * .06
    low, high = low - padding, high + padding
    # 眼部窗口沿用全脸窗口的宽高比，保证等毫米比例下三行的绘图区宽度一致。
    aspect = (high[0] - low[0]) / (high[1] - low[1])
    half_height = comparison.eye_half_heights_mm.max()
    eye_half_span = np.array([half_height * aspect, half_height])
    limits = ((low, high), (-eye_half_span, eye_half_span), (-eye_half_span, eye_half_span))
    figure = Figure(figsize=(comparison.profile_width / 200, comparison.profile_height * 3 / 200),
                    dpi=200, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3)
    # 顶部只显示一组列标题，各行分别保留行标签和颜色图例。
    figure.subplots_adjust(left=.08, right=.97, bottom=.045, top=.90, wspace=.35, hspace=.35)
    labels = ("(a) Full face", "(b) Left eye", "(c) Right eye")
    for row_index, (row_axes, curves, (row_low, row_high), label) in enumerate(zip(axes, rows, limits, labels)):
        row_top = row_axes[0].get_position().y1
        figure.text(.06, row_top + .050, label, ha="left", va="top", fontsize=12)
        for axis, pair in zip(row_axes, ((0, 1), (0, 2), (1, 2))):
            for index in pair:
                collection = LineCollection(curves[index], colors=PROFILE_COLORS[index], linewidths=.3,
                                            label=MODEL_NAMES[index])
                axis.add_collection(collection)
            axis.set_xlim(row_low[0], row_high[0])
            axis.set_ylim(row_low[1], row_high[1])
            # 同一行的三组比较共享范围，纵横始终使用相同毫米尺度。
            axis.set_aspect("equal", adjustable="box")
            box = axis.get_position()
            center_x = (box.x0 + box.x1) / 2
            if row_index == 0:
                figure.text(center_x, .985, f"{MODEL_NAMES[pair[0]]} vs {MODEL_NAMES[pair[1]]}",
                            ha="center", va="top", fontsize=11, weight="bold")
            # 图例线条从实际绘图区左框开始，去掉图例自身的内外留白。
            axis.legend(loc="upper left", bbox_to_anchor=(box.x0, row_top + .030),
                        bbox_transform=figure.transFigure, borderaxespad=0, borderpad=0,
                        fontsize=8, frameon=False)
            axis.set_xlabel("(mm)", fontsize=9)
            axis.set_ylabel("(mm)", fontsize=9)
            axis.tick_params(labelsize=9)
            axis.grid(color="#e8eaed", linewidth=.5)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_color("#b6b8bc")
    figure.savefig(path, dpi=200)
    figure.clear()


class ComparisonVisualizer:
    """将已计算的配准与剖面数据输出为两张比较图片。"""

    def __init__(self, directory: Path):
        """保存由流水线和产物存储器管理的阶段目录。"""
        self.directory = directory

    def render(self, data: PipelineData) -> None:
        """绘制三行三列灰模和全脸、双眼剖面，不重新配准或调整模型位置。"""
        assert data.comparison is not None and data.constraint is not None and data.gt is not None
        comparison = data.comparison
        meshes = (data.mesh, data.enhanced_mesh,
                  MeshData(comparison.aligned_gt_vertices, data.gt.faces))
        render_models(meshes, comparison).save(self.directory / "render-comparison.png")
        render_profiles(comparison, self.directory / "midline-comparison.png")

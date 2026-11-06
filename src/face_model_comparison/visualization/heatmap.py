"""一次正交求交复用全部区域，未选中表面保持灰模。"""
from pathlib import Path
import numpy as np
import trimesh
from matplotlib import colormaps, get_data_path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw, ImageFont

from face_model_enhancement.common.geometry import SurfaceVisibility, sample_pixels
from face_model_enhancement.models import PipelineData
from face_model_enhancement.visualization.gray_model import compute_render_frame, MODEL_COLOR
from ..config import SurfaceDistanceConfig
from ..stages.comparison_regions import MESH_BANDS, select_surface_regions

TITLES = {"forehead": "Forehead", "eyes": "Eyes", "nose": "Nose", "mouth": "Perioral", "chin": "Chin", "cheeks": "Zygomatic & Medial Cheeks"}


def get_distance_colormap(name: str):
    """绿色代表小差异，黄/红代表更大差异；同时保留原有 matplotlib 色图选项。"""
    if name == "green_yellow_red":
        return LinearSegmentedColormap.from_list(name, ["#00ed20", "#a8ff00", "#ffff00", "#ff8c00", "#e00000"])
    return colormaps[name]


def render_region_panels(data: PipelineData, config: SurfaceDistanceConfig,
                         visibility_tolerance_mm: float = .1) -> list[Image.Image]:
    mesh = data.mesh
    regions = data.comparison_regions
    frame = compute_render_frame((mesh,), data.camera.model_to_view[:3, :3],
                                 config.render_resolution, config.supersampling)
    size = config.render_resolution * config.supersampling
    forward, up = frame.frame_rotation[2], frame.frame_rotation[1]
    right = np.cross(up, forward)
    offsets = ((np.arange(size) + .5) / size - .5) * frame.render_span_mm
    x, y = np.meshgrid(offsets, -offsets)
    plane = frame.render_center + x.reshape(-1, 1) * right + y.reshape(-1, 1) * up
    surface = SurfaceVisibility(mesh)
    hits, faces, coordinates = surface.intersections(plane + frame.render_span_mm * 2 * forward,
                                                    plane - frame.render_span_mm * 2 * forward)
    visible = faces >= 0
    triangles = mesh.faces[faces[visible]]
    uv = coordinates[visible, :2]
    weights = np.column_stack((1 - uv.sum(1), uv))
    # trimesh 显式持有副本，其缓存与法线计算不影响输入。
    geometry = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    normals = np.sum(np.asarray(geometry.vertex_normals)[triangles] * weights[:, :, None], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    light = forward + .6 * right + .9 * up
    light /= np.linalg.norm(light)
    fill = forward - .7 * right + .15 * up
    fill /= np.linalg.norm(fill)
    shade = (.28 + .57 * np.maximum(normals @ light, 0.) + .15 * np.maximum(normals @ fill, 0.))[:, None] * MODEL_COLOR
    base = np.full((size * size, 3), 255, dtype=np.uint8)
    base[visible] = np.clip(shade * 255, 0, 255).astype(np.uint8)
    values = np.sum(data.surface_distance.distance_mm[triangles] * weights, axis=1)
    heat = get_distance_colormap(config.colormap)(Normalize(config.color_min_mm, config.color_max_mm, clip=True)(values), bytes=True)[:, :3]
    heat = np.clip(heat * (.6 + .4 * shade[:, :1] / MODEL_COLOR[0]), 0, 255).astype(np.uint8)
    pixels, depth = data.camera.project(hits[visible])
    if regions.view_height_bounds_mm is None:
        projected = np.stack([sample_pixels(pixels, mask) & (depth > 0) for mask in regions.image_masks])
    else:
        projected = select_surface_regions(data, hits[visible], regions.image_masks,
                                           regions.names, regions.view_height_bounds_mm)
    camera_visible = np.zeros(visible.sum(), dtype=bool)
    mesh_bands = [regions.view_height_bounds_mm is not None and name in MESH_BANDS for name in regions.names]
    projected_rows = [index for index, is_band in enumerate(mesh_bands) if not is_band]
    if projected_rows:
        candidates = projected[projected_rows].any(0)
        camera_visible[candidates] = surface.visible(hits[visible][candidates], data.camera, visibility_tolerance_mm)
    panels = []
    for name, projected_mask, is_band in zip(regions.names, projected, mesh_bands):
        # 高度带只使用当前渲染射线的首个网格交点，原相机遮挡不再挖掉外侧面片。
        selected = projected_mask if is_band else projected_mask & camera_visible
        if not selected.any():
            raise ValueError(f"Region {name} has no visible pixels in the output view.")
        panel = base.copy()
        indices = np.flatnonzero(visible)[selected]
        panel[indices] = heat[selected]
        panels.append(Image.fromarray(panel.reshape(size, size, 3)).resize(
            (config.render_resolution, config.render_resolution), Image.Resampling.LANCZOS))
    return panels


class RegionalComparisonVisualizer:
    def __init__(self, config: SurfaceDistanceConfig, visibility_tolerance_mm: float = .1):
        self.config = config
        self.visibility_tolerance_mm = visibility_tolerance_mm

    def render(self, data: PipelineData) -> Image.Image:
        panels = render_region_panels(data, self.config, self.visibility_tolerance_mm)
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * panel.width, 0))
        return canvas


def compose_comparison(rows: list[Image.Image], names: list[str], regions: tuple[str, ...],
                       config: SurfaceDistanceConfig) -> Image.Image:
    """为原始区域面板添加行列标题与共享色标；单个区域也复用此布局。"""
    size = config.render_resolution
    header, row_header, footer = 36, 30, 92
    canvas = Image.new("RGB", (max(size * len(regions), 460), header + len(rows) * (size + row_header) + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, 16)
    small = ImageFont.truetype(font_path, 12)
    for index, region in enumerate(regions):
        draw.text((index * size + size // 2, 8), TITLES[region], font=font, anchor="mt", fill="#202020")
    for index, (row, name) in enumerate(zip(rows, names)):
        top = header + index * (size + row_header)
        draw.text((8, top + 4), name, font=font, fill="#202020")
        canvas.paste(row, (0, top + row_header))
    top = canvas.height - footer
    width = min(canvas.width - 80, 450)
    colors = get_distance_colormap(config.colormap)(np.linspace(0, 1, width), bytes=True)[:, :3]
    canvas.paste(Image.fromarray(np.repeat(colors[None], 14, axis=0)), (40, top + 7))
    draw.text((40, top + 23), f"{config.color_min_mm:g}", font=small, fill="black")
    draw.text((40 + width, top + 23), f"{config.color_max_mm:g} mm", font=small, anchor="rt", fill="black")
    draw.text((8, top + 43), "Unsigned closest-surface distance: mesh to reference (mm)", font=small, fill="black")
    draw.text((8, top + 62), "Colors clipped to range; gray = outside selected region", font=small, fill="black")
    return canvas

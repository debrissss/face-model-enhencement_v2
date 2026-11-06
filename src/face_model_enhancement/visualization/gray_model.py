"""以固定取景在 CPU 上渲染单个模型的正、左、右三视图。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from ..common.geometry import SurfaceVisibility
from ..models import ByteArray, ComparisonResult, FloatArray, MeshData, PipelineData

VIEWS = (("Front", (0., 0., 1.)), ("Left", (-1., 0., 0.)), ("Right", (1., 0., 0.)))
BACKGROUND = np.array([255, 255, 255], dtype=np.uint8)
MODEL_COLOR = np.array([.80, .80, .80])


@dataclass(frozen=True)
class ModelRenderFrame:
    """模型间共享的毫米取景和采样参数，不持有剖面或配准数据。"""
    frame_rotation: FloatArray
    render_center: FloatArray
    render_span_mm: float
    render_resolution: int = 512
    supersampling: int = 2


def compute_render_frame(meshes: tuple[MeshData, ...], rotation: FloatArray,
                         resolution: int = 512, supersampling: int = 2) -> ModelRenderFrame:
    """以给定模型的共同边界确定三视角共享取景，不分别缩放模型。"""
    bounds = np.concatenate([mesh.vertices @ rotation.T for mesh in meshes])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    return ModelRenderFrame(rotation, ((low + high) / 2) @ rotation,
                            float((high - low).max() * 1.12), resolution, supersampling)


def gray_panel(mesh: MeshData, surface: SurfaceVisibility, normals: FloatArray,
               comparison: ModelRenderFrame | ComparisonResult, direction: tuple[float, float, float]) -> ByteArray:
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



class GrayModelVisualizer:
    """每次返回一个模型的横排三视图，不导出模型或编排多个模型。"""
    def __init__(self, frame: ModelRenderFrame | ComparisonResult | None = None,
                 model_name: str = "largeV", enhanced: bool = False):
        self.frame = frame
        self.model_name = model_name
        self.enhanced = enhanced

    def render(self, data: PipelineData) -> Image.Image:
        from matplotlib import get_data_path

        mesh = data.enhanced_mesh if self.enhanced else data.mesh
        # 普通阶段以原始模型确定取景，避免各阶段单独归一化。
        frame = self.frame or compute_render_frame((data.mesh,), data.camera.model_to_view[:3, :3])
        resolution = frame.render_resolution
        header = 32
        canvas = Image.new("RGB", (resolution * 3, resolution + header), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 18)
        # 显式持有渲染副本，不让 trimesh 的缓存或法线计算影响上游数组。
        geometry = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
        normals = np.asarray(geometry.vertex_normals)
        surface = SurfaceVisibility(mesh)
        for column, (title, direction) in enumerate(VIEWS):
            panel = gray_panel(mesh, surface, normals, frame, direction)
            x = column * resolution
            canvas.paste(Image.fromarray(panel), (x, header))
            draw.text((x + 10, 6), f"{self.model_name} | {title}", font=font, fill="#202020")
        return canvas

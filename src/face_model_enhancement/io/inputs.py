"""分别读取命令行明确指定的网格、RGB、法线图和相机标定。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from ..models import ByteArray, Camera, DepthMap, MeshData, NormalMap, RGBImage


def read_mesh(path: Path) -> MeshData:
    # PLY 纹理接缝默认会拆分顶点，单独设置 process=False 仍不足以保留原始索引。
    """读取原始网格，关闭自动几何处理和纹理接缝拆点以保留索引。"""
    mesh = trimesh.load(path, process=False, force="mesh", maintain_order=True, fix_texture=False, skip_materials=True)
    return MeshData(np.asarray(mesh.vertices).copy(), np.asarray(mesh.faces).copy())


def read_rgb(path: Path) -> RGBImage:
    """将图像统一读取为 RGB，并持有脱离文件资源的像素副本。"""
    with Image.open(path) as image:
        return RGBImage(np.asarray(image.convert("RGB")).copy())


def decode_normal(pixels: ByteArray) -> NormalMap:
    """按 RGB=(相机法线+1)*127.5 解码并归一化，将全白像素视为无效背景。"""
    # 先识别白色背景，只对有效像素的解码向量进行归一化。
    valid = ~np.all(pixels == 255, axis=2)
    vectors = pixels.astype(np.float64) / 127.5 - 1.
    vectors[valid] /= np.linalg.norm(vectors[valid], axis=1, keepdims=True)
    # 无效像素置零，使后续插值同时依赖独立的有效性掩码。
    vectors[~valid] = 0.
    return NormalMap(vectors, valid)


def read_normal(path: Path) -> NormalMap:
    """读取法线 PNG，并依据约定的颜色编码生成法线图。"""
    with Image.open(path) as image:
        return decode_normal(np.asarray(image.convert("RGB")))


def read_camera(path: Path) -> Camera:
    """读取原始彩色图像内参和对齐后的模型到视图变换。"""
    parameters = json.loads(path.read_text(encoding="utf-8"))
    # 使用对齐后文件中的彩色相机内参，不使用红外内参或归一化网格变换。
    intrinsics = parameters["Cointrinsics"]
    return Camera(np.asarray(parameters["Transforms"]["Model_to_View"], dtype=np.float64),
                  **{key: intrinsics[key] for key in ("fx", "fy", "ppx", "ppy", "width", "height")})



def read_depth(path: Path) -> DepthMap:
    """读取原始浮点 NPZ 深度，不从预览图恢复数值。"""
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["raw"].astype(np.float64)
        valid = archive["valid"].astype(bool) & np.isfinite(raw)
    if raw.ndim != 2 or raw.shape != valid.shape:
        raise ValueError("Depth raw and valid must be matching two-dimensional arrays.")
    raw[~valid] = np.nan
    return DepthMap(raw, valid)

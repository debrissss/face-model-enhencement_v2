"""多视角点云读取、内存转换和 PLY/相机导出。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from ..models import MultiViewAlignment, MultiViewPointClouds, MultiViewRGB, PointCloudData, PointCloudView


def to_open3d(data: PointCloudData) -> o3d.geometry.PointCloud:
    """显式复制数组所有权，允许 Open3D 就地变换而不修改上游。"""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data.points)
    if data.colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(data.colors)
    if data.normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(data.normals)
    return cloud


def from_open3d(cloud: o3d.geometry.PointCloud) -> PointCloudData:
    """持有脱离 Open3D 对象的数组。"""
    return PointCloudData(np.asarray(cloud.points).copy(),
                          np.asarray(cloud.colors).copy() if cloud.has_colors() else None,
                          np.asarray(cloud.normals).copy() if cloud.has_normals() else None)


def read_point_clouds(directory: Path) -> MultiViewPointClouds:
    views = {}
    paths = sorted(directory.glob("*.ply"))
    if not paths:
        raise FileNotFoundError(f"No point clouds: {directory}")
    for path in paths:
        cloud = from_open3d(o3d.io.read_point_cloud(str(path)))
        finite = np.isfinite(cloud.points).all(axis=1)
        if not finite.any():
            raise ValueError(f"No finite points: {path}")
        cleaned = PointCloudData(cloud.points[finite],
                                 cloud.colors[finite] if cloud.colors is not None else None,
                                 cloud.normals[finite] if cloud.normals is not None else None)
        views[path.stem] = PointCloudView(cleaned, json.loads(path.with_suffix(".json").read_text()), int((~finite).sum()))
    return MultiViewPointClouds(views)


def read_multi_view_rgb(directory: Path) -> MultiViewRGB:
    images = {}
    for name in ("up", "down", "left", "right"):
        path = next((directory / f"{name}{ext}" for ext in (".png", ".jpg", ".jpeg")
                     if (directory / f"{name}{ext}").is_file()), None)
        if path is not None:
            with Image.open(path) as image:
                images[name] = np.asarray(image.convert("RGB")).copy()
    return MultiViewRGB(images)


def aligned_camera(camera: dict, transform: np.ndarray) -> dict:
    """保留厂商标定并记录配准位姿；不将 IR/Color 标定混入点云位姿。"""
    payload = deepcopy(camera)
    extrinsics = payload["Extrinsics"]
    old = np.eye(4)
    old[:3, :3] = np.asarray(extrinsics["rotation"]).reshape(3, 3)
    old[:3, 3] = extrinsics["translation"]
    payload["CoordinateSystem"] = {
        "point_cloud_frame": "Model", "source_point_cloud_frame": "View_camera",
        "projection_intrinsics": "Cointrinsics", "camera_forward_axis_in_view_frame": "-Z",
        "native_image_projection": "u=ppx-fx*x/z; v=ppy+fy*y/z (Cointrinsics)",
        "length_unit": "millimeter", "matrix_convention": "column_vector; p_target=matrix@[x,y,z,1]",
    }
    payload["Transforms"] = {"IR_to_Color": old.tolist(), "View_to_Model": transform.tolist(),
                             "Model_to_View": np.linalg.inv(transform).tolist()}
    return payload


def export_alignment(data: MultiViewAlignment, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, view in data.views.items():
        if not o3d.io.write_point_cloud(str(directory / f"{name}.ply"), to_open3d(view.cloud), write_ascii=False, compressed=False):
            raise OSError(f"Could not write point cloud: {name}")
        (directory / f"{name}.json").write_text(json.dumps(view.camera, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

"""提供网格面法线、原始像素采样和保持原始拓扑的表面可见性查询。"""
import numpy as np
from scipy.ndimage import map_coordinates
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import mutable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkGenericCell, vtkPolyData, vtkStaticCellLocator

from ..models import BoolArray, Camera, FloatArray, IntArray, MeshData, NormalMap


def compute_face_normals(vertices: FloatArray, faces: np.ndarray) -> FloatArray:
    """按三角面顶点绕序计算叉积，并归一化得到面法线。"""
    # 叉积方向由原始面顶点绕序决定，不额外修正网格朝向。
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def sample_pixels(pixels: FloatArray, image: np.ndarray) -> np.ndarray:
    """将图内投影坐标向下取整采样标签，图外或非有限坐标归为其他区域。"""
    h, w = image.shape[:2]
    inside = (np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    result = np.zeros(len(pixels), dtype=image.dtype)
    # 仅对图内坐标取整，避免将图外点强行夹到图像边缘。
    xy = np.floor(pixels[inside]).astype(np.int64)
    result[inside] = image[xy[:, 1], xy[:, 0]]
    return result


class SurfaceVisibility:
    """复用 VTK 静态单元定位器，以首个表面交点判断网格点是否可见。"""

    def __init__(self, mesh: MeshData):
        """按原始顶点和三角面构建 VTK 表面，并建立可复用的空间索引。"""
        # VTK 独立持有几何副本，构建空间索引时不修改流水线的原始数组。
        points = vtkPoints()
        points.SetData(numpy_to_vtk(mesh.vertices, deep=True))
        cells = vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, mesh.faces.size + 1, 3, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(mesh.faces.ravel(), deep=True))
        surface = vtkPolyData()
        surface.SetPoints(points)
        surface.SetPolys(cells)
        # 同一索引供顶点与面中心查询复用，避免重复构建加速结构。
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(surface)
        self.locator.BuildLocator()

    def visible(self, points: FloatArray, camera: Camera, tolerance: float) -> BoolArray:
        """从相机中心向目标点发射线段，检查首个交点与目标点的距离。"""
        result = np.zeros(len(points), dtype=bool)
        origin = camera.center
        t, sub_id, cell_id = mutable(0.), mutable(0), mutable(0)
        hit, pcoords, cell = [0., 0., 0.], [0., 0., 0.], vtkGenericCell()
        for i, point in enumerate(points):
            direction = point - origin
            # 将线段延伸到目标点稍后方，使位于端点的表面也能被求交命中。
            end = point + direction * (tolerance / np.linalg.norm(direction))
            found = self.locator.IntersectWithLine(origin, end, 1e-8, t, hit, pcoords, sub_id, cell_id, cell)
            # 首交点靠近目标才表示可见；只检查是否命中无法排除前景遮挡。
            result[i] = bool(found) and np.linalg.norm(np.asarray(hit) - point) < tolerance
        return result

    def closest_points(self, points: FloatArray) -> tuple[FloatArray, IntArray]:
        """查询最近表面位置及其原始三角面编号，供刚体配准复用。"""
        closest = np.empty_like(points, dtype=np.float64)
        faces = np.empty(len(points), dtype=np.int64)
        position = [0., 0., 0.]
        cell_id, sub_id, distance = mutable(0), mutable(0), mutable(0.)
        for index, point in enumerate(points):
            self.locator.FindClosestPoint(point, position, cell_id, sub_id, distance)
            closest[index] = position
            faces[index] = int(cell_id)
        return closest, faces

    def intersections(self, origins: FloatArray, ends: FloatArray) -> tuple[FloatArray, IntArray, FloatArray]:
        """查询一组线段的首个交点、面编号和面内参数坐标，未命中面编号为 -1。"""
        hits = np.zeros_like(origins, dtype=np.float64)
        faces = np.full(len(origins), -1, dtype=np.int64)
        coordinates = np.zeros_like(origins, dtype=np.float64)
        t, sub_id, cell_id = mutable(0.), mutable(0), mutable(0)
        hit, pcoords, cell = [0., 0., 0.], [0., 0., 0.], vtkGenericCell()
        for index, (origin, end) in enumerate(zip(origins, ends)):
            if self.locator.IntersectWithLine(origin, end, 1e-8, t, hit, pcoords, sub_id, cell_id, cell):
                hits[index] = hit
                faces[index] = int(cell_id)
                coordinates[index] = pcoords
        return hits, faces, coordinates


def sample_normals(points: FloatArray, normal: NormalMap, camera: Camera) -> FloatArray:
    """双线性采样、归一化并转换到模型坐标系，不筛选观测。"""
    uv, _ = camera.project(points)
    # 数组采样使用行、列顺序，对应图像坐标中的 v、u。
    coordinates = np.stack((uv[:, 1], uv[:, 0]))
    sampled = np.column_stack([
        map_coordinates(normal.vectors[:, :, c], coordinates, order=1, mode="constant", cval=0.)
        for c in range(3)
    ])
    # 插值后的法线先归一化，再用相机旋转的逆关系转到模型坐标系。
    lengths = np.linalg.norm(sampled, axis=1)
    np.divide(sampled, lengths[:, None], out=sampled, where=lengths[:, None] > 0)
    return sampled @ camera.model_to_view[:3, :3]


def extract_selected_region(mesh: MeshData, selected: BoolArray) -> tuple[IntArray, IntArray]:
    """返回选中区域的全局顶点索引及使用局部顶点索引的三角面数组。"""

    # 1. 收集选中面使用的全局顶点索引并去重。
    vertex_indices = np.unique(mesh.faces[selected])

    # 2. 建立全局顶点索引到局部顶点索引的映射。
    mapping = np.full(len(mesh.vertices), -1, dtype=np.int64)
    mapping[vertex_indices] = np.arange(len(vertex_indices))
    face_indices = mapping[mesh.faces[selected]]

    return vertex_indices, face_indices


def get_fixed_vertices(mesh: MeshData, selected: BoolArray, vertex_selected: BoolArray,
                       vertex_indices: IntArray, local_face_indices: IntArray) -> BoolArray:
    """返回与 vertex_indices 对应的新固定顶点掩码，用于固定求解时的 Z 坐标。"""

    # 1. 选中面的顶点标签不属于当前区域。
    fixed = ~vertex_selected[vertex_indices]

    # 2. 选中面的顶点同时被未选中面使用。
    outside = np.zeros(len(mesh.vertices), dtype=bool)
    outside[np.unique(mesh.faces[~selected])] = True
    fixed |= outside[vertex_indices]

    # 3. 顶点位于选中面片的开放边界。
    edge_indices = np.sort(local_face_indices[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edge_indices, count = np.unique(edge_indices, axis=0, return_counts=True)
    fixed[np.unique(edge_indices[count == 1])] = True

    return fixed

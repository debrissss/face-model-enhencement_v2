"""保存供三维查看的 PLY 网格，保留顶点排列和三角面拓扑。"""
from pathlib import Path

import trimesh

from ..models import MeshData
from .inputs import read_mesh


class PlyMeshSerializer:
    """只负责 PLY 网格读写，不选择阶段或输出目录。"""
    def save(self, data: MeshData, path: Path) -> None:
        """导出独立网格副本，不修改上游几何或颜色。"""
        mesh = trimesh.Trimesh(data.vertices.copy(), data.faces.copy(), process=False)
        mesh.export(path)

    def load(self, path: Path) -> MeshData:
        return read_mesh(path)


SERIALIZERS = {MeshData: PlyMeshSerializer()}

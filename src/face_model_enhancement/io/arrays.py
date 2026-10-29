"""按数据类型序列化含数组的数据类，不按阶段名称分派读写逻辑。"""
from dataclasses import fields
import json
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from ..models import SmoothingResult, BrowDepthResult, DepthMap, Camera, ComparisonResult, ConstraintResult, MeshData, NormalMap, Regions, RGBImage

Artifact = SmoothingResult | BrowDepthResult | DepthMap | MeshData | RGBImage | NormalMap | Camera | Regions | ConstraintResult | ComparisonResult
ArtifactT = TypeVar("ArtifactT", SmoothingResult, BrowDepthResult, DepthMap, MeshData, RGBImage, NormalMap, Camera, Regions, ConstraintResult, ComparisonResult)


class ArraySerializer(Generic[ArtifactT]):
    """将数据类的数组和非数组元数据共同存入 NPZ，并恢复为指定类型。"""
    def __init__(self, data_type: type[ArtifactT]):
        """绑定该序列化器负责恢复的数据类类型。"""
        self.data_type = data_type

    def save(self, data: ArtifactT, path: Path) -> None:
        """将数组直接压缩保存，其他字段编码为 JSON 元数据。"""
        arrays, metadata = {}, {}
        # 数组保留自身精度和形状；标量、字典等字段统一进入可读的 JSON 元数据。
        for field in fields(data):
            value = getattr(data, field.name)
            if isinstance(value, np.ndarray):
                arrays[field.name] = value
            else:
                metadata[field.name] = value
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
        np.savez_compressed(path, **arrays)

    def load(self, path: Path) -> ArtifactT:
        """在禁用 pickle 的条件下读取数组和元数据，再重建数据类。"""
        with np.load(path, allow_pickle=False) as archive:
            # 先恢复非数组字段，再合并数组字段，供绑定的数据类统一构造。
            values = json.loads(str(archive["metadata"]))
            values.update({key: archive[key] for key in archive.files if key != "metadata"})
        return self.data_type(**values)


# 序列化器按数据类型显式注册，新增阶段可以继续复用同一组读写实现。
SERIALIZERS = {
    SmoothingResult: ArraySerializer(SmoothingResult),
    DepthMap: ArraySerializer(DepthMap),
    BrowDepthResult: ArraySerializer(BrowDepthResult),
    MeshData: ArraySerializer(MeshData),
    RGBImage: ArraySerializer(RGBImage),
    NormalMap: ArraySerializer(NormalMap),
    Camera: ArraySerializer(Camera),
    Regions: ArraySerializer(Regions),
    ConstraintResult: ArraySerializer(ConstraintResult),
    ComparisonResult: ArraySerializer(ComparisonResult),
}
ARTIFACT_TYPES = {data_type.__name__: data_type for data_type in SERIALIZERS}


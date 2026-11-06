"""含嵌套数据类、视角字典与数组的数据类型序列化；不使用 pickle。"""
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np

from ..models import AlignedView, MultiViewAlignment, MultiViewPointClouds, MultiViewRGB, PointCloudData, PointCloudView
from .point_clouds import export_alignment

TYPES = {cls.__name__: cls for cls in (AlignedView, MultiViewAlignment, MultiViewPointClouds, MultiViewRGB, PointCloudData, PointCloudView)}


NestedT = TypeVar("NestedT", MultiViewAlignment, MultiViewPointClouds, MultiViewRGB)


class NestedArraySerializer(Generic[NestedT]):
    def save(self, data: NestedT, path: Path) -> None:
        arrays = {}

        def encode(value):
            if isinstance(value, np.ndarray):
                key = f"array_{len(arrays)}"
                arrays[key] = value
                return {"array": key}
            if is_dataclass(value):
                return {"type": type(value).__name__, "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
            if isinstance(value, dict):
                return {"dict": {k: encode(v) for k, v in value.items()}}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value

        metadata = encode(data)
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
        np.savez_compressed(path, **arrays)

    def load(self, path: Path) -> NestedT:
        with np.load(path, allow_pickle=False) as archive:
            def decode(value):
                if isinstance(value, dict):
                    if "array" in value:
                        return archive[value["array"]].copy()
                    if "type" in value:
                        return TYPES[value["type"]](**{k: decode(v) for k, v in value["fields"].items()})
                    return {k: decode(v) for k, v in value["dict"].items()}
                if isinstance(value, list):
                    return [decode(v) for v in value]
                return value
            return cast(NestedT, decode(json.loads(str(archive["metadata"]))))


class AlignmentSerializer(NestedArraySerializer[MultiViewAlignment]):
    def save(self, data: MultiViewAlignment, path: Path) -> None:
        super().save(data, path)
        export_alignment(data, path.parent / "multi_view_point_clouds_aligned")
        (path.parent / "alignment_report.json").write_text(json.dumps(
            {"passed": data.passed, "views": {name: view.metrics for name, view in data.views.items()}},
            ensure_ascii=False, indent=2, allow_nan=False) + "\n")

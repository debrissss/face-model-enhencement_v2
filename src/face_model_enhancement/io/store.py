"""组织阶段目录与产物清单，并通过显式类型注册表统一保存和加载。"""
from dataclasses import fields
import json
from pathlib import Path

from .arrays import ARTIFACT_TYPES, SERIALIZERS
from ..models import PipelineData


class ArtifactStore:
    """将数据包字段交给对应序列化器，并管理恢复数据包所需的类型清单。"""
    def save(self, data: PipelineData, stage_dir: Path) -> Path:
        """按字段类型保存已有数据，并记录来源名称和字段类型。"""
        stage_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"source_name": data.source_name, "fields": {}}
        for field in fields(data):
            value = getattr(data, field.name)
            if value is None or field.name == "source_name":
                continue
            # 存储入口按字段的实际类型分派，具体序列化器不关心阶段名称。
            SERIALIZERS[type(value)].save(value, stage_dir / f"{field.name}.npz")
            manifest["fields"][field.name] = type(value).__name__
        (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return stage_dir

    def load(self, stage_dir: Path) -> PipelineData:
        """根据产物清单选择序列化器，重新组装完整数据包。"""
        manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
        # 清单保存字段类型名称，加载时再通过静态注册表恢复对应类型。
        values = {name: SERIALIZERS[ARTIFACT_TYPES[type_name]].load(stage_dir / f"{name}.npz")
                  for name, type_name in manifest["fields"].items()}
        return PipelineData(source_name=manifest["source_name"], **values)


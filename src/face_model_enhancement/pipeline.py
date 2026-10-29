"""编排内存计算阶段，并按配置执行产物保存、加载和可视化。"""
from dataclasses import asdict
from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path
import shutil
from time import perf_counter
from uuid import uuid4

import yaml

from .config import EyeHeightConfig, FaceSmoothingConfig, BrowDepthConfig, FrontalNormalConfig, PipelineConfig, RegionDetectionConfig, ResultComparisonConfig
from .io.store import ArtifactStore
from .models import PipelineData
from .stages.brow_depth_constraint import BrowDepthConstraintStage
from .stages.eye_height_constraint import EyeHeightConstraintStage
from .stages.frontal_normal_constraint import FrontalNormalConstraintStage
from .stages.region_detection import RegionDetectionStage
from .stages.face_smoothing import FaceSmoothingStage
from .stages.result_comparison import ResultComparisonStage
from .utils.logging import close_logger, run_logger
from .visualization.displacement import DisplacementVisualizer
from .visualization.regions import RegionsVisualizer
from .visualization.registry import VisualizerRegistry
from .visualization.comparison import ComparisonVisualizer


class Pipeline:
    """管理单次运行目录、配置快照、阶段执行和分项耗时记录。"""
    def __init__(self, config: PipelineConfig, output: Path, input_paths: Mapping[str, Path] | None = None):
        """保存运行配置，并根据配置对象的类型装配阶段实例。"""
        self.config = config
        self.output = output
        self.input_paths = dict(input_paths or {})
        self.directory: Path | None = None
        self.stages = []
        # 配置层已检查执行顺序，这里只负责实例化对应的计算阶段。
        for spec in config.stages:
            if isinstance(spec.config, RegionDetectionConfig):
                self.stages.append((RegionDetectionStage(), spec.config))
            elif isinstance(spec.config, FaceSmoothingConfig):
                self.stages.append((FaceSmoothingStage(), spec.config))
            elif isinstance(spec.config, FrontalNormalConfig):
                self.stages.append((FrontalNormalConstraintStage(), spec.config))
            elif isinstance(spec.config, EyeHeightConfig):
                self.stages.append((EyeHeightConstraintStage(), spec.config))
            elif isinstance(spec.config, BrowDepthConfig):
                self.stages.append((BrowDepthConstraintStage(), spec.config))
            elif isinstance(spec.config, ResultComparisonConfig):
                self.stages.append((ResultComparisonStage(), spec.config))

    def _create_run(self, data: PipelineData) -> Path:
        """组合时间、输入名称和随机标识创建唯一运行目录，发生重名时重新生成。"""
        self.output.mkdir(parents=True, exist_ok=True)
        while True:
            # 时间、输入名称和随机后缀共同区分运行；创建目录仍采用排他操作。
            run_id = f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}-{data.source_name}-{uuid4().hex[:8]}"
            directory = self.output / run_id
            try:
                directory.mkdir()
                return directory
            except FileExistsError:
                continue

    def _snapshot(self, directory: Path) -> None:
        """同时保存原始配置、解析后的可重载配置及本次输入路径。"""
        snapshot = directory / "config_snapshot"
        (snapshot / "stages").mkdir(parents=True)
        (snapshot / "originals").mkdir()
        shutil.copyfile(self.config.path, snapshot / "originals/pipeline.yaml")
        resolved = {"visualize": self.config.visualize, "stages": []}
        for spec in self.config.stages:
            # 保留原始文本，同时将模型路径写为绝对路径，便于从快照重新加载。
            shutil.copyfile(spec.path, snapshot / "originals" / f"{spec.name}.yaml")
            parameters = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(spec.config).items()}
            resolved["stages"].append({"id": spec.name, "parameters": parameters})
            (snapshot / "stages" / f"{spec.name}.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "parameters": parameters}, sort_keys=False), encoding="utf-8")
        definition = {"schema_version": 1, "pipeline": {
            "stages": [{"id": spec.name, "config": f"stages/{spec.name}.yaml"} for spec in self.config.stages],
            "visualize": list(self.config.visualize),
        }}
        (snapshot / "pipeline.yaml").write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")
        (snapshot / "resolved.json").write_text(json.dumps(resolved, default=str, indent=2), encoding="utf-8")
        # 输入路径在阶段启动前落盘，使失败运行也保留定位问题所需的上下文。
        (snapshot / "inputs.json").write_text(
            json.dumps({name: str(path.resolve()) for name, path in self.input_paths.items()}, indent=2), encoding="utf-8")

    def run(self, data: PipelineData) -> PipelineData:
        """顺序更新内存数据包；仅对指定阶段保存并渲染产物，隔离可视化异常。"""
        self.directory = self._create_run(data)
        directory = self.directory
        logger = run_logger(directory.name, directory)
        store = ArtifactStore()
        visualizer = VisualizerRegistry({
            "region_detection": RegionsVisualizer(directory / "region_detection"),
            "face_smoothing": DisplacementVisualizer(directory / "face_smoothing", "smoothing", "XYZ length", unsigned=True),
            "frontal_normal_constraint": DisplacementVisualizer(directory / "frontal_normal_constraint"),
            "eye_height_constraint": DisplacementVisualizer(directory / "eye_height_constraint", "eye_height"),
            "brow_depth_constraint": DisplacementVisualizer(directory / "brow_depth_constraint", "brow_depth", "Camera depth"),
            "result_comparison": ComparisonVisualizer(directory / "result_comparison"),
        })
        logger.info("stage_id=pipeline status=start")
        try:
            self._snapshot(directory)
            for stage, config in self.stages:
                logger.info("stage_id=%s status=start", stage.name)
                start = perf_counter()
                try:
                    # 下一个阶段直接接收新的内存数据包，不读取阶段落盘文件。
                    data = stage.run(data, config)
                except Exception:
                    logger.exception("stage_id=%s status=failed algorithm_seconds=%.6f", stage.name, perf_counter() - start)
                    raise
                logger.info("stage_id=%s status=complete algorithm_seconds=%.6f", stage.name, perf_counter() - start)
                if stage.name in self.config.visualize:
                    try:
                        start = perf_counter()
                        try:
                            # 保存并重新加载供渲染使用；计算链继续持有原来的内存结果。
                            saved = store.save(data, directory / stage.name)
                            loaded = store.load(saved)
                        finally:
                            logger.info("stage_id=%s operation=serialization seconds=%.6f", stage.name, perf_counter() - start)
                        start = perf_counter()
                        try:
                            # 渲染单独计时，避免将绘图开销计入算法耗时。
                            visualizer.render(stage.name, loaded)
                        finally:
                            logger.info("stage_id=%s operation=visualization seconds=%.6f", stage.name, perf_counter() - start)
                    except Exception:
                        # 序列化或渲染失败只记录异常，不中断后续阶段的计算。
                        logger.exception("stage_id=%s status=visualization_failed", stage.name)
            logger.info("stage_id=pipeline status=complete")
            return data
        except Exception:
            logger.exception("stage_id=pipeline status=failed")
            raise
        finally:
            # 成功或失败均释放本次运行的日志文件句柄。
            close_logger(logger)


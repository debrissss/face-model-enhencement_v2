"""编排内存计算阶段，仅保存配置指定的可视化产物。"""
from dataclasses import asdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from time import perf_counter

import yaml

from .config import MultiViewAlignmentConfig, EyeHeightConfig, FaceSmoothingConfig, BrowDepthConfig, FrontalNormalConfig, PipelineConfig, RegionDetectionConfig, ResultComparisonConfig
from .io.store import ArtifactStore
from .models import PipelineData
from .stages.multi_view_point_cloud_alignment import MultiViewPointCloudAlignmentStage
from .stages.brow_depth_constraint import BrowDepthConstraintStage
from .stages.eye_height_constraint import EyeHeightConstraintStage
from .stages.frontal_normal_constraint import FrontalNormalConstraintStage
from .stages.region_detection import RegionDetectionStage
from .stages.face_smoothing import FaceSmoothingStage
from .stages.result_comparison import ResultComparisonStage
from .utils.logging import close_logger, run_logger
from .visualization.gray_model import GrayModelVisualizer
from .visualization.regions import RegionsVisualizer
from .visualization.registry import VisualizerRegistry


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
            if isinstance(spec.config, MultiViewAlignmentConfig):
                self.stages.append((MultiViewPointCloudAlignmentStage(), spec.config))
            elif isinstance(spec.config, RegionDetectionConfig):
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

    def _create_run(self) -> Path:
        """按分钟时间和配置任务名创建目录，同名任务追加递增编号。"""
        self.output.mkdir(parents=True, exist_ok=True)
        task = self.config.path.stem
        prefix = f"{datetime.now().astimezone():%y-%m-%d-%H-%M}-{task}"
        index = 1
        while True:
            run_id = prefix if index == 1 else f"{prefix}_{index:02d}"
            directory = self.output / run_id
            try:
                directory.mkdir()
                return directory
            except FileExistsError:
                index += 1

    def _snapshot(self, directory: Path) -> None:
        """保存解析后可重载的 Pipeline 和各阶段 YAML 配置。"""
        snapshot = directory / "config_snapshot"
        (snapshot / "stages").mkdir(parents=True)
        for spec in self.config.stages:
            # 模型路径写为绝对路径，便于从快照重新加载。
            parameters = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(spec.config).items()}
            (snapshot / "stages" / f"{spec.name}.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "parameters": parameters}, sort_keys=False), encoding="utf-8")
        definition = {"schema_version": 1, "pipeline": {
            "stages": [{"id": spec.name, "config": f"stages/{spec.name}.yaml"} for spec in self.config.stages],
            "visualize": list(self.config.visualize),
        }}
        (snapshot / "pipeline.yaml").write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")

    def run(self, data: PipelineData) -> PipelineData:
        """顺序更新内存数据包；结果对比在 Stage 内出图，对齐阶段不设可视化。"""
        self.directory = self._create_run()
        directory = self.directory
        logger = run_logger(directory.name, directory)
        group_directory = directory / data.source_name
        store = ArtifactStore()
        gray_model = GrayModelVisualizer(model_name="largeV-enhanced", enhanced=True)
        visualizer = VisualizerRegistry({
            "region_detection": RegionsVisualizer(),
            "face_smoothing": gray_model,
            "frontal_normal_constraint": gray_model,
            "eye_height_constraint": gray_model,
            "brow_depth_constraint": gray_model,
        })
        logger.info("stage_id=pipeline status=start group=%s inputs=%s", data.source_name,
                    {name: str(path.resolve()) for name, path in self.input_paths.items()})
        try:
            self._snapshot(directory)
            for stage, config in self.stages:
                logger.info("stage_id=%s status=start", stage.name)
                internal_rendering = isinstance(stage, ResultComparisonStage)
                if internal_rendering:
                    stage.directory = group_directory / stage.name if stage.name in self.config.visualize else None
                duration_field = "stage_seconds" if internal_rendering else "algorithm_seconds"
                start = perf_counter()
                try:
                    # 下一个阶段直接接收新的内存数据包，不读取阶段落盘文件。
                    data = stage.run(data, config)
                except Exception:
                    logger.exception("stage_id=%s status=failed %s=%.6f", stage.name, duration_field, perf_counter() - start)
                    raise
                logger.info("stage_id=%s status=complete %s=%.6f", stage.name, duration_field, perf_counter() - start)
                if stage.name in self.config.visualize and stage.name not in ("result_comparison", "multi_view_point_cloud_alignment"):
                    try:
                        stage_directory = group_directory / stage.name
                        stage_directory.mkdir(parents=True, exist_ok=True)
                        if stage.name != "region_detection":
                            start = perf_counter()
                            try:
                                store.save_mesh(data.enhanced_mesh, stage_directory)
                            finally:
                                logger.info("stage_id=%s operation=serialization seconds=%.6f", stage.name, perf_counter() - start)
                        start = perf_counter()
                        try:
                            picture = visualizer.render(stage.name, data)
                            filename = "regions.png" if stage.name == "region_detection" else "views.png"
                            picture.save(stage_directory / filename)
                        finally:
                            logger.info("stage_id=%s operation=visualization seconds=%.6f", stage.name, perf_counter() - start)
                    except Exception:
                        logger.exception("stage_id=%s status=visualization_failed", stage.name)
                if stage.name == "multi_view_point_cloud_alignment" and not data.alignment.passed:
                    failed = [name for name, view in data.alignment.views.items() if view.metrics["status"] != "review_candidate"]
                    raise ValueError(f"Low-confidence alignment blocks downstream stages: {failed}")
            logger.info("stage_id=pipeline status=complete")
            return data
        except Exception:
            logger.exception("stage_id=pipeline status=failed")
            raise
        finally:
            # 成功或失败均释放本次运行的日志文件句柄。
            close_logger(logger)


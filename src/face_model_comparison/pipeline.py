"""组织单次多组计算、配置快照和图片产物。"""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
import re
import yaml

from face_model_enhancement.io.inputs import read_camera, read_mesh, read_rgb
from face_model_enhancement.models import PipelineData
from face_model_enhancement.utils.logging import close_logger, run_logger
from face_model_enhancement.visualization.registry import VisualizerRegistry
from .config import ComparisonConfig, InputGroup, validate_groups
from .io.images import ArtifactStore
from .stages.comparison_regions import ComparisonRegionsStage
from .stages.surface_distance import SurfaceDistanceStage
from .visualization.heatmap import RegionalComparisonVisualizer, compose_comparison


def read_group(group: InputGroup) -> PipelineData:
    data = PipelineData(mesh=read_mesh(group.mesh), reference=read_mesh(group.reference),
                        rgb=read_rgb(group.rgb), camera=read_camera(group.camera), source_name=group.name)
    if data.rgb.pixels.shape != (data.camera.height, data.camera.width, 3):
        raise ValueError(f"group={group.name}: RGB and camera dimensions must match.")
    for name, mesh in (("mesh", data.mesh), ("reference", data.reference)):
        if not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"group={group.name}: {name} must contain a triangle surface.")
    return data


class Pipeline:
    def __init__(self, config: ComparisonConfig, output: Path, task: str | None = None):
        self.config = config
        self.output = output
        self.task = task
        self.directory: Path | None = None
        self.output_complete = True
        self.stages = ((ComparisonRegionsStage(), config.regions), (SurfaceDistanceStage(), config.distance))
        self.visualizer = VisualizerRegistry({"surface_distance": RegionalComparisonVisualizer(
            config.distance, config.regions.visibility_tolerance_mm)})
        self.store = ArtifactStore()
        if output.name not in ("artifacts", "artifacts_test"):
            raise ValueError("--output must name an artifacts or artifacts_test directory.")
        if output.name == "artifacts_test" and (task is None or re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", task) is None):
            raise ValueError("Development runs require an explicit --task using lowercase words, digits and underscores.")
        if output.name == "artifacts_test" and task in ("pipeline", "face_enhancement", "surface_distance", "comparison_regions"):
            raise ValueError("--task must describe the verification purpose, rather than the pipeline or stage name.")

    def _create_run(self) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        if self.output.name == "artifacts":
            numbers = [int(match.group(1)) for entry in self.output.iterdir()
                       if (match := re.fullmatch(r"run_(\d+)", entry.name))]
            index = max(numbers, default=0) + 1
            prefix = None
        else:
            index = 1
            prefix = f"{datetime.now().astimezone():%y-%m-%d-%H-%M}-{self.task}"
        while True:
            name = f"run_{index:04d}" if prefix is None else prefix + (f"_{index:02d}" if index > 1 else "")
            directory = self.output / name
            try:
                directory.mkdir()
                return directory
            except FileExistsError:
                index += 1

    def _snapshot(self, groups: list[InputGroup]) -> None:
        root = self.directory / "config_snapshot"
        (root / "stages").mkdir(parents=True)
        for name, config in (("comparison_regions", self.config.regions), ("surface_distance", self.config.distance)):
            values = {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
                      for key, value in asdict(config).items()}
            (root / "stages" / f"{name}.yaml").write_text(yaml.safe_dump({"schema_version": 1, "parameters": values}, sort_keys=False), encoding="utf-8")
        definition = {"schema_version": 1, "pipeline": {
            "stages": [{"id": stage.name, "config": f"stages/{stage.name}.yaml"} for stage, _ in self.stages],
            "visualize": list(self.config.visualize)}}
        (root / "pipeline.yaml").write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")
        inputs = [{key: str(value.resolve()) for key, value in asdict(group).items()} for group in groups]
        (root / "inputs.yaml").write_text(yaml.safe_dump({"schema_version": 1, "groups": inputs}, sort_keys=False), encoding="utf-8")

    def _save_images(self, images, logger, group: str) -> None:
        start = perf_counter()
        try:
            for path, picture in images:
                self.store.save(picture, path)
        finally:
            logger.info("group=%s stage_id=surface_distance operation=serialization serialization_seconds=%.6f", group, perf_counter() - start)

    def run(self, groups: list[InputGroup]) -> list[PipelineData]:
        validate_groups(groups)
        self.directory = self._create_run()
        self.output_complete = True
        logger = run_logger(self.directory.name, self.directory)
        results, rows, names = [], [], []
        group_name, stage_name = "all", "pipeline"
        logger.info("group=all stage_id=pipeline status=start")
        try:
            self._snapshot(groups)
            for group in groups:
                group_name, stage_name = group.name, "inputs"
                logger.info("group=%s stage_id=inputs status=start inputs=%s", group_name, asdict(group))
                data = read_group(group)
                for stage, config in self.stages:
                    stage_name = stage.name
                    logger.info("group=%s stage_id=%s status=start", group_name, stage.name)
                    start = perf_counter()
                    try:
                        data = stage.run(data, config)
                    finally:
                        logger.info("group=%s stage_id=%s operation=algorithm algorithm_seconds=%.6f", group_name, stage.name, perf_counter() - start)
                    logger.info("group=%s stage_id=%s status=complete", group_name, stage.name)
                results.append(data)
                if "surface_distance" not in self.config.visualize:
                    continue
                start = perf_counter()
                try:
                    row = self.visualizer.render("surface_distance", data)
                    regions = data.comparison_regions.names
                    pictures = []
                    for index, name in enumerate(regions):
                        size = self.config.distance.render_resolution
                        panel = row.crop((index * size, 0, (index + 1) * size, size))
                        pictures.append((self.directory / group_name / "surface_distance" / f"{name}.png",
                                         compose_comparison([panel], [group_name], (name,), self.config.distance)))
                    pictures.append((self.directory / group_name / "surface_distance" / "comparison.png",
                                     compose_comparison([row], [group_name], regions, self.config.distance)))
                except Exception:
                    self.output_complete = False
                    logger.exception("group=%s stage_id=surface_distance status=visualization_failed", group_name)
                    continue
                finally:
                    logger.info("group=%s stage_id=surface_distance operation=visualization visualization_seconds=%.6f", group_name, perf_counter() - start)
                try:
                    self._save_images(pictures, logger, group_name)
                except Exception:
                    self.output_complete = False
                    logger.exception("group=%s stage_id=surface_distance status=serialization_failed", group_name)
                    continue
                rows.append(row)
                names.append(group_name)
            # 任一组失败时不生成伪装成完整结果的批量总图。
            if rows and self.output_complete:
                start = perf_counter()
                try:
                    total = compose_comparison(rows, names, self.config.regions.regions, self.config.distance)
                except Exception:
                    self.output_complete = False
                    logger.exception("group=all stage_id=surface_distance status=visualization_failed")
                finally:
                    logger.info("group=all stage_id=surface_distance operation=visualization visualization_seconds=%.6f", perf_counter() - start)
                if self.output_complete:
                    try:
                        self._save_images([(self.directory / "comparison.png", total)], logger, "all")
                    except Exception:
                        self.output_complete = False
                        logger.exception("group=all stage_id=surface_distance status=serialization_failed")
            logger.info("group=all stage_id=pipeline status=%s", "complete" if self.output_complete else "output_incomplete")
            return results
        except Exception:
            logger.exception("group=%s stage_id=%s status=failed", group_name, stage_name)
            raise
        finally:
            close_logger(logger)

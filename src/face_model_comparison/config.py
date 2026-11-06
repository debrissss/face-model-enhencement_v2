"""独立对比流程的类型化 YAML 配置与输入清单。"""
from dataclasses import dataclass
from pathlib import Path
import math
from matplotlib import colormaps

from face_model_enhancement.config import read_yaml

REGIONS = ("forehead", "eyes", "nose", "mouth", "chin", "cheeks")


@dataclass(frozen=True)
class ComparisonRegionsConfig:
    model: Path
    regions: tuple[str, ...] = REGIONS
    eye_padding_ratio: float = .16
    mouth_padding_ratio: float = .1
    visibility_tolerance_mm: float = .1
    nose_padding_ratio: float = .08

    def __post_init__(self):
        if not self.regions or len(set(self.regions)) != len(self.regions) or not set(self.regions) <= set(REGIONS):
            raise ValueError(f"regions must be a nonempty ordered selection from {REGIONS}.")
        for value in (self.eye_padding_ratio, self.mouth_padding_ratio, self.nose_padding_ratio):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Region padding ratios must be finite and nonnegative.")
        if not 0 < self.visibility_tolerance_mm < math.inf:
            raise ValueError("visibility_tolerance_mm must be finite and positive.")


@dataclass(frozen=True)
class SurfaceDistanceConfig:
    color_min_mm: float = 0.
    color_max_mm: float = 3.
    colormap: str = "green_yellow_red"
    render_resolution: int = 512
    supersampling: int = 2

    def __post_init__(self):
        if not 0 <= self.color_min_mm < self.color_max_mm < math.inf:
            raise ValueError("Require 0 <= color_min_mm < color_max_mm < infinity.")
        if self.colormap != "green_yellow_red" and self.colormap not in colormaps:
            raise ValueError(f"Unknown colormap: {self.colormap}")
        if any(type(v) is not int or v < 1 for v in (self.render_resolution, self.supersampling)):
            raise ValueError("Render dimensions must be positive integers.")


@dataclass(frozen=True)
class ComparisonConfig:
    regions: ComparisonRegionsConfig
    distance: SurfaceDistanceConfig
    visualize: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class InputGroup:
    directory: Path
    mesh: Path
    reference: Path
    rgb: Path
    camera: Path

    @property
    def name(self) -> str:
        return self.directory.name


def load_config(path: Path) -> ComparisonConfig:
    path = path.resolve()
    definition = read_yaml(path)["pipeline"]
    stages = definition["stages"]
    if [stage["id"] for stage in stages] != ["comparison_regions", "surface_distance"]:
        raise ValueError("Expected comparison_regions followed by surface_distance.")
    parameters = []
    for stage in stages:
        stage_path = (path.parent / stage["config"]).resolve()
        values = read_yaml(stage_path)["parameters"].copy()
        if stage["id"] == "comparison_regions":
            values["model"] = (stage_path.parent / values["model"]).resolve()
            values["regions"] = tuple(values.get("regions", REGIONS))
        parameters.append(values)
    visualize = tuple(definition.get("visualize", ()))
    if not set(visualize) <= {"surface_distance"}:
        raise ValueError("Only surface_distance has a comparison visualizer.")
    return ComparisonConfig(ComparisonRegionsConfig(**parameters[0]), SurfaceDistanceConfig(**parameters[1]), visualize, path)


def load_manifest(path: Path) -> list[InputGroup]:
    path = path.resolve()
    groups = []
    for item in read_yaml(path)["groups"]:
        # 所有相对输入路径均以清单所在目录为基准，不以 group directory 为基准。
        groups.append(InputGroup(**{key: (path.parent / item[key]).resolve()
                                    for key in ("directory", "mesh", "reference", "rgb", "camera")}))
    validate_groups(groups)
    return groups


def validate_groups(groups: list[InputGroup]) -> None:
    names = [group.name for group in groups]
    if not names or any(name in ("", ".", "..", "config_snapshot") for name in names) or len(set(names)) != len(names):
        raise ValueError("Input groups must have nonempty, unique directory names (excluding config_snapshot).")
    for group in groups:
        for name in ("mesh", "reference", "rgb", "camera"):
            path = getattr(group, name)
            if not path.is_file():
                raise FileNotFoundError(f"group={group.name} input_{name}: {path}")

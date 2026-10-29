"""加载和校验 YAML 配置，并按引用所在文件解析相对路径。"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RegionDetectionConfig:
    """定义关键点模型和眼部欧氏距离外扩范围。"""
    model: Path
    eye_padding: int = 18

    def __post_init__(self) -> None:
        """检查眼部外扩像素数。"""
        if not isinstance(self.eye_padding, int) or self.eye_padding < 0:
            raise ValueError("Eye padding must be nonnegative.")


@dataclass(frozen=True)
class FrontalNormalConfig:
    """定义法线积分参数。"""
    iterations: int = 6
    strength: float = 1.0

    def __post_init__(self) -> None:
        """检查求解参数的数值范围以及迭代次数的整数约束。"""
        if not isinstance(self.iterations, int) or self.iterations < 1 or not 0 <= self.strength <= 1:
            raise ValueError("Invalid iterations, strength or normal Z threshold.")


@dataclass(frozen=True)
class EyeHeightConfig:
    """定义眼球相对局部眼睑包络的向内间距，单位 mm。"""
    eye_height_gap_mm: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.eye_height_gap_mm < float("inf"):
            raise ValueError("Eye height gap must be finite and nonnegative.")


@dataclass(frozen=True)
class ResultComparisonConfig:
    """定义刚体配准、灰模渲染和正中剖面对比的计算与图像尺寸。"""
    sample_count: int = 20000
    max_iterations: int = 60
    trim_fraction: float = .85
    seed: int = 1
    render_resolution: int = 512
    supersampling: int = 2
    profile_width: int = 2400
    profile_height: int = 1200

    def __post_init__(self) -> None:
        """校验采样、迭代和图像尺寸。"""
        integers = (self.sample_count, self.max_iterations, self.render_resolution,
                    self.supersampling, self.profile_width, self.profile_height)
        if any(not isinstance(value, int) or value < 1 for value in integers):
            raise ValueError("Comparison counts and image dimensions must be positive integers.")
        if self.sample_count < 6 or not 0 < self.trim_fraction <= 1:
            raise ValueError("Invalid registration sample count or trim fraction.")


@dataclass(frozen=True)
class BrowDepthConfig:
    """眉毛起伏迁移强度与本阶段欧氏位移预算。"""
    strength: float = .6
    maximum_displacement_mm: float = 3.

    def __post_init__(self) -> None:
        if not 0 <= self.strength <= 1:
            raise ValueError("Brow strength must lie in [0,1].")
        if not 0 < self.maximum_displacement_mm < float("inf"):
            raise ValueError("Brow displacement budget must be finite and positive.")


@dataclass(frozen=True)
class FaceSmoothingConfig:
    """有界三维表面整形参数，长度及位移预算均使用毫米。"""
    protect_nose: bool = True
    fairing_scale_mm: float = 4.
    normal_scale_mm: float = 3.
    domain_padding_mm: float = 6.
    minimum_facing_cosine: float = -.4
    maximum_displacement_mm: float = 2.
    transition_mm: float = 2.

    def __post_init__(self) -> None:
        if not -1 <= self.minimum_facing_cosine < 1:
            raise ValueError('Smoothing facing cosine must lie in [-1,1).')
        if not 0 <= self.normal_scale_mm < float('inf'):
            raise ValueError('Normal scale must be finite and nonnegative.')
        if not 0 <= self.domain_padding_mm < float('inf'):
            raise ValueError('Domain padding must be finite and nonnegative.')
        lengths = (self.maximum_displacement_mm, self.transition_mm, self.fairing_scale_mm)
        if any(not 0 < value < float('inf') for value in lengths):
            raise ValueError('Smoothing scales must be finite and positive.')


StageConfig = RegionDetectionConfig | FaceSmoothingConfig | FrontalNormalConfig | EyeHeightConfig | BrowDepthConfig | ResultComparisonConfig


@dataclass(frozen=True)
class StageSpec:
    """关联阶段名称、明确类型的阶段配置及其来源文件。"""
    name: str
    config: StageConfig
    path: Path


@dataclass(frozen=True)
class PipelineConfig:
    """保存已解析的阶段顺序、可视化阶段名单和主配置来源。"""
    stages: tuple[StageSpec, ...]
    visualize: tuple[str, ...]
    path: Path


def read_yaml(path: Path) -> dict:
    """读取 YAML 文档并检查配置模式版本。"""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document["schema_version"] != 1:
        raise ValueError(f"Unsupported schema_version: {path}")
    return document


def load_config(path: Path) -> PipelineConfig:
    """组装阶段配置对象，校验依赖顺序及可视化阶段名单。"""
    path = path.resolve()
    pipeline = read_yaml(path)["pipeline"]
    specs = []
    types = {"region_detection": RegionDetectionConfig, "frontal_normal_constraint": FrontalNormalConfig,
             "result_comparison": ResultComparisonConfig, "brow_depth_constraint": BrowDepthConfig,
             "face_smoothing": FaceSmoothingConfig, "eye_height_constraint": EyeHeightConfig}
    # 主配置中的阶段路径以主配置目录为基准，阶段内部的模型路径另行解析。
    for stage in pipeline["stages"]:
        stage_path = (path.parent / stage["config"]).resolve()
        parameters = read_yaml(stage_path)["parameters"]
        name = stage["id"]
        if name == "region_detection":
            # 模型路径属于阶段配置，应以阶段 YAML 所在目录为基准。
            parameters["model"] = (stage_path.parent / parameters["model"]).resolve()
        specs.append(StageSpec(name, types[name](**parameters), stage_path))
    names = [stage.name for stage in specs]
    # 平滑、眼球高度和眉毛阶段可省略，保留旧配置；接受各合法执行序列的非空前缀。
    orders = [["region_detection", *smoothing, "frontal_normal_constraint", *eye_height, *brow, "result_comparison"]
              for smoothing in ([], ["face_smoothing"])
              for eye_height in ([], ["eye_height_constraint"])
              for brow in ([], ["brow_depth_constraint"])]
    if not names or not any(names == order[:len(names)] for order in orders):
        raise ValueError("Stages must follow region_detection, optional face_smoothing, frontal_normal_constraint, "
                         "optional eye_height_constraint, optional brow_depth_constraint, result_comparison.")
    visualize = tuple(pipeline.get("visualize", []))
    # 可视化名单只能引用本次已配置的阶段。
    if not set(visualize) <= set(names):
        raise ValueError("pipeline.visualize contains an unconfigured stage.")
    return PipelineConfig(tuple(specs), visualize, path)


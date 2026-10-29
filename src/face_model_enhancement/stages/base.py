"""统一阶段计算和数据包写回约定，子类仅实现核心计算。"""
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Generic, Literal, TypeVar

from ..config import StageConfig
from ..models import BrowDepthResult, ComparisonResult, ConstraintResult, PipelineData, Regions, SmoothingResult

ConfigT = TypeVar("ConfigT", bound=StageConfig)
ResultT = TypeVar("ResultT", Regions, ConstraintResult, ComparisonResult, BrowDepthResult, SmoothingResult)


class BaseStage(ABC, Generic[ConfigT, ResultT]):
    """定义阶段计算接口，并通过浅替换生成包含新结果的数据包。"""
    name: str
    output_field: Literal["regions", "constraint", "comparison", "brow_depth", "smoothing", "eye_height"]

    def run(self, data: PipelineData, config: ConfigT) -> PipelineData:
        """调用阶段计算后统一写回结果字段，保留其他上游字段的引用。"""
        # 浅替换只创建新容器；需要修改数组时由计算实现显式持有副本。
        return replace(data, **{self.output_field: self.compute(data, config)})

    @abstractmethod
    def compute(self, data: PipelineData, config: ConfigT) -> ResultT:
        """约定子类将上游数据视为只读，并独立持有需要修改的计算结果。"""
        raise NotImplementedError


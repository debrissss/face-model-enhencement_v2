"""定义可视化接口，并按阶段名称显式分派到具体渲染器。"""
from collections.abc import Mapping
from typing import Protocol

from ..models import PipelineData


class Visualizer(Protocol):
    """约定渲染器统一接收流水线数据包。"""
    def render(self, data: PipelineData) -> None:
        """声明具体渲染器必须实现的渲染操作。"""
        ...


class VisualizerRegistry:
    """保存阶段名称与渲染器的映射，作为流水线的统一可视化入口。"""
    def __init__(self, renderers: Mapping[str, Visualizer]):
        """持有独立的映射容器，供后续按阶段查找渲染器。"""
        self._renderers = dict(renderers)

    def render(self, stage_name: str, data: PipelineData) -> None:
        """查找当前阶段对应的渲染器并调用其渲染方法。"""
        self._renderers[stage_name].render(data)


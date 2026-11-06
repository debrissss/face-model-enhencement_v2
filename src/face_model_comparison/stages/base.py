"""对比 Stage 只在内存数据包间交接。"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from face_model_enhancement.models import PipelineData

ConfigT = TypeVar("ConfigT")


class BaseStage(ABC, Generic[ConfigT]):
    name: str

    @abstractmethod
    def run(self, data: PipelineData, config: ConfigT) -> PipelineData:
        raise NotImplementedError

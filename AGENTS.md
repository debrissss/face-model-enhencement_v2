# Python 3D Processing Pipeline 工程开发规范

## 一、行为约束

### 1. 总体行为约束与原则

- 项目中的长度单位为 mm。
- 禁止防御性编程；仅针对正常执行流程中已经出现、且并非代码 bug 导致的问题做必要防御。
- 按照能够完成功能的最小实现开发，避免冗余代码和不必要的抽象；有适用的现成库时，优先复用其实现。
- 使用 `uv` 管理环境，环境保存在当前项目目录的 `environment/` 下，并在 `.gitignore` 中忽略该目录。
- 未经用户明确许可，不得编辑 `.gitignore`。

### 2. 项目结构约束

#### 2.1 目录结构

推荐项目结构如下：
```text
project_name/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/                          # 全局与 Stage 配置
│   ├── pipeline.yaml
│   └── stages/
│       └── <stage_name>.yaml
├── data/
│   └── input/
├── artifacts/                        # 运行产物
│   └── <run_id>/                     # 单次 Run 的产物目录
│       ├── config_snapshot/          # 本次 Run 的配置快照
│       ├── run_log                   # 本次 Run 的运行日志
│       └── <stage_name>/             # 该 Stage 的运行产物
│           ├── example.json
│           ├── example.npz
│           └── example.png
├── src/
│   └── pipeline_name/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py                 # 配置模型、加载、校验
│       ├── models.py                 # PipelineData 与领域数据模型
│       ├── pipeline.py               # Stage 编排、生命周期与状态机
│       ├── stages/                   # Stage 实现
│       │   ├── __init__.py
│       │   ├── base.py               # BaseStage 抽象基类
│       │   └── example.py            # Stage 实现，继承 bass.py
│       ├── io/                       # Serializer 与统一读写接口（按数据类型组织）
│       │   ├── __init__.py
│       │   └── example.py
│       ├── visualization/            # Visualizer 实现
│       │   ├── __init__.py
│       │   └── example.py
│       └── utils/                    # 通用工具
│           ├── __init__.py
│           └── logging.py            # 日志配置与工具
└── scripts/                          # 脚本
```

### 3. PipelineData 与数据约束（对应 `models.py`）

- Stage 之间流转的数据必须全部打包进 `PipelineData`，不得绕过 `PipelineData` 直接传递。

#### 示例
```python
# models.py
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class PipelineData:
    """Stage 之间流转的数据包；字段按具体任务扩展。"""
    source: Any = None
    target: Any = None
    result: Any = None
```

### 4. Stage 约束（对应 `stages/`）

- Stage 拆分依据是**独立计算责任与可复用性**，不得为每个输出/文件强制拆成一个新 Stage。
- 阶段内的算法步骤应分别封装为独立函数，通过参数接收所需输入，通过返回值传递结果。
- `run()` 本身不实现任何算法，只负责取得输入、按顺序调用步骤函数，最后将本阶段结果提交到新的 `PipelineData`。

#### 示例

**示例仅用于展示结构与职责划分。Agent 应参考其组织形式，根据实际任务设计步骤、函数、配置和结果字段，不得直接照搬示例的业务内容，也不得为了套用示例而增加无关步骤。**

```python
# base.py
from abc import ABC, abstractmethod
from typing import Any

from pipeline3d.models import PipelineData

class BaseStage(ABC):
    """Stage 基类：由具体 Stage 实现流程串联与结果提交。"""

    name: str

    @abstractmethod
    def run(self, data: PipelineData, config: Any) -> PipelineData:
        """串联独立步骤，并返回包含本阶段结果的新数据包。"""
        raise NotImplementedError
```

```python
# example.py
from dataclasses import dataclass, replace

from pipeline3d.models import PipelineData
from pipeline3d.stages.base import BaseStage

Point = tuple[float, float, float]


@dataclass(frozen=True)
class ExampleConfig:
    min_z: float  # mm
    offset_z: float  # mm


def filter_points(points: list[Point], min_z: float) -> list[Point]:
    """筛选满足高度条件的点，返回新列表。"""
    return [point for point in points if point[2] >= min_z]


def translate_points(points: list[Point], offset_z: float) -> list[Point]:
    """沿 Z 轴平移，返回新列表。"""
    return [(x, y, z + offset_z) for x, y, z in points]


class ExampleStage(BaseStage):
    name = "example_stage"

    def run(self, data: PipelineData, config: ExampleConfig) -> PipelineData:
        points = data.source
        filtered = filter_points(points, config.min_z)
        result = translate_points(filtered, config.offset_z)
        return replace(data, result=result)
```

说明：

- `run()` 是具体 Stage 必须实现的方法；只串联流程与最终提交结果，不包含算法实现。
- 步骤函数只负责各自的计算，不向 `PipelineData` 提交结果；阶段内的中间结果通过 `run()` 的局部变量传递，Stage 之间仍只通过 `PipelineData` 交接。
- `run()` 与步骤函数应把输入数据及其字段视为只读，不要原地修改上游数据；需要原地优化时，在步骤函数内显式 `copy()` 并说明所有权，不要无条件深拷贝点云。
- `replace()` 是浅拷贝：新容器与上游仍共享可变字段；是否复制由各 Stage 按数据大小与所有权策略决定。
- 示例假定 `PipelineData.source` 与 `PipelineData.result` 为点列表；实际项目应按任务定义具体字段类型，并把基类中的 `Any` 替换为具体类型或 `Protocol`/泛型，避免接口失去类型约束。

### 5. Pipeline 约束（对应 `pipeline.py`）


- 每次执行是一次 Run，必须具有唯一 ID 和配置快照；运行名称由 Pipeline 根据运行时间、输入文件名等自动组装，并处理重名以保证唯一性。
- Stage 之间只通过内存中的 `PipelineData` 传递数据，不得以文件作为常规交接。

#### 示例

```python
# pipeline.py
import logging
from typing import Any, Iterable

from pipeline3d.models import PipelineData


class Pipeline:
    """按顺序执行 Stage；Stage 之间只通过内存中的 PipelineData 交接。"""

    def __init__(
        self,
        stages: Iterable[tuple[Any, Any]],
        store: Any,
        visualizer: Any,  # 必须是按 stage.name 分发的 VisualizerRegistry
        visualize: Iterable[str] = (),
    ) -> None:
        self.stages = list(stages)
        self.store = store
        self.visualizer = visualizer
        self.visualize = set(visualize)

    def run(self, data: PipelineData) -> PipelineData:
        for stage, config in self.stages:
            data = stage.run(data, config)

            # 正常路径到此为止：data 仍在内存中，直接交给下一个 Stage。
            if stage.name not in self.visualize:
                continue

            # 只有命中 visualize 列表的 Stage 才落盘；失败只记录，不影响计算结果。
            try:
                self.visualize_stage(stage, data)
            except Exception:
                logging.exception("visualization failed: %s", stage.name)

        return data

    def visualize_stage(self, stage: Any, data: PipelineData) -> None:
        saved = self.store.save(data, stage.name)
        loaded = self.store.load(saved)
        # visualizer 必须是按 stage.name 分发的 VisualizerRegistry。
        self.visualizer.render(stage.name, loaded)
```

说明：

- `stages` 为 `(stage, config)` 对，由配置层装配；示例省略了配置加载细节。
- `visualize` 是从 `pipeline.yaml` 的 `pipeline.visualize` 读取的 Stage 名集合；为空表示不做可视化。

### 6. IO 与可视化约束（对应 `io/`、`visualization/`）

#### 6.1 共同要求

- 职责边界：具体实现只承担各自的序列化或渲染职责，不承担 Stage 编排或核心计算；处理器选择由外层统一入口负责，具体实现不得在内部再次分发。
- 复用与扩展：优先复用现有实现；只有出现新的序列化或可视化语义时才新增或扩展实现，不因新增一个 Stage 就自动新增实现。
- 数据只读：必须将输入数据及其字段视为只读；需要修改时复制相关数据，不得原地修改上游数据。

#### 6.2 IO 专属要求

- Serializer 负责序列化与反序列化，提供 `save(data, path)`、`load(path)`。
- IO 按数据类型组织；使用静态映射 `SERIALIZERS` 显式注册 `数据类型 → Serializer`，不按 Stage 名分叉或为单个 Stage 新建 IO 模块。
- `ArtifactStore` 负责按数据类型查表并调用 Serializer、组织保存路径，以及提供统一的保存和加载入口，如 `save(data, stage_dir)`、`load(stage_dir)`。

#### 6.3 Serializer 示例

```python
# io/example.py
import json
from pathlib import Path
from typing import Any, Protocol


class Serializer(Protocol):
    def save(self, data: Any, path: Path) -> None: ...
    def load(self, path: Path) -> Any: ...


class JsonSerializer:
    """结构化数据的简单 Serializer；实际按 Artifact 类型扩展。"""

    def save(self, data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))


# 按数据类型显式注册；不要按 Stage 名称分叉。
SERIALIZERS = {
    dict: JsonSerializer(),
}
```

示例省略了 `ArtifactStore` 的路径与 manifest 管理。

#### 6.4 可视化专属要求

- Visualizer 负责渲染，提供 `render(data)`，输入为 `PipelineData`。
- 使用 `VisualizerRegistry` 显式注册 `stage.name → Visualizer`；多个 Stage 共享同一可视化语义时，可以映射到同一个 Visualizer 实例。
- `VisualizerRegistry` 负责按 `stage.name` 查表并调用 Visualizer；`Pipeline` 只通过其 `render(stage_name, data)` 入口分发。

#### 6.5 Visualizer 示例

```python
# visualization/registry.py
from typing import Mapping, Protocol

from pipeline3d.models import PipelineData


class Visualizer(Protocol):
    """具体可视化实现。"""
    def render(self, data: PipelineData) -> None: ...


class VisualizerRegistry:
    """按 Stage 名分发到具体 Visualizer；Pipeline 的唯一可视化入口。"""

    def __init__(self, renderers: Mapping[str, Visualizer]) -> None:
        self._renderers = dict(renderers)

    def render(self, stage_name: str, data: PipelineData) -> None:
        renderer = self._renderers.get(stage_name)
        if renderer is None:
            raise ValueError(f"no visualizer for stage: {stage_name!r}")
        renderer.render(data)
```

装配：

```python
visualizer = VisualizerRegistry({
    "keypoints_source": KeypointsVisualizer(),
    "point_matching": MatchesVisualizer(),
})
```

### 7. 配置与 CLI 约束（对应 `config.py`、`configs/`、`cli.py`）

#### 7.1 配置文件约束

- 配置文件使用 YAML；Pipeline 配置与各 Stage 配置分文件组织。
- Pipeline 配置声明 Stage 列表、执行顺序及对应配置文件路径；列表顺序即执行顺序。
- 配置中引用的相对路径以引用它的配置文件所在目录为基准解析。
- 需要可视化的 Stage 在 `pipeline.visualize` 中声明。
- 必须将 pipeline 配置文件与 Stage 配置文件解析、校验后，转换成明确的 Stage 配置对象。

以下仅示意配置文件的组织格式；Stage 名和参数名均为占位示例，不代表具体任务或实际参数。

```yaml
# configs/pipeline.yaml
schema_version: 1
pipeline:
  stages:
    - id: example_stage
      config: stages/example_stage.yaml
  visualize:
    - example_stage
```

```yaml
# configs/stages/example_stage.yaml
schema_version: 1
parameters:
  example_option: example_value
```

#### 7.2 CLI 约束

- 输入路径至少一个，参数名采用 `--input_<name>` 形式，`<name>` 表示输入名称；每个参数对应一个输入路径。
- 有几个输入路径，就分别调用几次 IO 读取，并按输入名称将读取结果组装进 `PipelineData`。
- 输出路径只使用一个 `--output` 参数：正式运行输出到 `artifacts`，开发测试输出到 `artifacts_test`。
- `--config` 指定 Pipeline 配置文件路径。
- CLI 不覆盖 `pipeline.visualize`。
- 运行名称由 Pipeline 自动组装，不提供 `--run-name` 参数。

```bash
# 正式运行：示例包含两个输入路径，分别调用一次 IO 读取
pipeline3d run --input_example_a data/input/example_a.dat --input_example_b data/input/example_b.dat --output artifacts --config configs/pipeline.yaml

# 开发测试：示例包含一个输入路径，调用一次 IO 读取
pipeline3d run --input_example_a data/input/example_a.dat --output artifacts_test --config configs/pipeline.yaml
```

### 8. 日志约束（对应 `utils/logging.py`）

#### 8.1 日志记录

- 必须使用 `logging` 或项目统一日志系统；库代码禁止散落 `print()` 作为正式日志。
- 必须记录：`run_id`、`stage_id`、开始/结束状态、算法/序列化/可视化耗时、异常上下文；三类耗时必须分开，避免可视化开销污染算法性能基准。


### 9.方法名约束

* **行为语义优先**：根据方法对外提供的实际行为和职责命名，而非机械描述内部实现步骤。名称应让调用者能够理解方法做什么。

* **调用上下文清晰**：方法名应在其预期调用位置具有明确含义。利用类、模块等已有上下文避免重复，但不能依赖调用者无法直观获知的隐含信息。

* **使用准确的动词**：优先使用能够明确表达操作语义的动词，如 `get`、`find`、`extract`、`compute`、`convert`、`remap`、`update` 等。避免使用 `do`、`process`、`handle` 等含义宽泛的动词，除非它们准确描述该方法的职责。

* **区分相近操作**：根据实际行为区分获取、查找、提取、计算、转换、修改等操作，不要仅因命名习惯而混用具有不同语义的动词。

* **体现关键结果**：方法名应明确表达操作对象及必要的结果语义，尤其要区分对象、索引、标识符、掩码等不同的数据表示，但不必重复类型签名已充分表达的信息。

* **体现可观察的副作用**：名称应符合调用者对方法行为的预期，尤其应避免将修改状态、执行 I/O 或产生其他重要副作用的方法伪装成单纯的查询或计算操作。

* **职责单一且名称聚焦**：方法名应围绕其主要职责命名，不要堆砌内部处理步骤。若无法用简洁、准确的名称描述方法职责，应考虑是否需要拆分方法。

* **避免冗余上下文**：不要重复已由类、模块或方法所属抽象明确提供的业务信息，但应保留在调用位置区分不同操作或对象所必需的限定词。

* **遵循现有约定**：优先沿用项目中已有的术语、动词语义及命名风格，保持相似操作的命名一致性。

* **避免无意义的重命名**：除非现有名称存在歧义、语义不准确或明显违反项目约定，否则不要仅为统一风格而重命名方法。


### 10.变量名约束

- 把变量命名分成两个维度：业务身份、算法角色，在具体的业务入口中，业务身份很重要；进入局部算法后，算法角色通常更重要。

* **局部语义优先**：明确变量的作用域，根据变量在当前作用域中的实际含义和算法角色命名，而非机械继承外层业务流程或调用方的上下文。

* **避免冗余上下文**：当业务、函数、类或模块上下文在当前作用域中已足够明确时，不要在变量名中重复。

* **保留必要区分**：当变量在用途、数据表示、坐标系、索引空间、单位或状态等方面存在歧义时，应使用适当的限定词加以区分。

* **确保语义精确**：名称应准确反映变量所表示的数据及其含义，而不只是描述数据的来源或用途。

* **避免无意义的重命名**：除非现有名称存在歧义或语义不准确，否则不要仅为统一风格而重命名变量。

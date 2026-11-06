"""创建独立的单次运行日志，同时避免改动应用的根日志配置。"""
from contextvars import ContextVar
import logging
from pathlib import Path


_active_run: ContextVar[logging.LoggerAdapter | None] = ContextVar("active_pipeline_run", default=None)


def get_logger() -> logging.Logger | logging.LoggerAdapter:
    """计算步骤沿用当前 Run 日志，直接调用步骤时使用模块日志。"""
    return _active_run.get() or logging.getLogger("face_model_enhancement")


def run_logger(run_id: str, directory: Path) -> logging.LoggerAdapter:
    """创建同时写入文件和终端的日志器，并为每条记录关联运行标识。"""
    logger = logging.getLogger(f"face_model_enhancement.run.{run_id}")
    logger.setLevel(logging.INFO)
    # 禁止向根日志器重复传播，避免一条运行记录被多次输出。
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s run_id=%(run_id)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(directory / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # 通过适配器自动补充运行标识，阶段只需记录各自的状态和耗时。
    adapter = logging.LoggerAdapter(logger, {"run_id": run_id})
    _active_run.set(adapter)
    return adapter


def close_logger(logger: logging.LoggerAdapter) -> None:
    """关闭并移除本次运行的日志处理器，释放文件资源。"""
    _active_run.set(None)
    # 遍历处理器列表副本，使移除操作不影响本次遍历。
    for handler in logger.logger.handlers[:]:
        handler.close()
        logger.logger.removeHandler(handler)


"""
Booster T1 统一日志模块

所有模块通过 setup_logger() 获取统一格式的 Logger 实例。
"""

from __future__ import annotations

import logging
import sys

# 全局格式 — 各模块可覆写
_LOG_FORMAT = "[%(asctime)s][%(levelname)-5s][%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    *,
    stream=None,
) -> logging.Logger:
    """创建统一格式的 Logger 实例。

    Parameters
    ----------
    name : str
        Logger 名称（建议使用 ``__name__``）。
    level : int
        日志等级，默认 ``logging.INFO``。
    stream
        输出流，默认 ``sys.stderr``。

    Returns
    -------
    logging.Logger
        已配置 handler + formatter 的 Logger，控制台输出。
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（多次调用 setup_logger 不会叠加）
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # 阻止日志向 root logger 传播（避免重复输出）
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已有 Logger 的快捷函数（不会重新配置 handler）。

    Parameters
    ----------
    name : str
        Logger 名称。

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)

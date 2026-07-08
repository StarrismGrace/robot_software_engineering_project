"""
Booster T1 配置加载模块

从 YAML 文件加载配置，缺失时回退到默认值。
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from common.logger import setup_logger

_logger = setup_logger(__name__)

# ============================================================
# 默认配置
# ============================================================
_DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "input_path": "data/input_video.mp4",
        "fps": 30,
        "max_frames": 300,
    },
    "pose": {
        "model_name": "rtmpose-m",
        "device": "cpu",
        "confidence_threshold": 0.5,
    },
    "cleaner": {
        "interpolation_method": "linear",
        "smooth_window": 5,
        "outlier_threshold": 3.0,
    },
    "retargeting": {
        "method": "ik",           # ik / analytic / learned
        "scale_factor": 1.0,
    },
    "simulation": {
        "render": True,
        "output_video": "outputs/result.mp4",
        "duration": 10.0,
    },
    "logging": {
        "level": "INFO",
        "save_logs": True,
    },
}


# ============================================================
# get_default_config
# ============================================================
def get_default_config() -> Dict[str, Any]:
    """返回默认配置的深拷贝，调用方可安全修改。

    Returns
    -------
    dict
        默认配置字典。
    """
    return deepcopy(_DEFAULT_CONFIG)


# ============================================================
# load_config
# ============================================================
def load_config(
    config_path: str | Path = "config.yaml",
    *,
    use_defaults_on_missing: bool = True,
) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    配置文件不存在时：
      - ``use_defaults_on_missing=True``：返回默认配置。
      - ``use_defaults_on_missing=False``：抛出 FileNotFoundError。

    YAML 中存在的键会**递归合并**到默认配置上，
    未在 YAML 中指定的键保留默认值。

    Parameters
    ----------
    config_path : str | Path
        YAML 配置文件路径，默认 ``config.yaml``。
    use_defaults_on_missing : bool
        配置文件缺失时是否回退到默认值。

    Returns
    -------
    dict
        合并后的配置字典。

    Raises
    ------
    FileNotFoundError
        当 ``use_defaults_on_missing=False`` 且配置文件不存在时。
    ImportError
        当 PyYAML 未安装时。
    """
    config_path = Path(config_path)
    config = get_default_config()

    if not config_path.exists():
        if use_defaults_on_missing:
            _logger.warning(
                "配置文件 %s 不存在，使用默认配置", str(config_path.resolve())
            )
            return config
        raise FileNotFoundError(
            f"配置文件不存在: {config_path.resolve()}"
        )

    # ---- 读取 YAML ----
    yaml_content = _read_yaml(config_path)
    if yaml_content is None:
        _logger.warning("配置文件 %s 为空，使用默认配置", str(config_path.resolve()))
        return config

    # ---- 递归合并 ----
    _deep_merge(config, yaml_content)
    _logger.info("已加载配置: %s", str(config_path.resolve()))
    return config


# ============================================================
# 内部辅助
# ============================================================
def _read_yaml(path: Path) -> Dict[str, Any] | None:
    """读取并解析 YAML 文件。"""
    try:
        import yaml
    except ImportError:
        # 回退：尝试用标准库 json 读取（支持注释的 YAML 会失败，提示安装 PyYAML）
        raise ImportError(
            "读取 YAML 需要 PyYAML 库，请执行: pip install pyyaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return None

    if not isinstance(data, dict):
        _logger.warning("配置文件顶层应为 dict，实际: %s", type(data).__name__)
        return None

    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """递归将 override 合并到 base 中（原地修改）。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

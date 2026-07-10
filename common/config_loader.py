"""
配置文件加载器

支持 YAML 格式的配置文件加载，提供点号分隔的嵌套键访问。
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件的便捷函数
    
    Parameters
    ----------
    config_path : str
        配置文件路径

    Returns
    -------
    Dict[str, Any]
        配置字典
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns
    -------
    Dict[str, Any]
        默认配置字典
    """
    return {
        "video": {
            "fps": 30,
            "max_duration": 60,
        },
        "pose_extractor": {
            "model": "prompthmr",
            "confidence_threshold": 0.5,
        },
        "motion_cleaner": {
            "smooth_window": 5,
            "outlier_threshold": 3.0,
        },
        "retargeting": {
            "scale_factor": 1.0,
        },
        "mujoco": {
            "model_path": "scene.xml",
            "render_resolution": [640, 480],
        },
    }


class ConfigLoader:
    """YAML 配置文件加载器"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的嵌套键
        
        示例:
            config.get("video.fps")  # 获取 video 下的 fps 值
            config.get("model.path", "default.xml")  # 带默认值
        """
        config = self.load()
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self.load()

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path={self.config_path}, loaded={self._config is not None})"
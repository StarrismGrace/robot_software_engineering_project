#!/usr/bin/env python3
"""
Booster T1 舞蹈动作重定向系统 — 主入口

流水线：
  视频处理 → 姿态提取 → 数据清洗 → 动作重定向 → MuJoCo 仿真播放

用法:
  python main.py --video data/dance.mp4
  python main.py --mock                          # Mock 数据走全流程
  python main.py --video input.mp4 --skip_clean   # 跳过数据清洗
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from common.config_loader import get_default_config, load_config
from common.logger import setup_logger

# 模块接口（真实实现由各成员后续接入）
from common.interfaces import (
    MotionCleaner,
    MuJoCoPlayer,
    PoseExtractor,
    Retargeting,
    VideoProcessor,
)
from common.mock_factory import (
    create_mock_frames,
    create_mock_pose_data,
    create_mock_clean_data,
    create_mock_robot_motion,
)
from common.motion_data import MotionData

# ============================================================
# Logger
# ============================================================
_logger = setup_logger("BoosterT1")


# ============================================================
# 命令行参数
# ============================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Booster T1 — 舞蹈动作重定向系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --video data/ballet.mp4 --output outputs/result.mp4
  python main.py --mock
  python main.py --config my_config.yaml --skip_clean
        """,
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="输入视频路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 Mock 数据运行全流程（不依赖真实模块和视频文件）",
    )
    parser.add_argument(
        "--skip_pose",
        action="store_true",
        help="跳过姿态提取步骤",
    )
    parser.add_argument(
        "--skip_clean",
        action="store_true",
        help="跳过数据清洗步骤",
    )
    parser.add_argument(
        "--skip_retarget",
        action="store_true",
        help="跳过动作重定向步骤",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出视频路径 (覆盖配置文件中的值)",
    )
    return parser


# ============================================================
# 步骤占位（成员 H 后续替换为真实模块实例化逻辑）
# ============================================================
def _get_video_processor(cfg: dict) -> VideoProcessor:
    """TODO (成员H): 返回 VideoProcessor 实例 (成员E 实现)。"""
    _logger.debug("VideoProcessor 未接入（TODO）")
    # 返回一个匿名占位实现，使流程可以走通
    return _PlaceholderVideoProcessor()


def _get_pose_extractor(cfg: dict) -> PoseExtractor:
    """TODO (成员H): 返回 PoseExtractor 实例 (成员B 实现)。"""
    _logger.debug("PoseExtractor 未接入（TODO）")
    return _PlaceholderPoseExtractor()


def _get_motion_cleaner(cfg: dict) -> MotionCleaner:
    """TODO (成员H): 返回 MotionCleaner 实例 (成员I 实现)。"""
    _logger.debug("MotionCleaner 未接入（TODO）")
    return _PlaceholderMotionCleaner()


def _get_retargeting(cfg: dict) -> Retargeting:
    """TODO (成员H): 返回 Retargeting 实例 (成员D 实现)。"""
    _logger.debug("Retargeting 未接入（TODO）")
    return _PlaceholderRetargeting()


def _get_mujoco_player(cfg: dict) -> MuJoCoPlayer:
    """TODO (成员H): 返回 MuJoCoPlayer 实例 (成员F 实现)。"""
    _logger.debug("MuJoCoPlayer 未接入（TODO）")
    return _PlaceholderMuJoCoPlayer()


# ============================================================
# 占位实现（在真实模块接入前保证流程框架可运行）
# ============================================================
class _PlaceholderVideoProcessor(VideoProcessor):
    def extract_frames(self, video_path, fps=None, start_sec=0.0, end_sec=None):
        _logger.warning("VideoProcessor 占位: 返回空帧列表")
        return []

    def get_video_metadata(self, video_path):
        return {"fps": 30, "total_frames": 0, "duration_sec": 0, "width": 0, "height": 0}


class _PlaceholderPoseExtractor(PoseExtractor):
    def extract(self, frames, fps, **kwargs):
        _logger.warning("PoseExtractor 占位: 返回空 MotionData")
        raise NotImplementedError("PoseExtractor 未实现，请使用 --mock 模式")

    def get_joint_map(self):
        return {}


class _PlaceholderMotionCleaner(MotionCleaner):
    def clean(self, motion, **kwargs):
        _logger.info("MotionCleaner 占位: 透传数据")
        return motion

    def detect_outliers(self, motion, threshold=3.0):
        return np.zeros((motion.num_frames, motion.num_joints), dtype=bool)


class _PlaceholderRetargeting(Retargeting):
    def retarget(self, human_motion, **kwargs):
        _logger.warning("Retargeting 占位: 返回空 MotionData")
        raise NotImplementedError("Retargeting 未实现，请使用 --mock 模式")

    def get_joint_limits(self):
        return {}


class _PlaceholderMuJoCoPlayer(MuJoCoPlayer):
    def load_model(self, model_path):
        _logger.info("MuJoCoPlayer 占位: 跳过模型加载")

    def play(self, robot_motion, output_path=None, render=True, **kwargs):
        _logger.info("MuJoCoPlayer 占位: 跳过仿真播放")
        return []

    def get_physics_state(self):
        return {}


# ============================================================
# 流水线执行
# ============================================================
def run_pipeline(
    video_path: str | None,
    config: dict,
    *,
    mock: bool = False,
    skip_pose: bool = False,
    skip_clean: bool = False,
    skip_retarget: bool = False,
    output_path: str | None = None,
) -> int:
    """按顺序执行 Booster T1 处理流水线。

    Parameters
    ----------
    video_path : str | None
        输入视频路径。mock 模式下可为 None。
    config : dict
        项目配置字典。
    mock : bool
        True 时使用 Mock 数据走完全流程。
    skip_pose : bool
        跳过姿态提取。
    skip_clean : bool
        跳过数据清洗。
    skip_retarget : bool
        跳过动作重定向。
    output_path : str | None
        覆盖配置文件中的输出视频路径。

    Returns
    -------
    int
        0 表示成功，非 0 表示失败。
    """
    output_video = output_path or config["simulation"]["output_video"]

    _logger.info("=" * 60)
    _logger.info("Booster T1 流水线启动")
    _logger.info("  模式:      %s", "Mock 模拟" if mock else "真实运行")
    _logger.info("  输入视频:  %s", video_path or "(无)")
    _logger.info("  输出视频:  %s", output_video)
    _logger.info("  配置文件:  %s", config.get("_config_path", "默认"))
    _logger.info("=" * 60)

    # ---- 数据容器 ----
    frames: List[np.ndarray] = []
    motion: Optional[MotionData] = None
    robot_motion: Optional[MotionData] = None
    output_frames: List[np.ndarray] = []

    try:
        # ============================================================
        # Step 1: 视频处理
        # ============================================================
        _logger.info("── Step 1/5: 视频处理 ──")

        if mock:
            _logger.info("  [Mock] 生成 %d 帧模拟图像 ...", config["video"]["max_frames"])
            frames = create_mock_frames(
                num_frames=config["video"]["max_frames"],
                seed=1,
            )
        else:
            if not video_path:
                _logger.error("非 Mock 模式必须指定 --video")
                return 1
            processor = _get_video_processor(config)
            metadata = processor.get_video_metadata(video_path)
            _logger.info(
                "  视频信息: %.1f fps, %d 帧, %.1f 秒, %dx%d",
                metadata["fps"],
                metadata["total_frames"],
                metadata["duration_sec"],
                metadata["width"],
                metadata["height"],
            )
            frames = processor.extract_frames(
                video_path,
                fps=config["video"]["fps"],
                start_sec=0.0,
                end_sec=(
                    config["simulation"]["duration"]
                    if config["simulation"]["duration"] > 0
                    else None
                ),
            )

        if not frames:
            _logger.warning("  未抽取到任何帧")
        else:
            _logger.info("  ✓ 已抽取 %d 帧，shape=(%d,%d,%d)",
                         len(frames), *frames[0].shape)

        # ============================================================
        # Step 2: 姿态提取
        # ============================================================
        _logger.info("── Step 2/5: 姿态提取 ──")

        if skip_pose:
            _logger.info("  [跳过] --skip_pose")
        elif mock:
            _logger.info("  [Mock] 生成模拟人体姿态数据 ...")
            motion = create_mock_pose_data(
                num_frames=len(frames) if frames else config["video"]["max_frames"],
                fps=config["video"]["fps"],
                seed=2,
            )
        else:
            if not frames:
                _logger.error("无帧数据，无法进行姿态提取")
                return 2
            extractor = _get_pose_extractor(config)
            motion = extractor.extract(
                frames,
                fps=config["video"]["fps"],
                confidence_threshold=config["pose"]["confidence_threshold"],
            )

        if motion is not None:
            issues = motion.validate()
            if issues:
                _logger.error("姿态数据校验失败: %s", issues)
                return 2
            _logger.info("  ✓ MotionData: %d 关节, %d 帧, %.1f 秒",
                         motion.num_joints, motion.num_frames, motion.duration or 0.0)

        # ============================================================
        # Step 3: 数据清洗
        # ============================================================
        _logger.info("── Step 3/5: 数据清洗 ──")

        if skip_clean:
            _logger.info("  [跳过] --skip_clean")
        elif motion is not None:
            cleaner = _get_motion_cleaner(config)

            if mock:
                # Mock 模式下清洗器透传，但也跑一次 detect_outliers
                outliers = cleaner.detect_outliers(
                    motion,
                    threshold=config["cleaner"]["outlier_threshold"],
                )
                outlier_count = int(outliers.sum())
                _logger.info("  [Mock] 检测到 %d 个异常值 (透传数据)", outlier_count)

            motion = cleaner.clean(
                motion,
                interpolation_method=config["cleaner"]["interpolation_method"],
                smooth_window=config["cleaner"]["smooth_window"],
            )

            issues = motion.validate() if motion else []
            if issues:
                _logger.error("清洗后数据校验失败: %s", issues)
                return 3
            _logger.info("  ✓ 清洗完成: %d 关节, %d 帧", motion.num_joints, motion.num_frames)

        # ============================================================
        # Step 4: 动作重定向
        # ============================================================
        _logger.info("── Step 4/5: 动作重定向 ──")

        if skip_retarget:
            _logger.info("  [跳过] --skip_retarget")
        elif mock:
            _logger.info("  [Mock] 生成模拟机器人关节角度 ...")
            robot_motion = create_mock_robot_motion(
                num_frames=motion.num_frames if motion else config["video"]["max_frames"],
                fps=config["video"]["fps"],
                seed=3,
            )
        elif motion is not None:
            retargeter = _get_retargeting(config)
            robot_motion = retargeter.retarget(
                motion,
                method=config["retargeting"]["method"],
                scale_factor=config["retargeting"]["scale_factor"],
            )

        if robot_motion is not None:
            issues = robot_motion.validate()
            if issues:
                _logger.error("重定向数据校验失败: %s", issues)
                return 4
            _logger.info("  ✓ Robot MotionData: %d 关节, %d 帧",
                         robot_motion.num_joints, robot_motion.num_frames)

        # ============================================================
        # Step 5: MuJoCo 仿真播放
        # ============================================================
        _logger.info("── Step 5/5: MuJoCo 仿真播放 ──")

        if robot_motion is not None:
            player = _get_mujoco_player(config)

            if mock:
                _logger.info("  [Mock] 跳过模型加载与渲染，直接完成")
                output_frames = create_mock_frames(
                    num_frames=min(robot_motion.num_frames, 10),
                    seed=4,
                )
            else:
                player.load_model(config.get("simulation", {}).get("model_path", ""))
                output_frames = player.play(
                    robot_motion,
                    output_path=output_video,
                    render=config["simulation"]["render"],
                )

            _logger.info("  ✓ 仿真完成: %d 帧已渲染", len(output_frames))

        # ============================================================
        # 完成
        # ============================================================
        _logger.info("=" * 60)
        _logger.info("流水线执行成功")
        _logger.info(
            "  帧图像 %d → 姿态 %d 帧 → 清洗 %d 帧 → 机器人 %d 帧 → 渲染 %d 帧",
            len(frames),
            motion.num_frames if motion else 0,
            motion.num_frames if motion else 0,
            robot_motion.num_frames if robot_motion else 0,
            len(output_frames),
        )
        _logger.info("=" * 60)
        return 0

    except NotImplementedError as e:
        _logger.error("模块未实现: %s", e)
        _logger.error("提示: 请使用 --mock 模式进行集成测试，或等待对应成员完成模块开发")
        return 10

    except Exception as e:
        _logger.exception("流水线执行异常: %s", e)
        return 99


# ============================================================
# main 入口
# ============================================================
def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    Parameters
    ----------
    argv : list[str] | None
        命令行参数列表，为 None 时使用 sys.argv。

    Returns
    -------
    int
        退出码。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- 加载配置 ----
    config = load_config(args.config)
    config["_config_path"] = args.config

    # ---- Mock 模式下自动跳过非适用步骤 ----
    skip_pose = args.skip_pose
    skip_clean = args.skip_clean
    skip_retarget = args.skip_retarget

    if args.mock:
        if args.video:
            _logger.warning("--mock 模式下 --video 参数被忽略")
        if not args.skip_pose and not args.skip_retarget:
            pass  # mock 模式默认跑全流程

    # ---- 校验 ----
    if not args.mock and not args.video:
        _logger.error("请指定 --video <视频路径> 或使用 --mock 模式")
        parser.print_usage()
        return 1

    video_path = args.video

    return run_pipeline(
        video_path=video_path,
        config=config,
        mock=args.mock,
        skip_pose=skip_pose,
        skip_clean=skip_clean,
        skip_retarget=skip_retarget,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())

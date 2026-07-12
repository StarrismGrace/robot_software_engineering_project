from common.interfaces import PoseExtractor
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from common.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)


class PromptHMRExtractor(PoseExtractor):
    """基于PromptHMR的姿态提取实现（成员B负责模块）"""

    def __init__(self):
        self.model = None  # 后续接入真实模型时初始化
        logger.info("姿态提取器初始化完成（当前为Mock实现）")

    def extract(self, frames: list[np.ndarray],
                fps: int, **kwargs) -> MotionData:
        """
        从图像序列中提取人体姿态，输出标准化MotionData
        严格遵循公共接口：输入帧列表+帧率，输出MotionData（positions填充）
        """
        num_frames = len(frames)
        num_joints = len(BOOSTER_T1_JOINT_NAMES)
        logger.info(
            f"开始姿态提取，输入帧数: {num_frames}, 帧率: {fps}"
        )

        # ========== 核心逻辑占位（后续替换为真实PromptHMR推理） ==========
        # 真实实现时，这里会逐帧调用PromptHMR模型，输出人体3D关键点
        # 再将关键点映射到Booster T1的23个标准关节，填充positions字段

        # Mock数据：生成符合形状要求的零值位置数据，先跑通全链路格式
        mock_positions = np.zeros(
            (num_frames, num_joints, 3), dtype=np.float32)

        # 构造标准MotionData对象
        motion_data = MotionData(
            joint_names=BOOSTER_T1_JOINT_NAMES.copy(),
            fps=fps,
            num_frames=num_frames,
            positions=mock_positions,
            timestamps=np.arange(num_frames) / fps,  # 按帧率生成时间戳
        )

        logger.info(f"姿态提取完成，输出: {motion_data}")
        return motion_data

    def get_joint_map(self) -> dict:
        """返回PromptHMR关节到Booster T1标准关节的映射表"""
        # 真实实现时补充完整映射关系，这里先返回空占位
        joint_map = {
            # 示例："human_hip": "left_hip"
        }
        return joint_map

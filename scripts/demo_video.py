"""
成员F — 视频导出演示
用法: python scripts/demo_video.py
生成机器人跳舞的 MP4 视频文件
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from project1_dance.mujoco_player.player import MuJoCoPlayerImpl

# ---- 生成舞蹈动作（小幅摆动） ----
T = 300  # 10秒
J = 23
angles = np.zeros((T, J))
for t in range(T):
    for j in range(J):
        angles[t, j] = (
            0.08 * np.sin(2 * np.pi * t / 120 + j * 0.4)
            + 0.04 * np.sin(2 * np.pi * t / 60 + j * 0.7)
            + 0.02 * np.cos(2 * np.pi * t / 200 + j * 0.2)
        )

motion = MotionData(
    joint_names=list(BOOSTER_T1_JOINT_NAMES),
    fps=30,
    num_frames=T,
    angles=angles,
    timestamps=np.arange(T) / 30.0,
)

# ---- 播放并导出视频 ----
player = MuJoCoPlayerImpl(width=640, height=480)
player.load_model("scene.xml")

output = "outputs/dance_demo.mp4"
player.play(motion, output_path=output, render=True)
print(f"视频已导出: {output}")

"""
成员F — MuJoCo 实时播放演示
用法: python scripts/demo_play.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# 重置到 home 关键帧
mujoco.mj_resetDataKeyframe(model, data, 0)
home_qpos = data.qpos.copy()

# 自由关节前7个值不动，后面23个是你要控制的关节
FREE_DOF = 7

# 舞蹈动作（关节角度偏移量）
T = 600
J = model.nu
offsets = np.zeros((T, J))
for t in range(T):
    for j in range(J):
        offsets[t, j] = (
            0.08 * np.sin(2 * np.pi * t / 120 + j * 0.4)
            + 0.04 * np.sin(2 * np.pi * t / 60 + j * 0.7)
        )

print(f"模型已加载: {model.nu} 关节, 按 ESC 退出")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = -160
    viewer.cam.elevation = -20
    viewer.cam.distance = 2.0

    frame = 0
    while viewer.is_running():
        t = frame % T

        # 直接设置关节角度（自由关节不动，机器人不会倒）
        data.qpos[FREE_DOF:] = home_qpos[FREE_DOF:] + offsets[t, :]
        data.qvel[:] = 0  # 冻结速度，避免惯性导致摔倒

        # 更新物理状态
        mujoco.mj_forward(model, data)

        viewer.sync()
        frame += 1
        if frame % 30 == 0:
            print(f"  播放中... {frame // 30} 秒")

print(f"播放结束")

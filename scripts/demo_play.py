"""
成员F — MuJoCo 实时播放演示
用法: python scripts/demo_play.py
打开 MuJoCo 窗口，实时显示机器人跳舞动画
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

# ---- 加载模型 ----
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# 重置到 home 关键帧（机器人站立姿态）
mujoco.mj_resetDataKeyframe(model, data, 0)

# 记录站立姿态的 ctrl 作为基准
home_ctrl = data.ctrl.copy()

# ---- 生成舞蹈动作（小幅摆动，叠加在站立姿态上） ----
T = 600  # 20秒
J = model.nu
angles = np.zeros((T, J))
for t in range(T):
    for j in range(J):
        angles[t, j] = (
            0.08 * np.sin(2 * np.pi * t / 120 + j * 0.4)      # 慢速主周期
            + 0.04 * np.sin(2 * np.pi * t / 60 + j * 0.7)     # 中速叠加
            + 0.02 * np.cos(2 * np.pi * t / 200 + j * 0.2)    # 缓慢偏移
        )

print(f"模型已加载: {model.nu} 个执行器, {model.nq} 个自由度")
print("按 ESC 或关闭窗口退出")

# ---- 打开窗口播放 ----
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = -160
    viewer.cam.elevation = -20
    viewer.cam.distance = 2.0

    frame = 0
    while viewer.is_running():
        t = frame % T

        # 在站立姿态基础上叠加舞蹈动作
        data.ctrl[:] = home_ctrl + angles[t, :]

        # 物理步进
        mujoco.mj_step(model, data)

        # 同步渲染
        viewer.sync()

        frame += 1
        if frame % 30 == 0:
            print(f"  播放中... {frame // 30} 秒")

print(f"播放结束，共 {frame} 帧")

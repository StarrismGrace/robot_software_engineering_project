"""
从舞蹈视频提取 Booster T1 关节轨迹 (ONNX 姿态估计 v2)
用法: python scripts/extract_trajectory.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import onnxruntime as ort

import mujoco
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES

VIDEO_PATH = "inputs/dance.mp4"
MODEL_PATH = "models/yolov8n-pose.onnx"
OUTPUT_NPY = "outputs/trajectory.npy"
FPS_TARGET = 30
IMG_SIZE = 640

# 关键点索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def extract_trajectory(video_path):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / FPS_TARGET))
    print(f"视频: {fps:.0f}fps, {total}帧, 步长={step}")

    # home 姿态 (站立基准)
    mj_model = mujoco.MjModel.from_xml_path("scene.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    home_ctrl = mj_data.ctrl.copy()

    all_angles = []
    frame_idx = 0
    saved = 0
    prev_kpts = None

    # 累积统计用于归一化
    motion_history = {k: [] for k in [
        "lh_pitch", "rh_pitch", "lk_angle", "rk_angle",
        "lw_y", "rw_y", "le_angle", "re_angle",
        "torso_sway", "head_y"
    ]}

    # ---- 第一遍：收集所有关键点和运动数据 ----
    print("第一遍: 检测姿态...")
    raw_motions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

        preds = session.run(None, {"images": img})[0][0]
        best = int(np.argmax(preds[4]))
        if preds[4, best] < 0.3:
            frame_idx += 1
            continue

        kpts = np.zeros((17, 2))
        for k in range(17):
            kpts[k, 0] = (preds[5 + k * 3, best] / IMG_SIZE) * w
            kpts[k, 1] = (preds[5 + k * 3 + 1, best] / IMG_SIZE) * h
        prev_kpts = kpts

        # 核心测量
        m_shoulder = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2
        m_hip = (kpts[L_HIP] + kpts[R_HIP]) / 2
        body_h = np.linalg.norm(m_shoulder - m_hip) + 1e-8

        # 左腿: 髋-膝-踝角度
        lh = kpts[L_HIP]; lk = kpts[L_KNEE]; la = kpts[L_ANKLE]
        l_thigh = lk - lh; l_shin = la - lk
        l_hip_angle = np.arctan2(l_thigh[0], -l_thigh[1])  # 髋关节摆动
        l_knee_angle = np.arccos(np.clip(
            np.dot(-l_thigh, l_shin) / (np.linalg.norm(l_thigh) * np.linalg.norm(l_shin) + 1e-8), -1, 1))

        # 右腿
        rh = kpts[R_HIP]; rk = kpts[R_KNEE]; ra = kpts[R_ANKLE]
        r_thigh = rk - rh; r_shin = ra - rk
        r_hip_angle = np.arctan2(r_thigh[0], -r_thigh[1])
        r_knee_angle = np.arccos(np.clip(
            np.dot(-r_thigh, r_shin) / (np.linalg.norm(r_thigh) * np.linalg.norm(r_shin) + 1e-8), -1, 1))

        # 手臂相对于躯干的位置
        lw_rel_y = (kpts[L_WRIST, 1] - m_shoulder[1]) / body_h
        rw_rel_y = (kpts[R_WRIST, 1] - m_shoulder[1]) / body_h

        # 肘部角度
        ls = kpts[L_SHOULDER]; le = kpts[L_ELBOW]; lw = kpts[L_WRIST]
        l_upper = le - ls; l_forearm = lw - le
        l_elbow = np.arccos(np.clip(
            np.dot(-l_upper, l_forearm) / (np.linalg.norm(l_upper) * np.linalg.norm(l_forearm) + 1e-8), -1, 1))

        rs = kpts[R_SHOULDER]; re = kpts[R_ELBOW]; rw = kpts[R_WRIST]
        r_upper = re - rs; r_forearm = rw - re
        r_elbow = np.arccos(np.clip(
            np.dot(-r_upper, r_forearm) / (np.linalg.norm(r_upper) * np.linalg.norm(r_forearm) + 1e-8), -1, 1))

        # 躯干摇摆
        torso_sway = (m_shoulder[0] - m_hip[0]) / body_h

        # 头部位置
        head_y = (kpts[NOSE, 1] - m_shoulder[1]) / body_h

        raw_motions.append({
            "lh_pitch": l_hip_angle, "rh_pitch": r_hip_angle,
            "lk_angle": l_knee_angle, "rk_angle": r_knee_angle,
            "lw_y": lw_rel_y, "rw_y": rw_rel_y,
            "le_angle": l_elbow, "re_angle": r_elbow,
            "torso_sway": torso_sway, "head_y": head_y,
        })

        for k, v in raw_motions[-1].items():
            motion_history[k].append(v)

        saved += 1
        frame_idx += 1
        if saved % 50 == 0:
            print(f"  已检测 {saved} 帧...")

    cap.release()

    if saved < 10:
        print("姿态太少!")
        return None

    # ---- 第二遍: 归一化并映射到机器人关节 ----
    print(f"\n第二遍: 映射 {saved} 帧到机器人关节...")

    # 计算各通道的均值和标准差用于归一化
    stats = {}
    for k, vals in motion_history.items():
        v = np.array(vals)
        stats[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    all_offsets = []
    for i, rm in enumerate(raw_motions):
        # 归一化到 [-1, 1]
        def norm(key, default=0.0):
            s = stats[key]
            if s["std"] < 0.01:
                return default
            return float(np.clip((rm[key] - s["mean"]) / (s["std"] * 2), -1.0, 1.0))

        lhp = norm("lh_pitch")
        rhp = norm("rh_pitch")
        lka = norm("lk_angle")
        rka = norm("rk_angle")
        lwy = norm("lw_y")
        rwy = norm("rw_y")
        lea = norm("le_angle")
        rea = norm("re_angle")
        ts = norm("torso_sway")
        hy = norm("head_y")

        # 映射到关节角度偏移（幅度控制在安全范围内）
        angles = home_ctrl.copy()

        # 头: yaw(0), pitch(1)
        angles[0] += hy * 0.3
        angles[1] += hy * 0.15

        # 左臂: Shoulder_Pitch(2), Shoulder_Roll(3), Elbow_Pitch(4), Elbow_Yaw(5)
        angles[2] += lwy * 0.8       # 手臂上下
        angles[3] += lwy * 0.3       # 手臂旋转
        angles[4] += lea * 0.6       # 肘弯曲
        angles[5] = home_ctrl[5]     # 保持 home 值

        # 右臂: Shoulder_Pitch(6), Shoulder_Roll(7), Elbow_Pitch(8), Elbow_Yaw(9)
        angles[6] += rwy * 0.8
        angles[7] += rwy * 0.3
        angles[8] += rea * 0.6
        angles[9] = home_ctrl[9]

        # 躯干: Waist(10)
        angles[10] += ts * 0.5

        # 左腿: Hip_Pitch(11), Hip_Roll(12), Hip_Yaw(13), Knee_Pitch(14), Ankle_Pitch(15), Ankle_Roll(16)
        angles[11] += lhp * 0.5
        angles[12] = home_ctrl[12]
        angles[13] = home_ctrl[13]
        angles[14] += lka * 0.5
        angles[15] = home_ctrl[15]
        angles[16] = home_ctrl[16]

        # 右腿: Hip_Pitch(17), Hip_Roll(18), Hip_Yaw(19), Knee_Pitch(20), Ankle_Pitch(21), Ankle_Roll(22)
        angles[17] += rhp * 0.5
        angles[18] = home_ctrl[18]
        angles[19] = home_ctrl[19]
        angles[20] += rka * 0.5
        angles[21] = home_ctrl[21]
        angles[22] = home_ctrl[22]

        all_offsets.append(angles - home_ctrl)

    # 平滑
    angles_array = np.array(all_offsets)
    from scipy.signal import savgol_filter
    for j in range(23):
        try:
            w = min(11, max(3, len(all_offsets) // 3))
            if w % 2 == 0:
                w -= 1
            if w >= 3:
                angles_array[:, j] = savgol_filter(angles_array[:, j], w, 2)
        except Exception:
            pass

    print(f"完成: {saved} 帧, shape={angles_array.shape}")
    print(f"关节运动范围:")
    for j in range(23):
        rng = angles_array[:, j].max() - angles_array[:, j].min()
        if rng > 0.05:
            print(f"  [{j}] {BOOSTER_T1_JOINT_NAMES[j]}: {angles_array[:, j].min():.2f} ~ {angles_array[:, j].max():.2f}")

    return angles_array, saved


def save_trajectory(angles, num_frames):
    import os
    os.makedirs("outputs", exist_ok=True)
    np.save(OUTPUT_NPY, angles)
    print(f"\n轨迹: {OUTPUT_NPY} ({os.path.getsize(OUTPUT_NPY)} bytes)")
    motion = MotionData(
        joint_names=list(BOOSTER_T1_JOINT_NAMES), fps=FPS_TARGET,
        num_frames=num_frames, angles=angles,
        timestamps=np.arange(num_frames) / FPS_TARGET,
    )
    print(f"MotionData: {motion.num_joints} 关节, {motion.num_frames} 帧, {motion.duration:.1f}s")


if __name__ == "__main__":
    result = extract_trajectory(VIDEO_PATH)
    if result is not None:
        angles, num_frames = result
        save_trajectory(angles, num_frames)
        print("\n运行 python scripts/demo_play.py 查看效果")

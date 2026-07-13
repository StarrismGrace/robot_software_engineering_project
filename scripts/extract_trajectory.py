"""
从舞蹈视频提取 Booster T1 关节轨迹 (ONNX 姿态估计)
用法: python scripts/extract_trajectory.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import onnxruntime as ort
import json

import mujoco
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES

VIDEO_PATH = "inputs/dance.mp4"
MODEL_PATH = "models/yolov8n-pose.onnx"
OUTPUT_NPY = "outputs/trajectory.npy"
FPS_TARGET = 30

# COCO 17关键点
NOSE = 0; L_EYE = 1; R_EYE = 2; L_EAR = 3; R_EAR = 4
L_SHOULDER = 5; R_SHOULDER = 6
L_ELBOW = 7; R_ELBOW = 8
L_WRIST = 9; R_WRIST = 10
L_HIP = 11; R_HIP = 12
L_KNEE = 13; R_KNEE = 14
L_ANKLE = 15; R_ANKLE = 16

IMG_SIZE = 640


def angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_trajectory(video_path):
    # 加载 ONNX 模型
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"ONNX 模型已加载: {MODEL_PATH}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / FPS_TARGET))
    print(f"视频: {fps:.1f} fps, {total_frames} 帧, 步长={step}")

    # 加载 home 姿态
    mj_model = mujoco.MjModel.from_xml_path("scene.xml")
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    home_qpos = mj_data.qpos[7:].copy()

    all_angles = []
    frame_idx = 0
    saved = 0
    prev_kpts = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]

        # 预处理: resize + normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        # 推理
        outputs = session.run(None, {"images": img})
        preds = outputs[0]  # (1, 56, 8400)

        # 解析: 找最高置信度的检测
        preds = preds[0]  # (56, 8400)
        scores = preds[4, :]  # bbox 置信度
        best_idx = int(np.argmax(scores))

        if scores[best_idx] < 0.3:
            if prev_kpts is not None:
                kpts = prev_kpts
            else:
                frame_idx += 1
                continue
        else:
            # 提取关键点 (17个, 每个3个值: x, y, conf)
            kpts = np.zeros((17, 2))
            for k in range(17):
                kx = preds[5 + k * 3, best_idx]
                ky = preds[5 + k * 3 + 1, best_idx]
                # 反归一化
                kpts[k, 0] = (kx / IMG_SIZE) * w
                kpts[k, 1] = (ky / IMG_SIZE) * h
            prev_kpts = kpts.copy()

        # ---- 计算关节角度 ----
        mid_hip = (kpts[L_HIP] + kpts[R_HIP]) / 2
        mid_shoulder = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2
        scale = np.linalg.norm(mid_shoulder - mid_hip) + 1e-8

        # 头部
        head_yaw = (kpts[NOSE, 0] - mid_shoulder[0]) / scale
        head_pitch = (kpts[NOSE, 1] - mid_shoulder[1]) / scale

        # 左臂
        ls = np.array([*kpts[L_SHOULDER], 0]); le = np.array([*kpts[L_ELBOW], 0]); lw = np.array([*kpts[L_WRIST], 0])
        l_upper = le - ls; l_forearm = lw - le
        la_pitch = np.arctan2(l_upper[1], l_upper[0] + 1e-8)
        la_roll = np.arctan2(l_upper[2], np.linalg.norm(l_upper[:2]) + 1e-8)
        la_elbow = angle_between(-l_upper, l_forearm)

        # 右臂
        rs = np.array([*kpts[R_SHOULDER], 0]); re = np.array([*kpts[R_ELBOW], 0]); rw = np.array([*kpts[R_WRIST], 0])
        r_upper = re - rs; r_forearm = rw - re
        ra_pitch = np.arctan2(r_upper[1], r_upper[0] + 1e-8)
        ra_roll = np.arctan2(r_upper[2], np.linalg.norm(r_upper[:2]) + 1e-8)
        ra_elbow = angle_between(-r_upper, r_forearm)

        # 左腿
        lh = np.array([*kpts[L_HIP], 0]); lk = np.array([*kpts[L_KNEE], 0]); laa = np.array([*kpts[L_ANKLE], 0])
        l_thigh = lk - lh; l_shin = laa - lk
        lh_pitch = np.arctan2(l_thigh[1], l_thigh[0] + 1e-8)
        lh_roll = np.arctan2(l_thigh[2], np.linalg.norm(l_thigh[:2]) + 1e-8)
        l_knee = angle_between(-l_thigh, l_shin)

        # 右腿
        rh = np.array([*kpts[R_HIP], 0]); rk = np.array([*kpts[R_KNEE], 0]); raa = np.array([*kpts[R_ANKLE], 0])
        r_thigh = rk - rh; r_shin = raa - rk
        rh_pitch = np.arctan2(r_thigh[1], r_thigh[0] + 1e-8)
        rh_roll = np.arctan2(r_thigh[2], np.linalg.norm(r_thigh[:2]) + 1e-8)
        r_knee = angle_between(-r_thigh, r_shin)

        # 躯干
        torso = mid_shoulder - mid_hip
        waist = np.arctan2(torso[0], torso[1] + 1e-8)

        # ---- 映射到 23 个执行器 ----
        angles = home_qpos.copy()
        angles[0] = np.clip(head_yaw, -1.5, 1.5)
        angles[1] = np.clip(-head_pitch * 0.5, -0.3, 1.2)
        angles[2] = np.clip(-la_pitch, -3.0, 1.2)
        angles[3] = np.clip(la_roll, -1.7, 1.5)
        angles[4] = np.clip(la_elbow - 0.3, -2.2, 2.2)
        angles[5] = 0.0
        angles[6] = np.clip(-ra_pitch, -3.0, 1.2)
        angles[7] = np.clip(ra_roll, -1.5, 1.7)
        angles[8] = np.clip(ra_elbow - 0.3, -2.2, 2.2)
        angles[9] = 0.0
        angles[10] = np.clip(waist, -1.5, 1.5)
        angles[11] = np.clip(lh_pitch, -1.8, 1.5)
        angles[12] = np.clip(lh_roll, -0.2, 1.5)
        angles[13] = 0.0
        angles[14] = np.clip(l_knee, 0, 2.3)
        angles[15] = 0.0
        angles[16] = 0.0
        angles[17] = np.clip(rh_pitch, -1.8, 1.5)
        angles[18] = np.clip(rh_roll, -1.5, 0.2)
        angles[19] = 0.0
        angles[20] = np.clip(r_knee, 0, 2.3)
        angles[21] = 0.0
        angles[22] = 0.0

        all_angles.append(angles - home_qpos)
        saved += 1
        frame_idx += 1
        if saved % 30 == 0:
            print(f"  检测到人体, 已处理 {saved} 帧...")

    cap.release()

    if not all_angles:
        print("错误: 未检测到人体!")
        return None

    angles_array = np.array(all_angles)

    # 平滑
    from scipy.signal import savgol_filter
    for j in range(23):
        try:
            w = min(11, max(3, len(all_angles) // 2))
            if w % 2 == 0:
                w -= 1
            if w >= 3:
                angles_array[:, j] = savgol_filter(angles_array[:, j], w, 2)
        except Exception:
            pass

    print(f"提取完成: {saved} 帧, shape={angles_array.shape}")
    return angles_array, saved


def save_trajectory(angles, num_frames):
    import os
    os.makedirs("outputs", exist_ok=True)
    np.save(OUTPUT_NPY, angles)
    print(f"轨迹: {OUTPUT_NPY} ({os.path.getsize(OUTPUT_NPY)} bytes)")
    motion = MotionData(
        joint_names=list(BOOSTER_T1_JOINT_NAMES), fps=FPS_TARGET,
        num_frames=num_frames, angles=angles,
        timestamps=np.arange(num_frames) / FPS_TARGET,
    )
    print(f"MotionData: {motion.num_joints} joints, {motion.num_frames} frames, {motion.duration:.1f}s")


if __name__ == "__main__":
    result = extract_trajectory(VIDEO_PATH)
    if result is not None:
        angles, num_frames = result
        save_trajectory(angles, num_frames)
        print("\n完成! 运行 python scripts/demo_play.py 或 demo_video.py")

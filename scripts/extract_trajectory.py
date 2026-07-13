"""
从舞蹈视频提取 Booster T1 关节轨迹 (MediaPipe 3D + 中性姿态校准)
用法: python scripts/extract_trajectory.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mtp
from mediapipe.tasks.python import vision

import mujoco
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES
from scipy.signal import savgol_filter

VIDEO_PATH = "inputs/dance.mp4"
OUTPUT_NPY = "outputs/trajectory.npy"
FPS_TARGET = 30

# MediaPipe 关键点
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_trajectory(video_path):
    # PoseLandmarker
    base_options = mtp.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps_in / FPS_TARGET))

    # home
    mj_model = mujoco.MjModel.from_xml_path("scene.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    home_ctrl = mj_data.ctrl.copy()

    all_angles_raw = []  # 存储每帧的原始角度
    frame_idx = 0
    saved = 0

    print(f"视频: {fps_in:.0f}fps, {total_frames}帧")
    print("第一遍: 提取3D姿态 + 计算原始关节角...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx / fps_in * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            wlm = result.pose_world_landmarks[0]
            k = np.array([[wlm[i].x, wlm[i].y, wlm[i].z] for i in range(33)])
        elif all_angles_raw:
            # 用上一帧
            angle_vec = all_angles_raw[-1].copy()
            all_angles_raw.append(angle_vec)
            saved += 1
            frame_idx += 1
            continue
        else:
            frame_idx += 1
            continue

        # 中心点
        m_sh = (k[L_SHOULDER] + k[R_SHOULDER]) / 2
        m_hip = (k[L_HIP] + k[R_HIP]) / 2

        # ---- 计算原始关节角度（相对于身体坐标系） ----
        raw = np.zeros(15)  # 15个核心角度

        # 头 Yaw / Pitch
        hp = k[NOSE] - m_sh
        raw[0] = np.arctan2(hp[0], abs(hp[2]) + 1e-3)   # head_yaw
        raw[1] = np.arctan2(hp[1], abs(hp[2]) + 1e-3)    # head_pitch

        # 左臂
        lu = k[L_ELBOW] - k[L_SHOULDER]
        lf = k[L_WRIST] - k[L_ELBOW]
        raw[2] = np.arctan2(lu[1], abs(lu[2]) + 1e-3)   # L_Shoulder_Pitch
        raw[3] = np.arctan2(lu[0], abs(lu[2]) + 1e-3)   # L_Shoulder_Roll
        raw[4] = angle_between(-lu, lf)                   # L_Elbow

        # 右臂
        ru = k[R_ELBOW] - k[R_SHOULDER]
        rf = k[R_WRIST] - k[R_ELBOW]
        raw[5] = np.arctan2(ru[1], abs(ru[2]) + 1e-3)   # R_Shoulder_Pitch
        raw[6] = np.arctan2(ru[0], abs(ru[2]) + 1e-3)   # R_Shoulder_Roll
        raw[7] = angle_between(-ru, rf)                   # R_Elbow

        # 左腿
        lt = k[L_KNEE] - k[L_HIP]
        ls = k[L_ANKLE] - k[L_KNEE]
        raw[8] = np.arctan2(lt[1], abs(lt[2]) + 1e-3)    # L_Hip_Pitch
        raw[9] = np.arctan2(lt[0], abs(lt[2]) + 1e-3)    # L_Hip_Roll
        raw[10] = angle_between(-lt, ls)                   # L_Knee

        # 右腿
        rt = k[R_KNEE] - k[R_HIP]
        r_shi = k[R_ANKLE] - k[R_KNEE]
        raw[11] = np.arctan2(rt[1], abs(rt[2]) + 1e-3)   # R_Hip_Pitch
        raw[12] = np.arctan2(rt[0], abs(rt[2]) + 1e-3)   # R_Hip_Roll
        raw[13] = angle_between(-rt, r_shi)                # R_Knee

        # 躯干
        tv = m_sh - m_hip
        raw[14] = np.arctan2(tv[0], abs(tv[2]) + 1e-3)   # Waist_yaw

        all_angles_raw.append(raw)
        saved += 1
        frame_idx += 1
        if saved % 50 == 0:
            print(f"  {saved} 帧...")

    cap.release()
    detector.close()

    if saved < 10:
        print(f"不足: {saved}")
        return None

    raw_arr = np.array(all_angles_raw)  # (T, 15)

    # ---- 第二遍: 用中位数作为中性姿态，计算偏移 ----
    neutral = np.median(raw_arr, axis=0)  # 中位数比均值更稳定
    print(f"\n第二遍: 中性姿态校准, {saved} 帧 → 机器人关节")

    all_offsets = []
    for t in range(saved):
        delta = raw_arr[t] - neutral  # 相对于中性姿态的偏移

        angles = home_ctrl.copy()

        # Head
        angles[0] = np.clip(float(delta[0] * 1.5), -1.5, 1.5)
        angles[1] = np.clip(float(-delta[1] * 0.6), -0.3, 1.2)

        # Left Arm
        angles[2] = np.clip(float(-delta[2] * 1.5), -3.0, 1.2)
        angles[3] = np.clip(float(delta[3] * 1.5), -1.7, 1.5)
        angles[4] = np.clip(float((delta[4]) * 1.2), -2.2, 2.2)
        angles[5] = home_ctrl[5]

        # Right Arm
        angles[6] = np.clip(float(-delta[5] * 1.5), -3.0, 1.2)
        angles[7] = np.clip(float(delta[6] * 1.5), -1.5, 1.7)
        angles[8] = np.clip(float((delta[7]) * 1.2), -2.2, 2.2)
        angles[9] = home_ctrl[9]

        # Waist
        angles[10] = np.clip(float(delta[14] * 0.8), -1.5, 1.5)

        # Left Leg
        angles[11] = np.clip(float(delta[8] * 1.0), -1.8, 1.5)
        angles[12] = np.clip(float(delta[9] * 1.0), -0.2, 1.5)
        angles[13] = home_ctrl[13]
        angles[14] = np.clip(float(delta[10] * 0.8), 0, 2.3)
        angles[15] = home_ctrl[15]
        angles[16] = home_ctrl[16]

        # Right Leg
        angles[17] = np.clip(float(delta[11] * 1.0), -1.8, 1.5)
        angles[18] = np.clip(float(delta[12] * 1.0), -1.5, 0.2)
        angles[19] = home_ctrl[19]
        angles[20] = np.clip(float(delta[13] * 0.8), 0, 2.3)
        angles[21] = home_ctrl[21]
        angles[22] = home_ctrl[22]

        all_offsets.append(angles - home_ctrl)

    # 平滑
    offsets = np.array(all_offsets)
    for j in range(23):
        try:
            w = min(9, max(3, saved // 3))
            if w % 2 == 0:
                w -= 1
            if w >= 3:
                offsets[:, j] = savgol_filter(offsets[:, j], w, 2)
        except Exception:
            pass

    # 统计
    act_names = [
        "AAHead_yaw", "Head_pitch",
        "L_Shoulder_Pitch", "L_Shoulder_Roll", "L_Elbow_Pitch", "L_Elbow_Yaw",
        "R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Elbow_Pitch", "R_Elbow_Yaw",
        "Waist",
        "L_Hip_Pitch", "L_Hip_Roll", "L_Hip_Yaw", "L_Knee_Pitch", "L_Ankle_Pitch", "L_Ankle_Roll",
        "R_Hip_Pitch", "R_Hip_Roll", "R_Hip_Yaw", "R_Knee_Pitch", "R_Ankle_Pitch", "R_Ankle_Roll",
    ]
    print(f"\n完成: {saved} 帧")
    print("运动关节:")
    for j in range(23):
        rng = offsets[:, j].max() - offsets[:, j].min()
        if rng > 0.02:
            print(f"  {act_names[j]:20s} [{offsets[:, j].min():+.2f} ~ {offsets[:, j].max():+.2f}]")

    return offsets, saved


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
    print(f"MotionData: {motion.num_joints}关节, {motion.num_frames}帧, {motion.duration:.1f}s")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    result = extract_trajectory(VIDEO_PATH)
    if result is not None:
        angles, num_frames = result
        save_trajectory(angles, num_frames)
        print("\npython scripts/demo_play.py 或 demo_video.py")

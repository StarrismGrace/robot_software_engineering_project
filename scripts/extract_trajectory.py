"""
从舞蹈视频提取 Booster T1 关节轨迹 (MediaPipe 3D 姿态)
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

# MediaPipe 33个关键点索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32
L_EAR, R_EAR = 7, 8
MID_HIP = -1  # 左右髋中点（虚拟）


def angle_between_3d(v1, v2):
    """3D向量夹角"""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def extract_trajectory(video_path):
    # 加载模型
    with open("models/pose_landmarker.task", "wb") as f:
        pass  # placehold

    # MediaPipe PoseLandmarker 配置
    base_options = mtp.BaseOptions(
        model_asset_path="models/pose_landmarker_lite.task"
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps_in / FPS_TARGET))
    print(f"视频: {fps_in:.0f}fps, {total}帧, 步长={step}")

    # home 姿态
    mj_model = mujoco.MjModel.from_xml_path("scene.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    home_ctrl = mj_data.ctrl.copy()

    all_world = []  # 存储3D世界坐标
    frame_idx = 0
    saved = 0

    # 第一遍: 提取3D姿态
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
            wlm = result.pose_world_landmarks[0]  # 3D 世界坐标(米)
            kpts_3d = np.zeros((33, 3))
            for i in range(33):
                kpts_3d[i] = [wlm[i].x, wlm[i].y, wlm[i].z]
            all_world.append(kpts_3d)
            saved += 1
        elif all_world:
            all_world.append(all_world[-1].copy())  # 保持上一帧
            saved += 1

        frame_idx += 1
        if saved % 50 == 0:
            print(f"  3D姿态: {saved} 帧...")

    cap.release()
    detector.close()

    if saved < 10:
        print(f"3D姿态不足: {saved}")
        return None
    print(f"第一遍: {saved} 帧 3D 姿态已提取")

    # ---- 第二遍: 3D关节角 → 机器人关节 ----
    all_offsets = []

    for t in range(saved):
        k = all_world[t]

        # 中心点
        m_sh = (k[L_SHOULDER] + k[R_SHOULDER]) / 2
        m_hip = (k[L_HIP] + k[R_HIP]) / 2
        body_h = max(np.linalg.norm(m_sh - m_hip), 0.1)

        # ---- 头部 (Nose相对肩膀) ----
        head_pos = k[NOSE] - m_sh
        head_yaw = np.arctan2(head_pos[0], abs(head_pos[2]) + 1e-3)
        head_pitch = np.arctan2(head_pos[1], abs(head_pos[2]) + 1e-3)

        # ---- 左臂 ----
        l_up = k[L_ELBOW] - k[L_SHOULDER]
        l_fore = k[L_WRIST] - k[L_ELBOW]
        # 肩Pitch: 上臂在矢状面(YZ)的角度
        l_sh_pitch = np.arctan2(l_up[1], abs(l_up[2]) + 1e-3)
        # 肩Roll: 上臂在冠状面(XZ)的角度
        l_sh_roll = np.arctan2(l_up[0], abs(l_up[2]) + 1e-3)
        # 肘Pitch
        l_el = angle_between_3d(-l_up, l_fore)

        # ---- 右臂 ----
        r_up = k[R_ELBOW] - k[R_SHOULDER]
        r_fore = k[R_WRIST] - k[R_ELBOW]
        r_sh_pitch = np.arctan2(r_up[1], abs(r_up[2]) + 1e-3)
        r_sh_roll = np.arctan2(r_up[0], abs(r_up[2]) + 1e-3)
        r_el = angle_between_3d(-r_up, r_fore)

        # ---- 左腿 ----
        l_th = k[L_KNEE] - k[L_HIP]
        l_shin = k[L_ANKLE] - k[L_KNEE]
        l_hip_pitch = np.arctan2(l_th[1], abs(l_th[2]) + 1e-3)
        l_hip_roll = np.arctan2(l_th[0], abs(l_th[2]) + 1e-3)
        l_knee = angle_between_3d(-l_th, l_shin)

        # ---- 右腿 ----
        r_th = k[R_KNEE] - k[R_HIP]
        r_shin = k[R_ANKLE] - k[R_KNEE]
        r_hip_pitch = np.arctan2(r_th[1], abs(r_th[2]) + 1e-3)
        r_hip_roll = np.arctan2(r_th[0], abs(r_th[2]) + 1e-3)
        r_knee = angle_between_3d(-r_th, r_shin)

        # ---- 躯干 ----
        torso_vec = m_sh - m_hip
        waist_yaw = np.arctan2(torso_vec[0], abs(torso_vec[2]) + 1e-3)

        # ---- 映射到23个执行器 ----
        angles = home_ctrl.copy()
        angles[0] = np.clip(float(head_yaw * 1.5), -1.5, 1.5)    # AAHead_yaw
        angles[1] = np.clip(float(-head_pitch * 0.6), -0.3, 1.2) # Head_pitch
        angles[2] = np.clip(float(-l_sh_pitch * 1.5), -3.0, 1.2) # L_Shoulder_Pitch
        angles[3] = np.clip(float(l_sh_roll * 1.5), -1.7, 1.5)   # L_Shoulder_Roll
        angles[4] = np.clip(float((l_el - np.pi / 3) * 1.2), -2.2, 2.2) # L_Elbow_Pitch
        angles[5] = home_ctrl[5]
        angles[6] = np.clip(float(-r_sh_pitch * 1.5), -3.0, 1.2) # R_Shoulder_Pitch
        angles[7] = np.clip(float(r_sh_roll * 1.5), -1.5, 1.7)   # R_Shoulder_Roll
        angles[8] = np.clip(float((r_el - np.pi / 3) * 1.2), -2.2, 2.2) # R_Elbow_Pitch
        angles[9] = home_ctrl[9]
        angles[10] = np.clip(float(waist_yaw * 0.8), -1.5, 1.5)  # Waist
        angles[11] = np.clip(float(l_hip_pitch * 1.0), -1.8, 1.5) # L_Hip_Pitch
        angles[12] = np.clip(float(l_hip_roll * 1.0), -0.2, 1.5)  # L_Hip_Roll
        angles[13] = home_ctrl[13]
        angles[14] = np.clip(float(l_knee * 0.8), 0, 2.3)         # L_Knee_Pitch
        angles[15] = home_ctrl[15]
        angles[16] = home_ctrl[16]
        angles[17] = np.clip(float(r_hip_pitch * 1.0), -1.8, 1.5) # R_Hip_Pitch
        angles[18] = np.clip(float(r_hip_roll * 1.0), -1.5, 0.2)  # R_Hip_Roll
        angles[19] = home_ctrl[19]
        angles[20] = np.clip(float(r_knee * 0.8), 0, 2.3)         # R_Knee_Pitch
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
    print(f"\n完成: {saved} 帧 3D轨迹")
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
        print("\n运行 python scripts/demo_play.py 查看")

"""
从舞蹈视频生成 Booster T1 关节轨迹（光流追踪方案）
用法: python scripts/extract_trajectory.py
无需任何外部模型，仅用 OpenCV + NumPy
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import json

import mujoco
from common.motion_data import MotionData, BOOSTER_T1_JOINT_NAMES

VIDEO_PATH = "inputs/dance.mp4"
OUTPUT_NPY = "outputs/trajectory.npy"
OUTPUT_JSON = "outputs/trajectory.json"
FPS_TARGET = 30


def extract_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / FPS_TARGET))
    print(f"视频: {fps:.1f} fps, {total_frames} 帧, 采样步长={step}")

    # 加载 MuJoCo 模型获取 home 姿态
    mj_model = mujoco.MjModel.from_xml_path("scene.xml")
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    home_qpos = mj_data.qpos[7:].copy()

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频")
        return None

    h, w = prev_frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 在画面中检测特征点（人体轮廓点）
    # 用边缘检测 + 网格采样获取追踪点
    edges = cv2.Canny(prev_gray, 50, 150)
    grid_y, grid_x = np.mgrid[80:h - 80:20, 80:w - 80:20]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # 过滤：只保留边缘附近的点
    valid = []
    for x, y in grid_points:
        roi = edges[max(0, y - 5): min(h, y + 5), max(0, x - 5): min(w, x + 5)]
        if roi.mean() > 20:
            valid.append([x, y])
    prev_points = np.array(valid, dtype=np.float32).reshape(-1, 1, 2)

    if len(prev_points) < 10:
        print("特征点太少，使用全网格")
        prev_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32).reshape(-1, 1, 2)

    print(f"追踪 {len(prev_points)} 个特征点")

    all_angles = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 光流追踪
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if next_points is None or len(next_points) < 5:
            frame_idx += 1
            continue

        # 计算运动向量
        good_prev = prev_points[status[:, 0] == 1]
        good_next = next_points[status[:, 0] == 1]
        motions = good_next - good_prev  # (N, 1, 2)

        if len(motions) < 3:
            frame_idx += 1
            continue

        # 按图像区域分组（5个区域：头、左臂、右臂、左腿、右腿）
        centers = good_prev[:, 0, :]
        mid_x, mid_y = w / 2, h / 2

        def region_mean(cx_min, cx_max, cy_min, cy_max):
            mask = (
                (centers[:, 0] >= cx_min) & (centers[:, 0] < cx_max)
                & (centers[:, 1] >= cy_min) & (centers[:, 1] < cy_max)
            )
            if mask.sum() == 0:
                return 0.0, 0.0, 1
            mx = motions[mask, 0, 0].mean() / w
            my = motions[mask, 0, 1].mean() / h
            return mx, my, mask.sum()

        # 头部区域(上中)
        head_mx, head_my, _ = region_mean(w * 0.3, w * 0.7, 0, h * 0.3)
        # 左臂(左上 + 左中)
        la_mx, la_my, _ = region_mean(0, w * 0.4, h * 0.1, h * 0.55)
        # 右臂(右上 + 右中)
        ra_mx, ra_my, _ = region_mean(w * 0.6, w, h * 0.1, h * 0.55)
        # 躯干(中中)
        torso_mx, torso_my, _ = region_mean(w * 0.3, w * 0.7, h * 0.25, h * 0.55)
        # 左腿(左下)
        ll_mx, ll_my, _ = region_mean(0, w * 0.45, h * 0.55, h)
        # 右腿(右下)
        rl_mx, rl_my, _ = region_mean(w * 0.55, w, h * 0.55, h)

        # ---- 将运动映射到 23 个关节角度（叠加在 home 姿态上） ----
        angles = home_qpos.copy()
        scale = 2.5  # 运动幅度缩放

        # 头 (0, 1): AAHead_yaw, Head_pitch
        angles[0] += np.clip(head_mx * scale, -0.6, 0.6)
        angles[1] += np.clip(-head_my * scale, -0.3, 0.3)

        # 左臂 (2,3,4,5): Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw
        angles[2] += np.clip(-la_my * scale, -1.0, 0.8)
        angles[3] += np.clip(la_mx * scale * 0.5, -0.5, 0.5)
        angles[4] += np.clip(np.sqrt(la_mx ** 2 + la_my ** 2) * scale, -1.0, 1.0)
        angles[5] += 0.0

        # 右臂 (6,7,8,9)
        angles[6] += np.clip(-ra_my * scale, -1.0, 0.8)
        angles[7] += np.clip(ra_mx * scale * 0.5, -0.5, 0.5)
        angles[8] += np.clip(np.sqrt(ra_mx ** 2 + ra_my ** 2) * scale, -1.0, 1.0)
        angles[9] += 0.0

        # 躯干 (10): Waist
        angles[10] += np.clip(torso_mx * scale, -0.5, 0.5)

        # 左腿 (11-16): Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll
        angles[11] += np.clip(-ll_my * scale, -0.8, 0.8)
        angles[12] += np.clip(ll_mx * scale * 0.3, -0.3, 0.3)
        angles[13] += 0.0
        angles[14] += np.clip(np.sqrt(ll_mx ** 2 + ll_my ** 2) * scale, 0, 1.2)
        angles[15] += np.clip(ll_my * scale * 0.5, -0.3, 0.3)
        angles[16] += 0.0

        # 右腿 (17-22)
        angles[17] += np.clip(-rl_my * scale, -0.8, 0.8)
        angles[18] += np.clip(rl_mx * scale * 0.3, -0.3, 0.3)
        angles[19] += 0.0
        angles[20] += np.clip(np.sqrt(rl_mx ** 2 + rl_my ** 2) * scale, 0, 1.2)
        angles[21] += np.clip(rl_my * scale * 0.5, -0.3, 0.3)
        angles[22] += 0.0

        all_angles.append(angles - home_qpos)  # 存偏移量

        # 更新追踪点
        prev_gray = gray
        prev_points = good_next.reshape(-1, 1, 2)

        # 定期重新检测特征点，防止漂移
        if saved % 30 == 0 and saved > 0:
            edges = cv2.Canny(gray, 50, 150)
            new_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
            valid_new = []
            for x, y in new_grid:
                xi, yi = int(x), int(y)
                if 0 <= xi < w and 0 <= yi < h:
                    roi = edges[max(0, yi - 3): min(h, yi + 3), max(0, xi - 3): min(w, xi + 3)]
                    if roi.mean() > 15:
                        valid_new.append([x, y])
            if len(valid_new) > 20:
                prev_points = np.array(valid_new, dtype=np.float32).reshape(-1, 1, 2)

        saved += 1
        frame_idx += 1
        if saved % 30 == 0:
            print(f"  已处理 {saved} 帧...")

    cap.release()

    if not all_angles:
        print("错误: 无法提取运动信息")
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
    print(f"轨迹文件: {OUTPUT_NPY} ({os.path.getsize(OUTPUT_NPY)} bytes)")

    data = {
        "joint_names": list(BOOSTER_T1_JOINT_NAMES),
        "fps": FPS_TARGET,
        "num_frames": num_frames,
        "angles_shape": list(angles.shape),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"轨迹文件: {OUTPUT_JSON}")

    motion = MotionData(
        joint_names=list(BOOSTER_T1_JOINT_NAMES),
        fps=FPS_TARGET,
        num_frames=num_frames,
        angles=angles,
        timestamps=np.arange(num_frames) / FPS_TARGET,
    )
    print(f"MotionData OK: {motion.num_joints} joints, {motion.num_frames} frames, {motion.duration:.1f}s")


if __name__ == "__main__":
    result = extract_trajectory(VIDEO_PATH)
    if result is not None:
        angles, num_frames = result
        save_trajectory(angles, num_frames)
        print(f"\n轨迹已生成! 运行:")
        print(f"  python scripts/demo_play.py")

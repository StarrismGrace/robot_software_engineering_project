"""
数据清洗模块

对原始姿态数据进行去噪、插值、平滑、异常值剔除。
"""

import numpy as np
from scipy import interpolate, signal
from sklearn.neighbors import LocalOutlierFactor

from common.interfaces import MotionCleaner
from common.motion_data import MotionData


class MotionCleanerImpl(MotionCleaner):
    """MotionCleaner 接口的实现类"""

    def clean(self, motion: MotionData, **kwargs) -> MotionData:
        """
        清洗原始动作数据。

        步骤：
        1. 检测异常值（调用 detect_outliers）
        2. 对异常值进行插值替换（线性/样条插值）
        3. 对整体数据做平滑滤波（Savitzky-Golay）
        4. 返回清洗后的 MotionData

        Parameters
        ----------
        motion : MotionData
            原始姿态数据
        **kwargs :
            可选参数：
            - smooth_window: int, 平滑窗口大小（默认 5）
            - interpolate_method: str, 插值方法（'linear' / 'cubic'）
            - filter_type: str, 滤波类型（'savgol' / 'butter'）

        Returns
        -------
        MotionData
            清洗后的数据
        """
        # 如果没有位置数据，直接返回
        if motion.positions is None:
            return motion

        # 1. 获取参数
        smooth_window = kwargs.get("smooth_window", 5)
        interpolate_method = kwargs.get("interpolate_method", "linear")
        filter_type = kwargs.get("filter_type", "savgol")

        # 2. 复制数据，避免修改原始数据
        positions = motion.positions.copy()
        T, J, _ = positions.shape

        # 3. 检测异常值
        outlier_mask = self.detect_outliers(motion)

        # 4. 对每个关节的每个坐标轴进行插值修复
        for j in range(J):
            for axis in range(3):
                data = positions[:, j, axis]

                # 找出异常值位置
                outlier_indices = np.where(outlier_mask[:, j])[0]

                if len(outlier_indices) > 0:
                    # 将异常值设为 NaN
                    data_with_nan = data.copy()
                    data_with_nan[outlier_indices] = np.nan

                    # 插值修复
                    valid_mask = ~np.isnan(data_with_nan)
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = data_with_nan[valid_mask]

                    if len(valid_indices) > 1:
                        # 使用 scipy 插值
                        interp_func = interpolate.interp1d(
                            valid_indices,
                            valid_values,
                            kind=interpolate_method,
                            fill_value="extrapolate",
                            bounds_error=False,
                        )
                        repaired = interp_func(np.arange(T))
                        positions[:, j, axis] = repaired
                    elif len(valid_indices) == 1:
                        # 只有一个有效值，填充为常数
                        positions[:, j, axis] = valid_values[0]

        # 5. 平滑滤波（先确保没有 NaN）
        positions = self._smooth_positions(positions, smooth_window, filter_type)

        # 6. 创建新的 MotionData
        cleaned_motion = MotionData(
            joint_names=motion.joint_names.copy(),
            fps=motion.fps,
            num_frames=motion.num_frames,
            positions=positions,
            angles=motion.angles.copy() if motion.angles is not None else None,
            timestamps=(
                motion.timestamps.copy() if motion.timestamps is not None else None
            ),
        )

        return cleaned_motion

    def detect_outliers(self, motion: MotionData, threshold: float = 3.0) -> np.ndarray:
        """
        检测并标记异常帧/关节。

        使用速度 Z-score 方法：
        1. 计算每个关节在每个时间步的速度（差分）
        2. 计算 Z-score，标记 |Z| > threshold 的点

        Parameters
        ----------
        motion : MotionData
            待检测的动作数据
        threshold : float
            Z-score 阈值，默认 3.0

        Returns
        -------
        np.ndarray
            布尔掩码，形状 (T, J)，True 表示异常值
        """
        if motion.positions is None:
            return np.zeros((motion.num_frames, len(motion.joint_names)), dtype=bool)

        positions = motion.positions
        T, J, _ = positions.shape

        # 初始化异常掩码
        outlier_mask = np.zeros((T, J), dtype=bool)

        # 对每个关节和每个坐标轴计算速度
        for j in range(J):
            for axis in range(3):
                data = positions[:, j, axis]

                # 检查是否有 NaN，如果有则用线性插值填补
                if not np.isfinite(data).all():
                    valid_mask = np.isfinite(data)
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = data[valid_mask]
                    if len(valid_indices) >= 2:
                        interp_func = interpolate.interp1d(
                            valid_indices,
                            valid_values,
                            kind="linear",
                            fill_value="extrapolate",
                            bounds_error=False,
                        )
                        data = interp_func(np.arange(T))
                    elif len(valid_indices) == 1:
                        data = np.full(T, valid_values[0])
                    else:
                        data = np.zeros(T)

                # 计算速度（差分）
                velocity = np.diff(data, prepend=data[0])

                # 计算 Z-score
                mean = np.mean(velocity)
                std = np.std(velocity)

                if std > 1e-6:  # 避免除以零
                    z_scores = np.abs((velocity - mean) / std)
                    # 标记异常值
                    outlier_mask[:, j] = outlier_mask[:, j] | (z_scores > threshold)

        return outlier_mask

    def _smooth_positions(
        self, positions: np.ndarray, window: int, filter_type: str
    ) -> np.ndarray:
        """
        对位置数据进行平滑滤波。

        Parameters
        ----------
        positions : np.ndarray
            位置数据，形状 (T, J, 3)
        window : int
            窗口大小
        filter_type : str
            滤波类型（'savgol' / 'butter'）

        Returns
        -------
        np.ndarray
            平滑后的位置数据
        """
        T, J, _ = positions.shape

        # 如果帧数太少，不做平滑
        if T < window + 2:
            return positions

        smoothed = positions.copy()

        for j in range(J):
            for axis in range(3):
                data = positions[:, j, axis]

                # 检查是否有 NaN 或 Inf
                if not np.isfinite(data).all():
                    # 如果有非有限值，先用线性插值填补
                    valid_mask = np.isfinite(data)
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = data[valid_mask]

                    if len(valid_indices) >= 2:
                        interp_func = interpolate.interp1d(
                            valid_indices,
                            valid_values,
                            kind="linear",
                            fill_value="extrapolate",
                            bounds_error=False,
                        )
                        data = interp_func(np.arange(T))
                    elif len(valid_indices) == 1:
                        data = np.full(T, valid_values[0])
                    else:
                        data = np.zeros(T)

                # 现在进行平滑
                if filter_type == "savgol":
                    # Savitzky-Golay 滤波
                    if len(data) >= window + 2:
                        # 确保 window 是奇数
                        win = window if window % 2 == 1 else window + 1
                        # 确保 polyorder 小于 window
                        polyorder = min(2, win - 1)
                        smoothed[:, j, axis] = signal.savgol_filter(
                            data, window_length=win, polyorder=polyorder
                        )
                    else:
                        smoothed[:, j, axis] = data
                elif filter_type == "butter":
                    # Butterworth 低通滤波
                    if len(data) > 4:
                        b, a = signal.butter(4, 0.3, btype="low")
                        smoothed[:, j, axis] = signal.filtfilt(b, a, data)
                    else:
                        smoothed[:, j, axis] = data
                else:
                    # 默认移动平均
                    kernel = np.ones(window) / window
                    smoothed[:, j, axis] = np.convolve(data, kernel, mode="same")

        return smoothed

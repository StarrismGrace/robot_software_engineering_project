"""
视频处理模块 - 成员E
职责：读取输入视频，按固定帧率抽帧，输出图像列表
"""
import os
import cv2
import numpy as np
from typing import List, Optional
from common.interfaces import VideoProcessorInterface


class VideoProcessor(VideoProcessorInterface):
    """
    视频处理类，继承自 VideoProcessorInterface
    实现 process() 方法：读取视频 → 抽帧 → 返回图像列表
    """

    def process(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        处理视频文件，返回图像列表

        Args:
            video_path: 视频文件的路径

        Returns:
            List[np.ndarray]: 图像列表，每个元素是 RGB 格式的 numpy 数组

        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: 视频无法打开或解码
        """
        # 1. 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 2. 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 3. 抽帧（每隔 N 帧取一帧，可配置）
        frame_interval = 1  # 每1帧都取，可改为 2 或 3
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                # 将 BGR 转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1

        # 4. 释放资源并返回
        cap.release()
        return frames
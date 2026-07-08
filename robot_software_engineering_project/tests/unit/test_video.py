"""
单元测试 - 视频处理模块 (成员E)
测试 VideoProcessor 是否能正确读取视频并抽帧
"""
import os
import pytest
import numpy as np
from project1_dance.video_processor.extractor import VideoProcessor


class TestVideoProcessor:
    """视频处理模块的单元测试类"""

    def test_extract_frames_valid(self):
        """测试：给定存在的视频文件，应返回非空帧列表"""
        video_path = os.path.join(os.path.dirname(__file__), "test_video1.mp4")
        vp = VideoProcessor()
        frames = vp.extract_frames(video_path)
        
        assert frames is not None, "返回结果不应为 None"
        assert isinstance(frames, list), "返回结果应为列表"
        assert len(frames) > 0, "帧列表不能为空"
        assert isinstance(frames[0], np.ndarray), "每一帧应为 numpy 数组"

    def test_extract_frames_with_fps(self):
        """测试：指定目标帧率抽帧"""
        video_path = os.path.join(os.path.dirname(__file__), "test_video1.mp4")
        vp = VideoProcessor()
        # 先获取元信息
        meta = vp.get_video_metadata(video_path)
        original_fps = meta["fps"]
        # 以一半帧率抽帧
        frames = vp.extract_frames(video_path, fps=original_fps / 2)
        # 帧数应该约为原来的一半（允许误差）
        total_meta = meta["total_frames"]
        expected = total_meta / 2
        assert abs(len(frames) - expected) < 5, f"抽帧数量与预期不符: {len(frames)} vs {expected}"

    def test_get_video_metadata(self):
        """测试：获取视频元信息"""
        video_path = os.path.join(os.path.dirname(__file__), "test_video1.mp4")
        vp = VideoProcessor()
        meta = vp.get_video_metadata(video_path)
        
        assert "fps" in meta
        assert "total_frames" in meta
        assert "duration_sec" in meta
        assert "width" in meta
        assert "height" in meta
        assert meta["total_frames"] > 0
        assert meta["fps"] > 0

    def test_extract_frames_file_not_found(self):
        """测试：给定不存在的文件路径，应抛出 FileNotFoundError"""
        vp = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            vp.extract_frames("non_existent_video.mp4")

    def test_extract_frames_invalid_video(self):
        """测试：给定无效视频文件（如文本文件），应抛出 ValueError"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a video")
            temp_path = f.name
        try:
            vp = VideoProcessor()
            with pytest.raises(ValueError):
                vp.extract_frames(temp_path)
        finally:
            os.remove(temp_path)

    def test_extract_frames_start_end(self):
        """测试：指定起始和结束时间"""
        video_path = os.path.join(os.path.dirname(__file__), "test_video1.mp4")
        vp = VideoProcessor()
        meta = vp.get_video_metadata(video_path)
        # 只取前 1 秒
        frames = vp.extract_frames(video_path, start_sec=0, end_sec=1.0)
        expected = int(meta["fps"] * 1.0)
        assert abs(len(frames) - expected) < 5, f"帧数不符预期: {len(frames)} vs {expected}"
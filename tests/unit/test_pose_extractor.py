import sys
import os
import numpy as np

# 补全项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project1_dance.pose_extractor.extractor import PromptHMRExtractor


def test_pose_extract_single_frame():
    """单帧姿态提取测试"""
    extractor = PromptHMRExtractor()

    # 生成一张假图做快速测试
    dummy_frame = np.random.randint(0, 255, (720, 1080, 3), dtype=np.uint8)
    result = extractor.extract([dummy_frame], fps=30)

    # 验证输出格式符合项目标准
    assert result.num_frames == 1
    assert result.positions is not None
    assert len(result.positions.shape) == 3  # (帧数, 关节数, 3)
    assert result.positions.shape[2] == 3
    print("✅ 单帧姿态提取测试通过")

def test_pose_extract_multi_frames():
    """多帧姿态提取测试"""
    extractor = PromptHMRExtractor()
    frames = [np.random.randint(0, 255, (720, 1080, 3), dtype=np.uint8) for _ in range(10)]
    result = extractor.extract(frames, fps=30)
    
    assert result.num_frames == 10
    assert result.positions.shape == (10, 23, 3)
    print("✅ 多帧姿态提取测试通过")


def test_joint_map():
    """关节映射表测试"""
    extractor = PromptHMRExtractor()
    joint_map = extractor.get_joint_map()
    
    assert isinstance(joint_map, dict)
    assert len(joint_map) > 0
    print("✅ 关节映射表测试通过")

if __name__ == "__main__":
    test_pose_extract_single_frame()
    test_pose_extract_multi_frames()
    test_joint_map()
    print("\n🎉 全部测试通过")

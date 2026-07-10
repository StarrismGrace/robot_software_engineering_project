# Booster T1 — 舞蹈动作重定向与 MuJoCo 仿真演示系统

基于输入舞蹈视频，提取人体姿态信息，完成动作数据清洗与机器人动作重定向，使 **Booster T1** 机器人在 MuJoCo 物理引擎中完成舞蹈动作演示。

> **课程**：软件工程导论 · 短学期  
> **团队规模**：9 人  
> **技术栈**：Python · MuJoCo · OpenCV · NumPy/SciPy

---

## 目录

- [项目简介](#项目简介)
- [团队信息](#团队信息)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [模块负责人](#模块负责人)
- [开发约定](#开发约定)
- [许可证](#许可证)

---

## 项目简介

本系统实现从**舞蹈视频**到**机器人仿真演示**的完整流水线：

```
输入视频 ──▶ 视频抽帧 ──▶ 人体姿态提取 ──▶ 数据清洗 ──▶ 动作重定向 ──▶ MuJoCo 仿真播放
              (成员E)       (成员B)          (成员I)       (成员D)         (成员F)
```

各模块通过统一的数据容器 `MotionData` 进行通信，确保接口一致、可独立开发测试。

---

## 团队信息

| 角色 | 姓名 | 职责 |
|------|------|------|
| 组长 (A) | — | 项目管理、系统集成、公共模块 |
| 成员 B | — | 姿态提取 (`PoseExtractor`) |
| 成员 C | — | 系统测试 |
| 成员 D | — | 动作重定向 (`Retargeting`) |
| 成员 E | — | 视频处理 (`VideoProcessor`) |
| 成员 F | — | MuJoCo 仿真 (`MuJoCoPlayer`) |
| 成员 G | — | 文档撰写 |
| 成员 H | — | 主流程集成与调度 |
| 成员 I | — | 数据清洗 (`MotionCleaner`) |

> 💡 请各成员在开发启动前填入姓名。

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n booster-t1 python=3.10
conda activate booster-t1
```

### 2. 安装依赖

```bash
cd robot_software_engineering_project
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import mujoco; import numpy; import cv2; print('环境就绪')"
```

> **注意**：MuJoCo 在 Linux 上可能需要额外安装系统依赖，请参考 [MuJoCo 官方文档](https://mujoco.readthedocs.io/)。

---

## 快速开始

### Mock 模式（无需视频/模型，立即体验全流程）

```bash
python main.py --mock
```

输出示例：

```
[2026-07-08 10:43:30][INFO ][BoosterT1] ============================================================
[2026-07-08 10:43:30][INFO ][BoosterT1] Booster T1 流水线启动
[2026-07-08 10:43:30][INFO ][BoosterT1]   模式:      Mock 模拟
[2026-07-08 10:43:31][INFO ][BoosterT1] ── Step 1/5: 视频处理 ──
[2026-07-08 10:43:31][INFO ][BoosterT1]   ✓ 已抽取 300 帧，shape=(480,640,3)
[2026-07-08 10:43:31][INFO ][BoosterT1] ── Step 2/5: 姿态提取 ──
[2026-07-08 10:43:31][INFO ][BoosterT1]   ✓ MotionData: 22 关节, 300 帧, 10.0 秒
[2026-07-08 10:43:31][INFO ][BoosterT1] ── Step 3/5: 数据清洗 ──
[2026-07-08 10:43:31][INFO ][BoosterT1]   ✓ 清洗完成: 22 关节, 300 帧
[2026-07-08 10:43:31][INFO ][BoosterT1] ── Step 4/5: 动作重定向 ──
[2026-07-08 10:43:31][INFO ][BoosterT1]   ✓ Robot MotionData: 22 关节, 300 帧
[2026-07-08 10:43:31][INFO ][BoosterT1] ── Step 5/5: MuJoCo 仿真播放 ──
[2026-07-08 10:43:31][INFO ][BoosterT1]   ✓ 仿真完成: 10 帧已渲染
[2026-07-08 10:43:31][INFO ][BoosterT1] 流水线执行成功
```

### 真实视频模式

```bash
# 基本用法
python main.py --video data/your_dance.mp4

# 指定配置文件与输出
python main.py --video data/ballet.mp4 --config my_config.yaml --output outputs/result.mp4

# 跳过某些步骤（调试用）
python main.py --video data/test.mp4 --skip_clean
```

### 运行测试

```bash
pytest tests/ -v
```

---

## 目录结构

```
robot_software_engineering_project/
│
├── main.py                     # 主入口，流水线调度
├── config.yaml                 # 全局配置文件
├── requirements.txt            # Python 依赖清单
├── README.md                   # 你正在读的文件
├── .gitignore
│
├── common/                     # 公共层（组长维护）
│   ├── __init__.py             # 统一导出
│   ├── motion_data.py          # MotionData 数据容器
│   ├── interfaces.py           # 5 个模块抽象接口
│   ├── mock_factory.py         # Mock 数据工厂
│   ├── logger.py               # 统一日志模块
│   └── config_loader.py        # YAML 配置加载
│
├── project1_dance/             # 核心实现（各成员负责）
│   ├── __init__.py
│   ├── video_processor/        # 视频读取与抽帧（成员E）
│   │   └── __init__.py
│   ├── pose_extractor/         # 人体姿态提取（成员B）
│   │   └── __init__.py
│   ├── motion_cleaner/         # 数据清洗（成员I）
│   │   └── __init__.py
│   ├── retargeting/            # 动作重定向（成员D）
│   │   └── __init__.py
│   └── mujoco_player/          # MuJoCo 仿真播放（成员F）
│       └── __init__.py
│
├── tests/                      # 测试（成员C）
│   ├── __init__.py
│   ├── unit/                   # 单元测试
│   │   └── __init__.py
│   └── integration/            # 集成测试
│       └── __init__.py
│
├── outputs/                    # 输出目录（视频、日志）
├── scripts/                    # 工具脚本
├── docs/                       # 文档
└── .github/
    └── pull_request_template.md
```

---

## 模块负责人

| 模块 | 路径 | 接口 | 负责人 | 状态 |
|------|------|------|--------|------|
| 公共数据 & 接口 | `common/` | `MotionData`, `ABC` | 组长 A | ✅ 已完成 |
| 主流程调度 | `main.py` | — | 成员 H | 🚧 待接入 |
| 视频处理 | `project1_dance/video_processor/` | `VideoProcessor` | 成员 E | ⏳ 待开发 |
| 姿态提取 | `project1_dance/pose_extractor/` | `PoseExtractor` | 成员 B | ⏳ 待开发 |
| 数据清洗 | `project1_dance/motion_cleaner/` | `MotionCleaner` | 成员 I | ⏳ 待开发 |
| 动作重定向 | `project1_dance/retargeting/` | `Retargeting` | 成员 D | ⏳ 待开发 |
| MuJoCo 仿真 | `project1_dance/mujoco_player/` | `MuJoCoPlayer` | 成员 F | ⏳ 待开发 |
| 测试 | `tests/` | — | 成员 C | ⏳ 待开发 |
| 文档 | `docs/` | — | 成员 G | ⏳ 待开发 |

> 状态图例：✅ 已交付 · 🚧 进行中 · ⏳ 待启动

---

## 开发约定

### 数据传递

所有模块间统一使用 `MotionData` 传递动作数据，不得使用自定义 dict：

```python
from common import MotionData

# ✓ 正确
def clean(self, motion: MotionData) -> MotionData: ...

# ✗ 错误
def clean(self, data: dict) -> dict: ...
```

### 接口实现

各模块实现对应抽象接口，写在 `project1_dance/<模块>/` 目录下：

```python
from common.interfaces import VideoProcessor

class OpenCVVideoProcessor(VideoProcessor):
    def extract_frames(self, video_path, fps=None, start_sec=0.0, end_sec=None):
        # 你的实现
        ...
```

### 提交规范

- 提交前运行 `black . && flake8 .`
- PR 使用 `.github/pull_request_template.md` 模板
- 确保 `python main.py --mock` 返回退出码 0

---

## 许可证

本项目用于教学目的（软件工程导论课程），不对外发布。

# Booster T1 文档

## 快速导航

| 文档 | 说明 | 读者 |
|------|------|------|
| [README.md](../README.md) | 项目总览与环境配置 | 所有人 |
| [接口规范](../common/interfaces.py) | 5 个模块抽象接口定义 | 模块开发者 |
| [MotionData 规范](../common/motion_data.py) | 统一数据容器格式 | 模块开发者 |
| [配置说明](../config.yaml) | 配置项说明 | 成员 H, 全体 |
| [PR 模板](../.github/pull_request_template.md) | 提交 PR 的自检清单 | 全体 |

## 模块开发文档

各模块完成后，在此目录下补充设计文档：

```
docs/
├── index.md                    ← 当前文件
├── video_processor.md          ← 成员 E 撰写
├── pose_extractor.md           ← 成员 B 撰写
├── motion_cleaner.md           ← 成员 I 撰写
├── retargeting.md              ← 成员 D 撰写
├── mujoco_player.md            ← 成员 F 撰写
└── integration_test.md         ← 成员 C 撰写
```

## 文档撰写规范

1. 每个模块文档至少包含：
   - 算法/方案选择理由
   - 输入输出格式
   - 关键参数说明
   - 已知限制
2. 出现接口变更时必须同步更新文档
3. 图表推荐使用 Mermaid 语法（GitHub 原生渲染）

"""
Booster T1 MuJoCo 仿真播放器
成员F — 实现 MuJoCoPlayer 接口
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from common.interfaces import MuJoCoPlayer
from common.motion_data import MotionData
from common.logger import setup_logger

logger = setup_logger("MuJoCoPlayer")


class MuJoCoPlayerImpl(MuJoCoPlayer):
    """使用 MuJoCo 物理引擎播放 Booster T1 机器人动作数据。"""

    def __init__(self, width: int = 640, height: int = 480):
        self._width = width
        self._height = height
        self._model: Optional["mujoco.MjModel"] = None
        self._data: Optional["mujoco.MjData"] = None
        self._renderer: Optional["mujoco.Renderer"] = None
        self._actuator_ids: List[int] = []
        self._fps: int = 30

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------
    def load_model(self, model_path: Union[str, Path]) -> None:
        """加载 MuJoCo 模型并初始化仿真数据与渲染器。"""
        import mujoco

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, self._height, self._width)

        self._actuator_ids = list(range(self._model.nu))
        logger.info(
            f"模型已加载: {model_path} (actuators={self._model.nu}, qpos={self._model.nq})"
        )

    # ------------------------------------------------------------------
    # play
    # ------------------------------------------------------------------
    def play(
        self,
        robot_motion: MotionData,
        output_path: Optional[Union[str, Path]] = None,
        render: bool = True,
        **kwargs,
    ) -> List[np.ndarray]:
        """播放机器人动作数据，可选导出视频。

        Parameters
        ----------
        robot_motion : MotionData
            关节角度数据（angles 形状 (T, J)）。
        output_path : Optional[Union[str, Path]]
            输出视频路径，None 则不保存。
        render : bool
            是否渲染画面。
        **kwargs
            fps : 播放帧率
            loop : 循环播放次数（默认 1）

        Returns
        -------
        List[np.ndarray]
            渲染帧列表。
        """
        import mujoco

        if self._model is None or self._data is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        angles = robot_motion.angles
        if angles is None:
            raise ValueError("robot_motion.angles 为 None，无法播放")

        T = angles.shape[0]

        if angles.shape[1] != self._model.nu:
            logger.warning(
                f"angles 关节数 ({angles.shape[1]}) 与模型 actuator 数 "
                f"({self._model.nu}) 不匹配，将按较小维度截取"
            )
        J = min(angles.shape[1], self._model.nu)

        fps = kwargs.get("fps", robot_motion.fps)
        loop = kwargs.get("loop", 1)

        self._fps = fps

        # --- 初始化 actuator → joint name 映射 ---
        actuator_map = self._build_actuator_map(robot_motion.joint_names)

        rendered_frames: List[np.ndarray] = []
        video_writer = None

        if output_path is not None:
            import imageio

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                output_path, fps=fps, format="FFMPEG", codec="libx264"
            )
            logger.info(f"视频输出: {output_path}")

        # --- 逐帧播放 ---
        try:
            for _ in range(loop):
                for t in range(T):
                    # 设置控制信号（关节角度）
                    self._data.ctrl[:] = 0.0
                    for j in range(J):
                        actuator_idx = actuator_map[j]
                        self._data.ctrl[actuator_idx] = float(angles[t, j])

                    # 物理步进
                    mujoco.mj_step(self._model, self._data)

                    if render and self._renderer is not None:
                        self._renderer.update_scene(self._data)
                        frame = self._renderer.render()
                        rendered_frames.append(frame.copy())

                        if video_writer is not None:
                            video_writer.append_data(frame)
        finally:
            if video_writer is not None:
                video_writer.close()
                logger.info(
                    f"视频已保存: {output_path} ({len(rendered_frames)} 帧)"
                )

        logger.info(
            f"播放完成: {len(rendered_frames)} 帧, "
            f"{len(rendered_frames) / max(fps, 1):.1f} 秒"
        )
        return rendered_frames

    # ------------------------------------------------------------------
    # get_physics_state
    # ------------------------------------------------------------------
    def get_physics_state(self) -> dict:
        """返回当前物理状态快照。"""
        if self._data is None:
            return {"qpos": np.zeros(0), "qvel": np.zeros(0), "time": 0.0}

        return {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "time": float(self._data.time),
        }

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _build_actuator_map(self, joint_names: List[str]) -> List[int]:
        """建立 joint_names → actuator id 的映射。

        优先按名称匹配，失败则返回 [0, 1, 2, ...] 的直通映射。
        """
        if self._model is None:
            return list(range(len(joint_names)))

        # 收集模型所有 actuator 对应的 joint 名称
        model_actuator_names: List[str] = []
        for act_id in range(self._model.nu):
            joint_id = self._model.actuator_trnid[act_id][0]
            jnt_name = self._model.joint(joint_id).name or f"joint_{joint_id}"
            model_actuator_names.append(jnt_name)

        mapping: List[int] = []
        matched = 0
        for j_idx, name in enumerate(joint_names):
            try:
                act_id = model_actuator_names.index(name)
                matched += 1
            except ValueError:
                # 尝试部分匹配
                act_id = j_idx  # fallback: 按索引直通
                for a_id, a_name in enumerate(model_actuator_names):
                    if name.lower().replace("_", "") == a_name.lower().replace("_", ""):
                        act_id = a_id
                        matched += 1
                        break
            mapping.append(act_id)

        if matched > 0:
            logger.info(
                f"关节映射: {matched}/{len(joint_names)} 个已匹配"
            )
        else:
            logger.info("使用索引直通映射（joint_names 与 actuator 名称不匹配）")

        return mapping

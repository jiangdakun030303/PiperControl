#!/usr/bin/env python3
"""Piper 机械臂逆运动学 (Pinocchio + IPOPT, 基于 URDF 模型).

依赖: pinocchio>=3.0, casadi, numpy
conda install pinocchio=3.6.0 -c conda-forge
pip install casadi
"""

import os
import math
import numpy as np

from typing import Optional

_URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'piperros', 'src', 'piper_description', 'urdf', 'piper_description.urdf'
)
# 如果 SDK 和 piperros 是同级目录, 修正路径
if not os.path.exists(_URDF_PATH):
    _URDF_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'piperros', 'src', 'piper_description', 'urdf', 'piper_description.urdf'
    )


class C_PiperIKPinocchio:
    """基于 URDF 的 Pinocchio 逆运动学求解器."""

    def __init__(self, urdf_path: Optional[str] = None):
        import casadi
        import pinocchio as pin
        from pinocchio import casadi as cpin

        self._casadi = casadi
        self._pin = pin
        self._cpin = cpin

        urdf = urdf_path or _URDF_PATH
        if not os.path.exists(urdf):
            raise FileNotFoundError(f"URDF 不存在: {urdf}")

        # 设置 ROS_PACKAGE_PATH, 使 Pinocchio 能解析 package:// URI
        pkg_dir = os.path.dirname(os.path.dirname(urdf))
        existing = os.environ.get('ROS_PACKAGE_PATH', '')
        if pkg_dir not in existing:
            os.environ['ROS_PACKAGE_PATH'] = f"{pkg_dir}:{existing}" if existing else pkg_dir

        # 1. 加载完整机器人模型
        self._robot_full = pin.RobotWrapper.BuildFromURDF(urdf)
        robot_full = self._robot_full

        # 2. 锁住夹爪关节 (joint7, joint8), 只保留 6-DOF 臂 (用于 IK)
        self._reduced = robot_full.buildReducedRobot(
            list_of_joints_to_lock=['joint7', 'joint8'],
            reference_configuration=np.array([0.0] * robot_full.model.nq),
        )

        # 3. 在 joint6 末端添加 ee frame (与 Pinocchio 原版对齐)
        self._reduced.model.addFrame(
            pin.Frame('ee',
                      self._reduced.model.getJointId('joint6'),
                      pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.0, 0.0, 0.0])),
                      pin.FrameType.OP_FRAME)
        )

        # 4. 构建碰撞检测 (用完整模型, 8关节)
        self._geom_model = pin.buildGeomFromUrdf(
            robot_full.model, urdf, pin.GeometryType.COLLISION)
        # 添加碰撞对: link4-9 vs base_link, link1, link2 (index 0-2)
        for i in range(4, 10):
            for j in range(0, 3):
                self._geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self._geometry_data = pin.GeometryData(self._geom_model)

        # 5. 构建 CasADi 符号模型
        self._cmodel = cpin.Model(self._reduced.model)
        self._cdata = self._cmodel.createData()
        self._cq = casadi.SX.sym("q", self._reduced.model.nq, 1)
        self._cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self._cmodel, self._cdata, self._cq)

        ee_id = self._reduced.model.getFrameId('ee')
        error = casadi.Function(
            "error", [self._cq, self._cTf],
            [casadi.vertcat(
                cpin.log6(self._cdata.oMf[ee_id].inverse() * cpin.SE3(self._cTf)).vector
            )],
        )

        # 5. 构建优化问题
        opti = casadi.Opti()
        var_q = opti.variable(self._reduced.model.nq)
        param_tf = opti.parameter(4, 4)

        err_vec = error(var_q, param_tf)
        pos_err = err_vec[:3]
        ori_err = err_vec[3:]

        w_pos = 1.0
        w_ori = 0.1
        totalcost = casadi.sumsqr(w_pos * pos_err) + casadi.sumsqr(w_ori * ori_err)
        reg = casadi.sumsqr(var_q)

        opti.subject_to(opti.bounded(
            self._reduced.model.lowerPositionLimit,
            var_q,
            self._reduced.model.upperPositionLimit,
        ))
        opti.minimize(20.0 * totalcost + 1e-6 * reg)

        opts = {'ipopt': {'print_level': 0, 'max_iter': 50, 'tol': 1e-4}, 'print_time': False}
        opti.solver("ipopt", opts)

        self._opti = opti
        self._var_q = var_q
        self._param_tf = param_tf
        self._error_func = error
        self._nq = self._reduced.model.nq
        self._lb = self._reduced.model.lowerPositionLimit
        self._ub = self._reduced.model.upperPositionLimit

        # 状态: 初始值 & 历史
        self._q_init = np.zeros(self._nq)
        self._q_history = np.zeros(self._nq)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def solve_full(self, x_m: float, y_m: float, z_m: float,
                   rx_deg: float, ry_deg: float, rz_deg: float,
                   q0: Optional[list[float]] = None,
                   gripper: float = 0.0) -> Optional[list[float]]:
        """6-DOF IK.

        Args:
            x_m, y_m, z_m: 目标位置 (m)
            rx_deg, ry_deg, rz_deg: 目标姿态 RPY (deg)
            q0: 初始关节角 (rad), None=自动选择
            gripper: 夹爪宽度 (m), 用于碰撞检测

        Returns:
            关节角度列表 (rad), 无解返回 None
        """
        T = self._build_target(x_m, y_m, z_m, rx_deg, ry_deg, rz_deg)
        return self._solve(T, q0)

    def solve_position(self, x_m: float, y_m: float, z_m: float,
                       q0: Optional[list[float]] = None) -> Optional[list[float]]:
        """3-DOF 位置 IK (保持当前姿态, 位置权重远大于姿态)."""
        if q0 is not None:
            self._q_init = np.array(q0, dtype=float)

        # 用 FK 获取当前姿态作为参考
        cur_pose = self._forward_kinematics(self._q_init)
        if cur_pose is None:
            return None
        rx_deg = math.degrees(cur_pose[3])
        ry_deg = math.degrees(cur_pose[4])
        rz_deg = math.degrees(cur_pose[5])
        return self.solve_full(x_m, y_m, z_m, rx_deg, ry_deg, rz_deg, q0=q0)

    def fk(self, q: list[float]) -> list[float]:
        """FK: 关节角 (rad) → [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]."""
        result = self._forward_kinematics(np.array(q, dtype=float))
        if result is None:
            return [0.0] * 6
        return list(result)

    def check_collision(self, q: list[float], gripper: float = 0.0) -> bool:
        """检查指定位型是否自碰撞.

        Args:
            q: 6 关节角 (rad)
            gripper: 夹爪开度 (m)

        Returns:
            True=碰撞, False=安全
        """
        q_full = np.concatenate([np.array(q, dtype=float),
                                 np.array([gripper / 2.0, -gripper / 2.0])])
        self._pin.forwardKinematics(self._robot_full.model, self._robot_full.data, q_full)
        self._pin.updateGeometryPlacements(
            self._robot_full.model, self._robot_full.data,
            self._geom_model, self._geometry_data)
        return self._pin.computeCollisions(self._geom_model, self._geometry_data, False)

    @property
    def joint_limits_rad(self) -> list[tuple[float, float]]:
        return [(float(self._lb[i]), float(self._ub[i])) for i in range(self._nq)]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_target(self, x_m, y_m, z_m, rx_deg, ry_deg, rz_deg):
        """构建目标 SE3 矩阵."""
        q = self._euler_to_quat(rx_deg, ry_deg, rz_deg)
        T = self._pin.SE3(
            self._pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x_m, y_m, z_m]),
        )
        return T.homogeneous

    def _solve(self, T_target: np.ndarray,
               q0: Optional[list[float]] = None) -> Optional[list[float]]:
        """核心优化求解, 自动拒绝碰撞解, 多初始值重试."""
        if q0 is not None:
            q_init_candidates = [np.array(q0, dtype=float)]
        else:
            q_init_candidates = [self._q_init]

        # 如果当前初始值太远, 追加零位
        if self._q_history is not None and np.any(self._q_init):
            max_diff = float(np.max(np.abs(self._q_init - self._q_history)))
            if max_diff > 30.0 / 180.0 * math.pi:
                q_init_candidates.append(np.zeros(self._nq))

        # 追加几个备用初始值
        q_init_candidates.extend([
            np.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0]),
            np.array([-0.5, 0.3, -1.0, -0.5, 0.0, 0.5]),
            np.array([0.0, 0.8, -1.5, 0.0, 0.5, 0.0]),
        ])

        for init_q in q_init_candidates:
            self._opti.set_initial(self._var_q, init_q)
            self._opti.set_value(self._param_tf, T_target)
            try:
                sol = self._opti.solve_limited()
                sol_q = np.array(self._opti.value(self._var_q)).flatten()
            except Exception:
                continue

            # 碰撞检测
            if self.check_collision(sol_q.tolist()):
                continue  # 碰撞, 换下一个初始值

            self._q_history = sol_q.copy()
            self._q_init = sol_q.copy()
            return sol_q.tolist()

        return None  # 所有初始值都失败

    def _forward_kinematics(self, q: np.ndarray) -> Optional[np.ndarray]:
        """Pinocchio FK: joint angles (rad) → [x, y, z, rx, ry, rz] (m, rad)."""
        data = self._reduced.model.createData()
        self._pin.forwardKinematics(self._reduced.model, data, q)
        ee_id = self._reduced.model.getFrameId('ee')
        self._pin.updateFramePlacements(self._reduced.model, data)
        T = data.oMf[ee_id].homogeneous

        pos = T[:3, 3]
        R = T[:3, :3]
        rpy = self._rot_to_rpy(R)
        return np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]])

    @staticmethod
    def _rot_to_rpy(R: np.ndarray) -> np.ndarray:
        """旋转矩阵 → RPY (ZYX 外旋), 返回 (roll, pitch, yaw) in rad."""
        if R[2, 0] < -1 + 0.0001:
            pitch = math.pi / 2
            yaw = 0.0
            roll = math.atan2(R[0, 1], R[1, 1])
        elif R[2, 0] > 1 - 0.0001:
            pitch = -math.pi / 2
            yaw = 0.0
            roll = -math.atan2(R[0, 1], R[1, 1])
        else:
            pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
            yaw = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))
            roll = math.atan2(R[2, 1] / math.cos(pitch), R[2, 2] / math.cos(pitch))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _euler_to_quat(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
        """RPY (deg, ZYX 外旋) → 四元数 [x, y, z, w]."""
        roll, pitch, yaw = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([x, y, z, w])

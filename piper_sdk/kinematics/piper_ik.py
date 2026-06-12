#!/usr/bin/env python3
"""Piper 机械臂逆运动学 (数值解, 带阻尼最小二乘法).

使用正确的 DH 参数表 (dh_is_offset=0x01, 即 J2/J3 带 2°偏移).
"""

import math
import numpy as np
from typing import Optional, Literal

try:
    from .piper_fk import C_PiperForwardKinematics
except ImportError:
    from piper_fk import C_PiperForwardKinematics


class C_PiperInverseKinematics:
    """数值逆运动学求解器."""

    JOINT_LIMITS_RAD = [
        (-2.62, 2.62),
        (0.0, 3.14),
        (-2.97, 0.0),
        (-1.75, 1.75),
        (-1.22, 1.22),
        (-2.09, 2.09),
    ]

    def __init__(self):
        self._fk = C_PiperForwardKinematics(dh_is_offset=0x01)
        self._eps = 1e-6       # 数值微分扰动
        self._lambda = 0.5     # 阻尼因子
        self._max_iter = 200
        self._tol_pos = 1e-3   # 位置收敛阈值 (mm)
        self._tol_ori = 1e-2   # 姿态收敛阈值 (rad)
        self._damp_decay = 0.95  # 每次迭代阻尼衰减

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def solve_full(self, x_m: float, y_m: float, z_m: float,
                   rx_deg: float, ry_deg: float, rz_deg: float,
                   q0: Optional[list[float]] = None) -> list[float]:
        """6-DOF 逆运动学: 目标末端位姿 → 关节角度 (rad).

        Args:
            x_m, y_m, z_m: 目标位置 (m)
            rx_deg, ry_deg, rz_deg: 目标姿态 RPY (deg)
            q0: 初始关节角度 (rad), 默认 HOME 附近

        Returns:
            关节角度列表 (rad), 长度 6
        """
        target = np.array([x_m * 1000, y_m * 1000, z_m * 1000,
                           math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)])
        R_target = self._rpy_to_R(rx_deg, ry_deg, rz_deg)

        if q0 is None:
            q = np.array([0.0, 0.3, -0.5, 0.3, 0.0, 0.0])
        else:
            q = np.array(q0, dtype=float)

        lam = self._lambda

        for it in range(self._max_iter):
            cur = self._forward(q)
            pos_err = target[:3] - cur[:3]
            R_cur = self._rpy_to_R(*[math.degrees(v) for v in cur[3:]])
            ori_err = self._ori_error(R_cur, R_target)
            err = np.concatenate([pos_err, ori_err])

            if np.linalg.norm(pos_err) < self._tol_pos and np.linalg.norm(ori_err) < self._tol_ori:
                break

            J = self._jacobian(q)
            dq = self._dls(J, err, lam)
            q = q + dq
            q = self._clamp(q)
            lam *= self._damp_decay

        return q.tolist()

    def solve_position(self, x_m: float, y_m: float, z_m: float,
                       q0: Optional[list[float]] = None,
                       ref_rx_deg: float = 0.0, ref_ry_deg: float = 0.0, ref_rz_deg: float = 0.0,
                       ) -> list[float]:
        """3-DOF 位置逆运动学: 目标位置 → 关节角度, 保持当前/参考姿态.

        Args:
            x_m, y_m, z_m: 目标位置 (m)
            q0: 初始关节角度 (rad), 默认 HOME 附近
            ref_rx_deg, ref_ry_deg, ref_rz_deg: 参考姿态 (仅 position-only 时 q0=None 使用)

        Returns:
            关节角度列表 (rad)
        """
        target = np.array([x_m * 1000, y_m * 1000, z_m * 1000])

        if q0 is None:
            q = np.array([0.0, 0.3, -0.5, 0.3, 0.0, 0.0])
            # 先做一次 full IK 确定参考姿态
            return self.solve_full(x_m, y_m, z_m, ref_rx_deg, ref_ry_deg, ref_rz_deg)
        else:
            q = np.array(q0, dtype=float)

        # 以当前姿态为参考, 只优化位置
        _, _, _, ref_rx, ref_ry, ref_rz = self._forward(q)
        R_ref = self._rpy_to_R(math.degrees(ref_rx), math.degrees(ref_ry), math.degrees(ref_rz))

        lam = self._lambda

        for it in range(self._max_iter):
            cur = self._forward(q)
            pos_err = target - cur[:3]

            if np.linalg.norm(pos_err) < self._tol_pos:
                break

            J_full = self._jacobian(q)
            J = J_full[:3, :]  # 只取位置部分

            # 加入空空间项: 尽量保持当前姿态
            # 姿态误差作为次要目标, 投影到零空间
            R_cur = self._rpy_to_R(*[math.degrees(v) for v in cur[3:]])
            ori_err = self._ori_error(R_cur, R_ref)
            ori_grad = J_full[3:, :].T @ ori_err  # 姿态梯度
            N = np.eye(6) - np.linalg.pinv(J) @ J  # 零空间投影
            nullspace_term = N @ ori_grad * 0.01

            dq = self._dls(J, pos_err, lam) + nullspace_term
            q = q + dq
            q = self._clamp(q)
            lam *= self._damp_decay

        return q.tolist()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _forward(self, q: np.ndarray) -> np.ndarray:
        """FK: joint angles (rad) → [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]."""
        result = self._fk.CalFK(q.tolist())
        ee = result[5]
        return np.array([ee[0], ee[1], ee[2],
                         math.radians(ee[3]), math.radians(ee[4]), math.radians(ee[5])])

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        """数值 Jacobian (6x6)."""
        f0 = self._forward(q)
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = self._eps
            fi = self._forward(q + dq)
            J[:3, i] = (fi[:3] - f0[:3]) / self._eps
            J[3:, i] = self._ori_error_from_rad(f0[3:], fi[3:]) / self._eps
        return J

    def _dls(self, J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray:
        """阻尼最小二乘: dq = J^T (J J^T + λ^2 I)^{-1} err."""
        m = J.shape[0]
        JJT = J @ J.T
        A = JJT + lam * lam * np.eye(m)
        try:
            return J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            return J.T @ (err / (lam * lam))

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        """关节限位裁剪."""
        qc = q.copy()
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS_RAD):
            qc[i] = max(lo, min(hi, qc[i]))
        return qc

    # ------------------------------------------------------------------
    # 旋转矩阵 & 姿态误差
    # ------------------------------------------------------------------

    @staticmethod
    def _rpy_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
        """RPY (ZYX 外旋) → 3x3 旋转矩阵: R = Rz(rz) Ry(ry) Rx(rx)."""
        rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        R = np.zeros((3, 3))
        R[0, 0] = cz * cy
        R[0, 1] = cz * sy * sx - sz * cx
        R[0, 2] = cz * sy * cx + sz * sx
        R[1, 0] = sz * cy
        R[1, 1] = sz * sy * sx + cz * cx
        R[1, 2] = sz * sy * cx - cz * sx
        R[2, 0] = -sy
        R[2, 1] = cy * sx
        R[2, 2] = cy * cx
        return R

    @staticmethod
    def _ori_error(R_cur: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        """旋转误差向量: 0.5 * skew^{-1}(R_cur^T R_target - R_target^T R_cur)."""
        dR = R_cur.T @ R_target - R_target.T @ R_cur
        return 0.5 * np.array([dR[2, 1], dR[0, 2], dR[1, 0]])

    @staticmethod
    def _ori_error_from_rad(cur_rpy_rad: np.ndarray, new_rpy_rad: np.ndarray) -> np.ndarray:
        """从两组 RPY (rad) 之间的差分近似姿态误差向量."""
        R_cur = C_PiperInverseKinematics._rpy_to_R(
            math.degrees(cur_rpy_rad[0]), math.degrees(cur_rpy_rad[1]), math.degrees(cur_rpy_rad[2]))
        R_new = C_PiperInverseKinematics._rpy_to_R(
            math.degrees(new_rpy_rad[0]), math.degrees(new_rpy_rad[1]), math.degrees(new_rpy_rad[2]))
        return C_PiperInverseKinematics._ori_error(R_cur, R_new)


# ------------------------------------------------------------------
# 便捷函数
# ------------------------------------------------------------------

def ik_full(x_m: float, y_m: float, z_m: float,
            rx_deg: float, ry_deg: float, rz_deg: float,
            q0: Optional[list[float]] = None) -> list[float]:
    """6-DOF IK 便捷函数."""
    return C_PiperInverseKinematics().solve_full(x_m, y_m, z_m, rx_deg, ry_deg, rz_deg, q0)


def ik_position(x_m: float, y_m: float, z_m: float,
                q0: Optional[list[float]] = None) -> list[float]:
    """3-DOF 位置 IK 便捷函数."""
    return C_PiperInverseKinematics().solve_position(x_m, y_m, z_m, q0)

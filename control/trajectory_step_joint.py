#!/usr/bin/env python3
"""轨迹步进控制 (Pinocchio IK) —— 从 .npz 加载路径点, IK 解算关节角, 空格键步进。
启动后自动 HOME → 第一个目标点, 然后每按一次空格走一步。"""

import time
import sys
import os
import math
import argparse
import tty
import termios
import numpy as np

_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from piper_sdk import C_PiperInterface_V2
from piper_sdk.kinematics.piper_ik_pinocchio import C_PiperIKPinocchio

RAD2JOINT = 57295.7795
M2POSE = 1_000_000
DEG2POSE = 1000

JOINT_LIMITS = [
    (-2.62, 2.62), (0.0, 3.14), (-2.97, 0.0),
    (-1.75, 1.75), (-1.22, 1.22), (-2.09, 2.09),
]


class TrajectoryStepper:
    """轨迹步进控制器 (Pinocchio IK + 关节角控制)。"""

    def __init__(self, can_name: str = "can0"):
        self.can_name = can_name
        self.piper = None
        self.ik = None
        self._poses = None
        self._last_joints_raw = None

    def connect(self):
        self.piper = C_PiperInterface_V2(self.can_name)
        self.piper.ConnectPort()
        time.sleep(0.3)
        print(f"[信息] CAN {self.can_name} 已连接")

    def enable(self):
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print("[信息] 电机已使能")

    def disconnect(self):
        if self.piper:
            self.piper.DisconnectPort()
            print("[信息] 已断开连接")

    def load_trajectory(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self._poses = data["positions_base"]
        print(f"[加载] {filepath}  ({self._poses.shape[0]} 个轨迹点)")

    # ------- FK / 状态读取 -------

    def _init_ik(self):
        if self.ik is None:
            self.ik = C_PiperIKPinocchio()

    def _read_joints_deg(self):
        j = self.piper.GetArmJointMsgs().joint_state
        return [j.joint_1 / 1000.0, j.joint_2 / 1000.0, j.joint_3 / 1000.0,
                j.joint_4 / 1000.0, j.joint_5 / 1000.0, j.joint_6 / 1000.0]

    def _read_pose(self):
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return (ep.X_axis / M2POSE, ep.Y_axis / M2POSE, ep.Z_axis / M2POSE,
                ep.RX_axis / DEG2POSE, ep.RY_axis / DEG2POSE, ep.RZ_axis / DEG2POSE)

    def _read_status(self):
        return self.piper.GetArmStatus().arm_status

    def print_state(self):
        joints = self._read_joints_deg()
        x, y, z, rx, ry, rz = self._read_pose()
        st = self._read_status()
        sys.stdout.write(
            f"\r\n"
            f"末端: ({x:.4f}, {y:.4f}, {z:.4f})m  "
            f"RX={rx:.1f}deg RY={ry:.1f}deg RZ={rz:.1f}deg\n"
            f"关节: [{', '.join(f'{v:.1f}deg' for v in joints)}]\n"
            f"状态: arm={st.arm_status.name}({st.arm_status.value})\n"
        )
        sys.stdout.flush()

    # ------- 运动 -------

    def go_home(self, speed: int = 30):
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                                move_spd_rate_ctrl=speed, is_mit_mode=0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)

    def _ik_to_target(self, x_m, y_m, z_m):
        """IK 求解 → 关节角 (deg) + 原始编码器值."""
        self._init_ik()
        q0 = [math.radians(d) for d in self._read_joints_deg()]
        sol = self.ik.solve_position(x_m, y_m, z_m, q0=q0)
        if sol is None:
            return None, None
        joints_deg = [math.degrees(r) for r in sol]
        joints_raw = [int(max(lim[0], min(lim[1], r)) * RAD2JOINT)
                      for r, lim in zip(sol, JOINT_LIMITS)]
        return joints_deg, joints_raw

    def move_joints(self, joints_raw: list[int], joints_deg: list[float], speed: int = 30):
        """MOVEJ 下发关节角."""
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                                move_spd_rate_ctrl=speed, is_mit_mode=0x00)
        self.piper.JointCtrl(*joints_raw)
        self._last_joints_raw = joints_raw
        self._last_joints_deg = joints_deg

    def wait_arrival(self, target_deg: list[float], timeout: float = 60.0) -> bool:
        """等待关节角到达目标, 误差 < 0.5°. """
        t0 = time.time()
        last_print = 0.0
        time.sleep(0.3)  # 等命令处理
        while time.time() - t0 < timeout:
            actual = self._read_joints_deg()
            errs = [abs(actual[i] - target_deg[i]) for i in range(6)]
            if max(errs) < 0.5:
                return True
            now = time.time()
            st = self._read_status()
            if st.arm_status != 0x00:
                sys.stdout.write(f"\r\033[K  [错误] 臂状态: {st.arm_status.name}\n")
                sys.stdout.flush()
                return False
            if now - last_print > 1.0:
                sys.stdout.write(
                    f"\r\033[K  ... 运动中 max_err={max(errs):.1f}deg  "
                    f"关节: [{actual[0]:.1f} {actual[1]:.1f} {actual[2]:.1f} "
                    f"{actual[3]:.1f} {actual[4]:.1f} {actual[5]:.1f}]deg\n"
                )
                sys.stdout.flush()
                last_print = now
            time.sleep(0.1)
        sys.stdout.write(f"\r\033[K  [警告] 到达超时 ({timeout}s)\n")
        sys.stdout.flush()
        return False

    def resend_cmd(self):
        """重发上次关节角, 防止卸力."""
        if self._last_joints_raw is not None:
            self.piper.JointCtrl(*self._last_joints_raw)


# ════════════════════ 主程序 ════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="轨迹步进控制 (Pinocchio IK) —— 自动 HOME → 第一个点, 空格键步进",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
按键:
  空格 / n    下一个点
  p           上一个点
  j           跳转到指定序号
  s           打印当前状态
  r           复位
  q / Ctrl+C  退出 (保持使能)

示例:
  %(prog)s trajectory_base_20260610_070851.npz
  %(prog)s trajectory.npz --can can0 --speed 20
        """,
    )
    parser.add_argument("trajectory", help="轨迹 .npz 文件路径")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--speed", type=int, default=30, help="运动速度 0-100")
    parser.add_argument("--timeout", type=float, default=60.0, help="单点到达超时 (秒)")

    args = parser.parse_args()

    ctrl = TrajectoryStepper(args.can)
    ctrl.load_trajectory(args.trajectory)

    idx = 0
    total = ctrl._poses.shape[0]
    raw_mode = False
    fd = None

    try:
        ctrl.connect()
        ctrl.enable()

        # ---- 阶段 1: HOME ----
        print("\n[阶段 1/2] 回到 HOME ...")
        ctrl.go_home(args.speed)
        if not ctrl.wait_arrival([0.0] * 6, args.timeout):
            print("[警告] HOME 未到达, 继续")
        ctrl.print_state()

        # ---- 阶段 2: 第一个轨迹点 ----
        print(f"\n[阶段 2/2] 移动到第一个轨迹点 ...")
        p = ctrl._poses[0]
        print(f"  目标位置: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m")
        j_deg, j_raw = ctrl._ik_to_target(p[0], p[1], p[2])
        if j_deg is None:
            print("[错误] IK 无解, 退出")
            return
        print(f"  IK 关节: [{', '.join(f'{d:.1f}deg' for d in j_deg)}]")
        ctrl.move_joints(j_raw, j_deg, args.speed)
        if not ctrl.wait_arrival(j_deg, args.timeout):
            print("[警告] 第一个点未到达")
        idx = 1

        # ---- 阶段 3: 按键步进 ----
        ctrl.print_state()
        if idx < total:
            nxt = ctrl._poses[idx]
            sys.stdout.write(
                f"\n=== 按键步进 =====================\n"
                f"下一个 [{idx}/{total-1}]: ({nxt[0]:.4f}, {nxt[1]:.4f}, {nxt[2]:.4f})m\n"
                f" 空格/n: 下一个  |  p: 上一个  |  j: 跳转  |  s: 状态  |  q: 退出\n"
            )
        else:
            sys.stdout.write("\n[完成] 已到达最后一个轨迹点 (q 退出)\n")
        sys.stdout.flush()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        raw_mode = True

        while True:
            ch = sys.stdin.read(1)

            if ch in ('\x03', 'q'):
                sys.stdout.write("\r\n\033[K[退出] (保持使能)\r\n")
                break

            elif ch in (' ', 'n'):
                if idx >= total:
                    sys.stdout.write(f"\r\n\033[K已是最后一个点\r\n")
                    sys.stdout.flush()
                    continue
                p = ctrl._poses[idx]
                sys.stdout.write(f"\r\n\033[K→ [{idx}/{total-1}] "
                                 f"目标: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m\n")
                sys.stdout.flush()
                j_deg, j_raw = ctrl._ik_to_target(p[0], p[1], p[2])
                if j_deg is None:
                    sys.stdout.write(f"\r\033[K  [错误] IK 无解\n")
                    sys.stdout.flush()
                    continue
                sys.stdout.write(f"\r\033[K  IK: [{', '.join(f'{d:.1f}deg' for d in j_deg)}]\n")
                sys.stdout.flush()
                ctrl.move_joints(j_raw, j_deg, args.speed)
                ok = ctrl.wait_arrival(j_deg, args.timeout)
                sys.stdout.write(f"  结果: {'到达' if ok else '失败/超时'}\n")
                sys.stdout.flush()
                ctrl.print_state()
                if ok:
                    idx += 1
                if idx < total:
                    nxt = ctrl._poses[idx]
                    sys.stdout.write(f"\r\033[K下一个 [{idx}/{total-1}]: "
                                     f"({nxt[0]:.4f}, {nxt[1]:.4f}, {nxt[2]:.4f})m\n")
                else:
                    sys.stdout.write(f"\r\033[K[完成] 全部轨迹点已走完\n")
                sys.stdout.flush()

            elif ch == 'p':
                if idx <= 1:
                    sys.stdout.write(f"\r\n\033[K已在第一个点, 无法回退\r\n")
                    sys.stdout.flush()
                    continue
                idx -= 2
                p = ctrl._poses[idx]
                sys.stdout.write(f"\r\n\033[K← [{idx}/{total-1}] "
                                 f"目标: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m\n")
                sys.stdout.flush()
                j_deg, j_raw = ctrl._ik_to_target(p[0], p[1], p[2])
                if j_deg is None:
                    sys.stdout.write(f"\r\033[K  [错误] IK 无解\n")
                    sys.stdout.flush()
                    continue
                ctrl.move_joints(j_raw, j_deg, args.speed)
                ctrl.wait_arrival(j_deg, args.timeout)
                idx += 1
                ctrl.print_state()
                sys.stdout.write(f"\r\033[K下一个 [{idx}/{total-1}]: "
                                 f"({ctrl._poses[idx][0]:.4f}, {ctrl._poses[idx][1]:.4f}, {ctrl._poses[idx][2]:.4f})m\n")
                sys.stdout.flush()

            elif ch == 'j':
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                try:
                    num_str = input(f"\r\n\033[K跳转到序号 (0~{total-1}): ")
                    target_idx = int(num_str.strip())
                    target_idx = max(0, min(target_idx, total - 1))
                    idx = target_idx
                    p = ctrl._poses[idx]
                    print(f"→ [{idx}/{total-1}] ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m")
                    j_deg, j_raw = ctrl._ik_to_target(p[0], p[1], p[2])
                    if j_deg is None:
                        print("  [错误] IK 无解")
                    else:
                        print(f"  IK: [{', '.join(f'{d:.1f}deg' for d in j_deg)}]")
                        ctrl.move_joints(j_raw, j_deg, args.speed)
                        ctrl.wait_arrival(j_deg, args.timeout)
                        idx += 1
                except (ValueError, EOFError):
                    print("无效序号")
                finally:
                    tty.setraw(fd)

            elif ch == 's':
                ctrl.print_state()

            elif ch == 'r':
                sys.stdout.write("\r\n\033[K[复位]\n")
                sys.stdout.flush()
                ctrl.piper.MotionCtrl_1(0x02, 0, 0)
                time.sleep(0.5)
                ctrl.enable()
                # 复位后重新使能, 发送 HOME 指令防止卸力
                ctrl.go_home(args.speed)

            # 空闲时定期重发防止卸力
            ctrl.resend_cmd()

    except (KeyboardInterrupt, EOFError):
        sys.stdout.write("\r\n[退出] (保持使能)\r\n")
    except Exception as e:
        sys.stdout.write(f"\r\n[错误] {e}\r\n")
    finally:
        if raw_mode and fd is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if ctrl.piper:
            ctrl.disconnect()


if __name__ == "__main__":
    main()

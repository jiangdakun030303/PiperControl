#!/usr/bin/env python3
"""轨迹步进控制 —— 从 .npz 加载路径点，按空格键逐一移动。
启动后自动 HOME → 第一个目标点，然后每按一次空格走一步。"""

import time
import sys
import os
import argparse
import tty
import termios
import numpy as np

_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from piper_sdk import C_PiperInterface_V2


class TrajectoryStepper:
    """轨迹步进控制器。"""

    M2POSE = 1_000_000
    DEG2POSE = 1000
    RAD2JOINT = 57295.7795

    JOINT_LIMITS = [
        (-2.62, 2.62), (0.0, 3.14), (-2.97, 0.0),
        (-1.75, 1.75), (-1.22, 1.22), (-2.09, 2.09),
    ]

    def __init__(self, can_name: str = "can0"):
        self.can_name = can_name
        self.piper = None
        self._poses = None
        self._last_cmd = None

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

    def _read_pose(self):
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return (ep.X_axis / self.M2POSE, ep.Y_axis / self.M2POSE, ep.Z_axis / self.M2POSE,
                ep.RX_axis / self.DEG2POSE, ep.RY_axis / self.DEG2POSE, ep.RZ_axis / self.DEG2POSE)

    def _read_joints_deg(self):
        j = self.piper.GetArmJointMsgs().joint_state
        return [v / 1000.0 for v in
                [j.joint_1, j.joint_2, j.joint_3,
                 j.joint_4, j.joint_5, j.joint_6]]

    def _read_status(self):
        return self.piper.GetArmStatus().arm_status

    def motion_status(self):
        return self._read_status().motion_status

    def arm_status(self):
        return self._read_status().arm_status

    def print_state(self, prefix: str = ""):
        """打印当前机械臂状态。"""
        x, y, z, rx, ry, rz = self._read_pose()
        st = self._read_status()
        joints = self._read_joints_deg()
        sys.stdout.write(
            f"\r\n{prefix}"
            f"末端: ({x:.4f}, {y:.4f}, {z:.4f})m  "
            f"RX={rx:.1f}° RY={ry:.1f}° RZ={rz:.1f}°\n"
            f"关节: [{', '.join(f'{v:.1f}°' for v in joints)}]\n"
            f"状态: motion={st.motion_status.name}({st.motion_status.value}) "
            f"arm={st.arm_status.name}({st.arm_status.value})\n"
        )
        sys.stdout.flush()

    def load_trajectory(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self._poses = data["positions_base"]
        print(f"[加载] {filepath}  ({self._poses.shape[0]} 个轨迹点)")

    # ------- 运动 -------

    def go_home(self, speed: int = 30):
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                                move_spd_rate_ctrl=speed, is_mit_mode=0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)

    def move_to_point(self, x_m: float, y_m: float, z_m: float, speed: int = 30):
        """MOVEP 到目标位置，保持当前姿态。"""
        *_, rx, ry, rz = self._read_pose()
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00,
                                move_spd_rate_ctrl=speed, is_mit_mode=0x00)
        cmd = (int(x_m * self.M2POSE), int(y_m * self.M2POSE), int(z_m * self.M2POSE),
               int(rx * self.DEG2POSE), int(ry * self.DEG2POSE), int(rz * self.DEG2POSE))
        self.piper.EndPoseCtrl(*cmd)
        self._last_cmd = cmd  # 保存指令用于重发

    # ------- 到达等待 -------

    def wait_arrival(self, timeout: float = 60.0) -> bool:
        t0 = time.time()
        last_print = 0.0
        last_resend = 0.0
        while time.time() - t0 < timeout:
            ms = self.motion_status()
            arm_st = self.arm_status()
            if ms == 0x00:
                return True
            now = time.time()
            if now - last_print > 1.0:
                x, y, z, _, _, _ = self._read_pose()
                sys.stdout.write(f"\r\033[K  ... 运动中 motion={ms.name}({ms.value})  "
                                 f"当前: ({x:.4f}, {y:.4f}, {z:.4f})m\n")
                sys.stdout.flush()
                last_print = now
            # 每隔 0.5s 重发 EndPoseCtrl，确保指令被接收
            if now - last_resend > 0.5 and self._last_cmd is not None:
                self.piper.EndPoseCtrl(*self._last_cmd)
                last_resend = now
            if arm_st in (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07):
                sys.stdout.write(f"\r\033[K  [错误] 臂状态: {arm_st.name}\n")
                sys.stdout.flush()
                return False
            time.sleep(0.1)
        sys.stdout.write(f"\r\033[K  [警告] 到达超时 ({timeout}s)\n")
        sys.stdout.flush()
        return False


# ════════════════════ 主程序 ════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="轨迹步进控制 —— 自动 HOME → 第一个点, 空格键步进",
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
        if not ctrl.wait_arrival(args.timeout):
            print("[警告] HOME 未到达, 继续")
        x, y, z, rx, ry, rz = ctrl._read_pose()
        print(f"HOME 到达: ({x:.4f}, {y:.4f}, {z:.4f})m  "
              f"姿态: RX={rx:.1f}° RY={ry:.1f}° RZ={rz:.1f}°")

        # ---- 阶段 2: 第一个目标点 ----
        print(f"\n[阶段 2/2] 移动到第一个轨迹点 ...")
        p = ctrl._poses[0]
        print(f"  目标: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m")
        ctrl.move_to_point(p[0], p[1], p[2], args.speed)
        if not ctrl.wait_arrival(args.timeout):
            print("[警告] 第一个点未到达")
        idx = 1  # 已到达第 0 个点, 下一个是第 1 个

        # ---- 阶段 3: 按键步进 ----
        x, y, z, rx, ry, rz = ctrl._read_pose()
        if idx < total:
            nxt = ctrl._poses[idx]
            sys.stdout.write(
                f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"当前位置: ({x:.4f}, {y:.4f}, {z:.4f})m\n"
                f"下一个 [{idx}/{total-1}]: ({nxt[0]:.4f}, {nxt[1]:.4f}, {nxt[2]:.4f})m\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
                ctrl.move_to_point(p[0], p[1], p[2], args.speed)
                ok = ctrl.wait_arrival(args.timeout)
                # 打印指令参数和机械臂状态
                cmd = ctrl._last_cmd
                if cmd:
                    sys.stdout.write(
                        f"  指令: EndPoseCtrl(X={cmd[0]}, Y={cmd[1]}, Z={cmd[2]}, "
                        f"RX={cmd[3]}, RY={cmd[4]}, RZ={cmd[5]})\n"
                    )
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
                idx -= 2  # 回到上一个已到达的点
                p = ctrl._poses[idx]
                sys.stdout.write(f"\r\n\033[K← [{idx}/{total-1}] "
                                 f"目标: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m\n")
                sys.stdout.flush()
                ctrl.move_to_point(p[0], p[1], p[2], args.speed)
                ctrl.wait_arrival(args.timeout)
                idx += 1
                cmd = ctrl._last_cmd
                if cmd:
                    sys.stdout.write(
                        f"  指令: EndPoseCtrl(X={cmd[0]}, Y={cmd[1]}, Z={cmd[2]}, "
                        f"RX={cmd[3]}, RY={cmd[4]}, RZ={cmd[5]})\n"
                    )
                sys.stdout.flush()
                ctrl.print_state()
                sys.stdout.write(f"\r\033[K下一个 [{idx}/{total-1}]: "
                                 f"({ctrl._poses[idx][0]:.4f}, {ctrl._poses[idx][1]:.4f}, {ctrl._poses[idx][2]:.4f})m\n")
                sys.stdout.flush()

            elif ch == 'j':
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                try:
                    num_str = input("\r\n\033[K跳转到序号 (0~{0}): ".format(total - 1))
                    target = int(num_str.strip())
                    target = max(0, min(target, total - 1))
                    idx = target
                    p = ctrl._poses[idx]
                    print(f"→ [{idx}/{total-1}] ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})m")
                    ctrl.move_to_point(p[0], p[1], p[2], args.speed)
                    ctrl.wait_arrival(args.timeout)
                    idx += 1
                    cmd = ctrl._last_cmd
                    if cmd:
                        print(f"  指令: EndPoseCtrl(X={cmd[0]}, Y={cmd[1]}, Z={cmd[2]}, "
                              f"RX={cmd[3]}, RY={cmd[4]}, RZ={cmd[5]})")
                    x, y, z, _, _, _ = ctrl._read_pose()
                    print(f"  当前: ({x:.4f}, {y:.4f}, {z:.4f})m")
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

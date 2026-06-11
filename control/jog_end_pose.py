#!/usr/bin/env python3
"""末端位姿微调脚本 —— 无回车按键步进 XYZ + RX/RY/RZ。"""

import time
import sys
import tty
import termios
import argparse
from piper_sdk import C_PiperInterface_V2


class EndJogController:
    MM_STEP = 1000       # mm / deg → 0.001 units
    DEG_STEP = 1000      # deg → 0.001°

    def __init__(self, can_name: str = "can0", step_mm: float = 5.0, rot_step_deg: float = 1.0):
        self.can_name = can_name
        self.pos_step = int(step_mm * self.MM_STEP)
        self.rot_step = int(rot_step_deg * self.DEG_STEP)
        self.piper = None

    def connect(self):
        try:
            self.piper = C_PiperInterface_V2(self.can_name)
        except Exception as e:
            raise ConnectionError(f"创建接口失败: {e}")
        self.piper.ConnectPort()
        time.sleep(0.3)
        print(f"[信息] CAN {self.can_name} 已连接")

    def enable(self):
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print("[信息] 电机已使能")
        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)

    def disconnect(self):
        if self.piper:
            self.piper.DisconnectPort()

    def _read_pose(self) -> list:
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return [ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis]

    def _send_pose(self, pose: list):
        self.piper.EndPoseCtrl(*pose)

    def jog_pos(self, axis: int, delta_mm: float):
        """axis: 0=X, 1=Y, 2=Z"""
        step = int(delta_mm * self.MM_STEP)
        pose = self._read_pose()
        pose[axis] += step
        self._send_pose(pose)

    def jog_rot(self, axis: int, delta_deg: float):
        """axis: 3=RX, 4=RY, 5=RZ"""
        step = int(delta_deg * self.DEG_STEP)
        pose = self._read_pose()
        pose[axis] += step
        self._send_pose(pose)

    def print_pose(self):
        pose = self._read_pose()
        mm = [f"{pose[i] / self.MM_STEP:.1f}mm" for i in range(3)]
        deg = [f"{pose[i] / self.DEG_STEP:.2f}°" for i in range(3, 6)]
        sys.stdout.write(
            f"\r\033[K末端: X={mm[0]}  Y={mm[1]}  Z={mm[2]}  "
            f"RX={deg[0]}  RY={deg[1]}  RZ={deg[2]}"
        )
        sys.stdout.flush()

    def reset(self):
        if self.piper:
            self.piper.MotionCtrl_1(0x02, 0, 0)


def main():
    parser = argparse.ArgumentParser(description="末端位姿微调控制")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--step", type=float, default=5.0, help="位置步进量 (mm), 默认 5")
    parser.add_argument("--rot-step", type=float, default=1.0, help="姿态步进量 (°), 默认 1")
    args = parser.parse_args()

    ctrl = EndJogController(args.can, args.step, args.rot_step)
    raw_mode = False
    fd = None

    try:
        ctrl.connect()
        ctrl.enable()
        print(f"\n位置: ±{args.step:.0f}mm  |  姿态: ±{args.rot_step:.0f}°  |  速度: 30%")
        print("位置: w/s Z | a/d Y | q/e X")
        print("姿态: u/j RX | i/k RY | o/l RZ")
        print("其他: p 位姿 | r 复位 | Ctrl+C 退出\n")
        ctrl.print_pose()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        raw_mode = True

        while True:
            ch = sys.stdin.read(1)
            if ch == '\x03':                       # Ctrl+C
                sys.stdout.write("\r\n[退出] (保持使能)\r\n")
                break
            # ---- 位置 ----
            elif ch == 'w':
                ctrl.jog_pos(2, +args.step)
                sys.stdout.write(f"\r\033[K  Z +{args.step:.0f}mm\r\n")
            elif ch == 's':
                ctrl.jog_pos(2, -args.step)
                sys.stdout.write(f"\r\033[K  Z -{args.step:.0f}mm\r\n")
            elif ch == 'a':
                ctrl.jog_pos(1, -args.step)
                sys.stdout.write(f"\r\033[K  Y -{args.step:.0f}mm\r\n")
            elif ch == 'd':
                ctrl.jog_pos(1, +args.step)
                sys.stdout.write(f"\r\033[K  Y +{args.step:.0f}mm\r\n")
            elif ch == 'q':
                ctrl.jog_pos(0, +args.step)
                sys.stdout.write(f"\r\033[K  X +{args.step:.0f}mm\r\n")
            elif ch == 'e':
                ctrl.jog_pos(0, -args.step)
                sys.stdout.write(f"\r\033[K  X -{args.step:.0f}mm\r\n")
            # ---- 姿态 ----
            elif ch == 'u':
                ctrl.jog_rot(3, +args.rot_step)
                sys.stdout.write(f"\r\033[K  RX +{args.rot_step:.0f}°\r\n")
            elif ch == 'j':
                ctrl.jog_rot(3, -args.rot_step)
                sys.stdout.write(f"\r\033[K  RX -{args.rot_step:.0f}°\r\n")
            elif ch == 'i':
                ctrl.jog_rot(4, +args.rot_step)
                sys.stdout.write(f"\r\033[K  RY +{args.rot_step:.0f}°\r\n")
            elif ch == 'k':
                ctrl.jog_rot(4, -args.rot_step)
                sys.stdout.write(f"\r\033[K  RY -{args.rot_step:.0f}°\r\n")
            elif ch == 'o':
                ctrl.jog_rot(5, +args.rot_step)
                sys.stdout.write(f"\r\033[K  RZ +{args.rot_step:.0f}°\r\n")
            elif ch == 'l':
                ctrl.jog_rot(5, -args.rot_step)
                sys.stdout.write(f"\r\033[K  RZ -{args.rot_step:.0f}°\r\n")
            # ---- 其他 ----
            elif ch == 'p':
                ctrl.print_pose()
                sys.stdout.write("\r\n")
            elif ch == 'r':
                ctrl.reset()
                time.sleep(0.5)
                ctrl.enable()
                sys.stdout.write("\r\033[K[复位]\r\n")

            ctrl.print_pose()

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

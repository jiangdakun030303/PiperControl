#!/usr/bin/env python3
"""
Piper 机械臂交互控制脚本 —— 绝对移动 / 步进微调分开。

主菜单:
  1. 末端绝对控制 (MOVEP / MOVEL)
  2. 末端步进控制
  3. 夹爪控制
  4. 关节绝对控制 (MOVEJ)
  5. 关节步进控制
  s. 状态  h. 回零  e. 急停  r. 复位
  v. 速度  d. 失能  q. 退出 (不掉落)
"""

import time
import sys
import re
import argparse
import numpy as np
from piper_sdk import C_PiperInterface_V2


# ---------- constants ----------

RAD2JOINT = 57295.7795
MM2POSE = 1000
MM2GRIPPER = 1000
DEG2POSE = 1000

JOINT_LIMITS_RAD = [
    (-2.62, 2.62),
    (0.0, 3.14),
    (-2.97, 0.0),
    (-1.75, 1.75),
    (-1.22, 1.22),
    (-2.09, 2.09),
]

JOINT_LIMITS_DEG = [
    (-150, 150),
    (0, 180),
    (-170, 0),
    (-100, 100),
    (-70, 70),
    (-120, 120),
]


# ---------- controller wrapper ----------

class Arm:
    def __init__(self, can_name="can0"):
        self.can = can_name
        self._p = None
        self._speed = 50
        self.step_mm = 10.0
        self.step_deg = 5.0

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, v):
        self._speed = max(1, min(100, int(v)))

    # -- connect / disconnect --

    def connect(self):
        self._p = C_PiperInterface_V2(self.can)
        self._p.ConnectPort()
        time.sleep(0.3)
        if not self._p.isOk():
            print(f"[ERR] CAN {self.can} 连接失败")
            return False
        print(f"[OK] CAN {self.can} 已连接")

        t0 = time.time()
        while not self._p.EnablePiper():
            if time.time() - t0 > 10:
                print("[ERR] 使能超时")
                return False
            time.sleep(0.01)
        print("[OK] 电机已使能")
        return True

    def disconnect(self):
        if self._p:
            self._p.DisconnectPort()

    def disable(self):
        if self._p:
            self._p.DisablePiper()
            print("[!!] 电机已失能 (机械臂掉落)")

    # -- mode --

    def _mode(self, move_mode):
        self._p.MotionCtrl_2(
            ctrl_mode=0x01, move_mode=move_mode,
            move_spd_rate_ctrl=self._speed, is_mit_mode=0x00,
        )

    # -- read current state (user units: mm / deg) --

    def read_end_pose(self):
        p = self._p.GetArmEndPoseMsgs().end_pose
        return [p.X_axis / MM2POSE, p.Y_axis / MM2POSE, p.Z_axis / MM2POSE,
                p.RX_axis / DEG2POSE, p.RY_axis / DEG2POSE, p.RZ_axis / DEG2POSE]

    def read_joints_deg(self):
        j = self._p.GetArmJointMsgs().joint_state
        return [j.joint_1 / 1000, j.joint_2 / 1000, j.joint_3 / 1000,
                j.joint_4 / 1000, j.joint_5 / 1000, j.joint_6 / 1000]

    # -- absolute moves --

    def end_pose(self, x_mm, y_mm, z_mm, rx_deg=0, ry_deg=0, rz_deg=0, linear=False):
        self._mode(0x02 if linear else 0x00)
        self._p.EndPoseCtrl(
            int(x_mm * MM2POSE), int(y_mm * MM2POSE), int(z_mm * MM2POSE),
            int(rx_deg * DEG2POSE), int(ry_deg * DEG2POSE), int(rz_deg * DEG2POSE),
        )

    def joints_deg(self, joints_deg):
        self._mode(0x01)
        cmd = [int(np.clip(np.deg2rad(d), *lim) * RAD2JOINT)
               for d, lim in zip(joints_deg, JOINT_LIMITS_RAD)]
        self._p.JointCtrl(*cmd)

    def home(self):
        print("[→] 回零...")
        self.joints_deg([0, 0, 0, 0, 0, 0])

    def gripper(self, width_mm, effort_nm=1.0):
        self._p.GripperCtrl(
            int(width_mm * MM2GRIPPER),
            int(effort_nm * 1000),
            gripper_code=0x01, set_zero=0x00,
        )

    # -- jog moves --

    def jog_end_pose(self, axis, delta):
        cur = self.read_end_pose()
        idx = {'x': 0, 'y': 1, 'z': 2, 'rx': 3, 'ry': 4, 'rz': 5}[axis]
        cur[idx] += delta
        self.end_pose(*cur, linear=False)

    def jog_joint(self, joint_idx, delta_deg):
        cur = self.read_joints_deg()
        cur[joint_idx - 1] += delta_deg
        self.joints_deg(cur)

    # -- estop / reset --

    def estop(self):
        self._p.MotionCtrl_1(0x01, 0, 0)
        print("[!!] 急停!")

    def reset(self):
        self._p.MotionCtrl_1(0x02, 0, 0)
        print("[OK] 已复位")
        time.sleep(0.3)
        t0 = time.time()
        while not self._p.EnablePiper():
            if time.time() - t0 > 10:
                print("[ERR] 重新使能超时")
                return
            time.sleep(0.01)
        print("[OK] 已重新使能")

    # -- status display --

    def status(self):
        try:
            j = self._p.GetArmJointMsgs().joint_state
            p = self._p.GetArmEndPoseMsgs().end_pose
            g = self._p.GetArmGripperMsgs().gripper_state
            s = self._p.GetArmStatus().arm_status
        except Exception as e:
            print(f"[ERR] 读取状态失败: {e}")
            return

        mode_names = {0: "MOVEP", 1: "MOVEJ", 2: "MOVEL", 3: "MOVEC"}

        print()
        print("┌" + "─" * 56 + "┐")
        joints = [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]
        j_str = "  ".join(f"{v/1000:7.2f}°" for v in joints)
        print(f"│ 关节:  {j_str} │")
        print(f"│ 末端:  X={p.X_axis/1e6:7.3f}m  Y={p.Y_axis/1e6:7.3f}m  Z={p.Z_axis/1e6:7.3f}m             │")
        print(f"│         RX={p.RX_axis/1000:7.2f}° RY={p.RY_axis/1000:7.2f}° RZ={p.RZ_axis/1000:7.2f}°                           │")
        print(f"│ 夹爪:  行程={g.grippers_angle/1000:6.1f}mm  力矩={g.grippers_effort/1000:5.2f}Nm  状态=0x{g._status_code:02X}               │")
        m = mode_names.get(s.mode_feed, "?")
        print(f"│ 臂:    模式={m}  ctrl={s.ctrl_mode}  arm={s.arm_status}  err=0x{s._err_code:04X}                     │")
        print(f"│ 步长:  位置{self.step_mm:.0f}mm  角度{self.step_deg:.0f}°  速度{self._speed}%                            │")
        print("└" + "─" * 56 + "┘")
        print()

    def status_compact(self):
        """简洁一行状态，用于步进菜单。"""
        try:
            p = self._p.GetArmEndPoseMsgs().end_pose
            j = self._p.GetArmJointMsgs().joint_state
            g = self._p.GetArmGripperMsgs().gripper_state
        except Exception:
            return
        j_deg = [f"{v/1000:.1f}" for v in
                 [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]]
        print(f"  末端: X={p.X_axis/1e6:.3f}m Y={p.Y_axis/1e6:.3f}m Z={p.Z_axis/1e6:.3f}m  "
              f"RX={p.RX_axis/1000:.2f}° RY={p.RY_axis/1000:.2f}° RZ={p.RZ_axis/1000:.2f}°")
        print(f"  关节: [{', '.join(j_deg)}]°  "
              f"夹爪={g.grippers_angle/1000:.0f}mm  速度={self._speed}%")


# ════════════════════════ sub-menus ════════════════════════

def menu_end_pose_abs(arm: Arm):
    """末端绝对位姿控制。"""
    while True:
        arm.status()
        print("── 末端绝对位姿 ──")
        print("  格式: [L] x y z [rx ry rz]")
        print("  例: 300 0 400        → 点到点, 姿态不变")
        print("  例: L 300 0 200      → 直线移动")
        print("  例: 300 0 400 0 0 90 → 带姿态")
        print("  b=返回  q=退出")

        s = input("末端绝对> ").strip()
        if s == "":
            continue
        low = s.lower()
        if low == "b":
            return
        if low == "q":
            _quit(arm)

        parts = s.split()
        linear = False
        if parts[0].upper() == "L":
            linear = True
            parts = parts[1:]
        elif parts[0].upper() == "P":
            parts = parts[1:]

        try:
            nums = [float(x) for x in parts]
        except ValueError:
            print("[!] 无法解析。例: 300 0 400")
            continue
        if len(nums) < 3 or len(nums) > 6:
            print("[!] 需要 3~6 个值 (x y z [rx ry rz])")
            continue

        x, y, z = nums[0], nums[1], nums[2]
        rx = ry = rz = 0.0
        if len(nums) >= 6:
            rx, ry, rz = nums[3], nums[4], nums[5]
        elif len(nums) >= 5:
            rx, ry = nums[3], nums[4]
        elif len(nums) >= 4:
            rx = nums[3]

        print(f"[→] X={x:.1f} Y={y:.1f} Z={z:.1f}  "
              f"RX={rx:.1f} RY={ry:.1f} RZ={rz:.1f}  "
              f"({'直线' if linear else '点到点'})")
        arm.end_pose(x, y, z, rx, ry, rz, linear=linear)


def menu_end_pose_jog(arm: Arm):
    """末端步进控制。"""
    while True:
        arm.status_compact()
        print("── 末端步进 ──")
        print("  格式: <轴><+/-><量>")
        print("  例: x+    → X +10mm (默认步长)")
        print("  例: z-5   → Z  -5mm")
        print("  例: rx+3  → RX +3°")
        print("  设步长: step_pos 20 | step_rot 10")
        print("  b=返回  q=退出")
        print(f"  当前步长: 位置{arm.step_mm:.0f}mm  姿态{arm.step_deg:.0f}°")

        s = input("末端步进> ").strip()
        if s == "":
            continue
        low = s.lower()
        if low == "b":
            return
        if low == "q":
            _quit(arm)

        # step size config
        if low.startswith("step_pos "):
            try:
                arm.step_mm = float(s.split()[1])
                print(f"[OK] 位置步长 → {arm.step_mm:.0f}mm")
            except (ValueError, IndexError):
                print("[!] 用法: step_pos <mm>")
            continue
        if low.startswith("step_rot "):
            try:
                arm.step_deg = float(s.split()[1])
                print(f"[OK] 角度步长 → {arm.step_deg:.0f}°")
            except (ValueError, IndexError):
                print("[!] 用法: step_rot <deg>")
            continue

        # jog: x+, z-5, rx+1.5, etc.
        m = re.match(r'^(x|y|z|rx|ry|rz)([+\-])(\d*\.?\d*)$', low)
        if not m:
            print("[!] 格式错误。例: x+  z-5  rx+3")
            continue

        axis = m.group(1)
        sign = m.group(2)
        val_str = m.group(3)
        if val_str:
            delta = float(val_str)
        else:
            delta = arm.step_mm if axis in ('x', 'y', 'z') else arm.step_deg
        if sign == '-':
            delta = -delta

        print(f"[→] 步进 {axis.upper()} {delta:+.1f}")
        arm.jog_end_pose(axis, delta)


def menu_gripper(arm: Arm):
    """夹爪控制。"""
    while True:
        arm.status()
        print("── 夹爪控制 ──")
        print("  格式: <开度mm> [力矩Nm]")
        print("  ┌──────┬──────────────────┐")
        print("  │ 输入 │ 含义             │")
        print("  ├──────┼──────────────────┤")
        print("  │ 0    │ 闭合, 力矩1Nm    │")
        print("  │ 50   │ 张开50mm, 力矩1Nm│")
        print("  │ 30 2 │ 张开30mm, 力矩2Nm│")
        print("  │ 0 3  │ 闭合, 力矩3Nm    │")
        print("  └──────┴──────────────────┘")
        print("  b=返回  q=退出")

        s = input("夹爪> ").strip()
        if s == "":
            continue
        if s.lower() == "b":
            return
        if s.lower() == "q":
            _quit(arm)

        parts = s.split()
        try:
            width = float(parts[0])
            effort = float(parts[1]) if len(parts) >= 2 else 1.0
        except ValueError:
            print("[!] 无法解析。例: 0 或 50 或 30 2")
            continue

        width = max(0, min(100, width))
        effort = max(0.1, min(5.0, effort))

        print(f"[→] 夹爪 → 开度={width:.0f}mm  力矩={effort:.1f}Nm")
        arm.gripper(width, effort)


def menu_joints_abs(arm: Arm):
    """关节绝对角度控制。"""
    while True:
        arm.status()
        print("── 关节绝对角度 ──")
        print("  格式: j1 j2 j3 j4 j5 j6   (度)")
        print("  例: 30 20 -15 10 0 0")
        print("  限位: J1±150 J2[0~180] J3[-170~0] J4±100 J5±70 J6±120")
        print("  h=回零  b=返回  q=退出")

        s = input("关节绝对> ").strip()
        if s == "":
            continue
        low = s.lower()
        if low == "b":
            return
        if low == "q":
            _quit(arm)
        if low == "h":
            arm.home()
            continue

        try:
            vals = [float(x) for x in s.split()]
        except ValueError:
            print("[!] 无法解析。例: 30 20 -15 10 0 0")
            continue
        if len(vals) < 1 or len(vals) > 6:
            print("[!] 请输入 1~6 个角度值")
            continue
        while len(vals) < 6:
            vals.append(0.0)

        print(f"[→] 关节 → {[f'{v:.1f}°' for v in vals]}")
        arm.joints_deg(vals)


def menu_joints_jog(arm: Arm):
    """关节步进控制。"""
    while True:
        arm.status_compact()
        print("── 关节步进 ──")
        print("  格式: j<轴号><+/-><量>")
        print("  例: j1+   → J1 +5° (默认步长)")
        print("  例: j3-10 → J3 -10°")
        print("  例: j2+3  → J2 +3°")
        print(f"  当前步长: {arm.step_deg:.0f}°")
        print("  h=回零  b=返回  q=退出")

        s = input("关节步进> ").strip()
        if s == "":
            continue
        low = s.lower()
        if low == "b":
            return
        if low == "q":
            _quit(arm)
        if low == "h":
            arm.home()
            continue

        m = re.match(r'^j([1-6])([+\-])(\d*\.?\d*)$', low)
        if not m:
            print("[!] 格式错误。例: j1+  j3-10  j2+3")
            continue

        joint_idx = int(m.group(1))
        sign = m.group(2)
        val_str = m.group(3)
        delta = float(val_str) if val_str else arm.step_deg
        if sign == '-':
            delta = -delta

        print(f"[→] 步进 J{joint_idx} {delta:+.1f}°")
        arm.jog_joint(joint_idx, delta)


def _quit(arm: Arm):
    print("[→] 退出 (保持使能)...")
    arm.disconnect()
    sys.exit(0)


# ════════════════════════ main ════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Piper 机械臂交互控制")
    parser.add_argument("--can", default="can0", help="CAN 端口名 (默认: can0)")
    args = parser.parse_args()

    arm = Arm(args.can)

    if not arm.connect():
        sys.exit(1)

    arm.gripper(50.0)

    try:
        while True:
            arm.status()
            print("┌─────────────── 主菜单 ───────────────┐")
            print("│  1. 末端绝对控制  │  2. 末端步进     │")
            print("│  3. 夹爪控制      │  4. 关节绝对控制 │")
            print("│  5. 关节步进      │                  │")
            print("│  s. 状态  h. 回零  e. 急停  r. 复位 │")
            print("│  v. 速度  d. 失能  q. 退出(不掉落)  │")
            print("└─────────────────────────────────────┘")

            try:
                cmd = input("主菜单> ").strip()
            except EOFError:
                _quit(arm)
                return

            if cmd == "":
                continue

            c = cmd.lower()

            if c == "1":
                menu_end_pose_abs(arm)
            elif c == "2":
                menu_end_pose_jog(arm)
            elif c == "3":
                menu_gripper(arm)
            elif c == "4":
                menu_joints_abs(arm)
            elif c == "5":
                menu_joints_jog(arm)
            elif c == "s":
                continue
            elif c == "h":
                arm.home()
            elif c == "e":
                arm.estop()
            elif c == "r":
                arm.reset()
            elif c == "v":
                s = input(f"速度百分比 (当前 {arm.speed}%, 1-100): ").strip()
                if s:
                    try:
                        arm.speed = int(s)
                        print(f"[OK] 速度 → {arm.speed}%")
                    except ValueError:
                        print("[!] 无效值")
            elif c == "d":
                confirm = input("确认失能? 机械臂会掉落! (yes/no): ").strip()
                if confirm.lower() == "yes":
                    arm.disable()
            elif c == "q":
                _quit(arm)
            else:
                print(f"[!] 未知选项: {cmd}")

    except KeyboardInterrupt:
        print("\n[→] 用户中断 (保持使能)...")
        arm.disconnect()
    except Exception as e:
        print(f"\n[ERR] {e}")
        arm.disconnect()


if __name__ == "__main__":
    main()

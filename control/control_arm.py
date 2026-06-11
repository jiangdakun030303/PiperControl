#!/usr/bin/env python3
"""Piper 机械臂控制脚本 —— 演示连接、关节/笛卡尔/夹爪控制与状态读取。"""

import time
import sys
from piper_sdk import C_PiperInterface_V2
import numpy as np


class PiperController:
    """Piper 机械臂控制器封装。"""

    RAD2JOINT = 57295.7795   # 弧度 → 关节指令 (0.001°)
    M2POSE = 1_000_000       # 米 → 位姿指令 (0.001mm)
    MM2POSE = 1000           # 毫米 → 位姿指令
    MM2GRIPPER = 1000        # 毫米 → 夹爪指令 (0.001mm)
    DEG2POSE = 1000          # 度 → 姿态指令 (0.001°)

    # 关节限位 (rad)
    JOINT_LIMITS = [
        (-2.62, 2.62),
        (0.0, 3.14),
        (-2.97, 0.0),
        (-1.75, 1.75),
        (-1.22, 1.22),
        (-2.09, 2.09),
    ]

    def __init__(self, can_name: str = "can0"):
        self.can_name = can_name
        self.piper = None

    # ------- 连接 / 断开 -------

    def connect(self) -> bool:
        try:
            self.piper = C_PiperInterface_V2(self.can_name)
        except Exception as e:
            print(f"[错误] 创建接口失败: {e}")
            return False
        self.piper.ConnectPort()
        time.sleep(0.3)
        if not self.piper.isOk():
            print(f"[错误] CAN 端口 {self.can_name} 连接失败")
            return False
        print(f"[信息] CAN 端口 {self.can_name} 已连接")
        return True

    def enable(self, timeout: float = 10.0) -> bool:
        t0 = time.time()
        while not self.piper.EnablePiper():
            if time.time() - t0 > timeout:
                print("[错误] 使能超时")
                return False
            time.sleep(0.01)
        print("[信息] 电机已使能")
        return True

    def disable(self):
        if self.piper:
            self.piper.DisablePiper()
            print("[信息] 电机已失能")

    def disconnect(self):
        if self.piper:
            self.piper.DisconnectPort()
            print("[信息] 已断开连接")

    # ------- 模式选择 -------

    def set_mode(self, move_mode: int, speed: int = 50):
        """设置控制模式。
        move_mode: 0=MOVEP, 1=MOVEJ, 2=MOVEL, 3=MOVEC, 4=MOVEM
        """
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=move_mode,
            move_spd_rate_ctrl=speed,
            is_mit_mode=0x00,
        )
        modes = {0: "MOVEP-位姿", 1: "MOVEJ-关节", 2: "MOVEL-直线", 3: "MOVEC-圆弧"}
        print(f"[信息] 控制模式: {modes.get(move_mode, '未知')}, 速度: {speed}%")

    # ------- 关节控制 (MOVE J) -------

    def move_joints_rad(self, joints: list[float], speed: int = 50):
        """按弧度值控制关节。"""
        self.set_mode(1, speed)
        cmd = [int(np.clip(j, *lim) * self.RAD2JOINT)
               for j, lim in zip(joints, self.JOINT_LIMITS)]
        self.piper.JointCtrl(*cmd)

    def move_joints_deg(self, joints: list[float], speed: int = 50):
        """按角度值控制关节。"""
        rad = [np.deg2rad(j) for j in joints]
        self.move_joints_rad(rad, speed)

    def go_home(self, speed: int = 50):
        """回到 HOME 位姿（关节全零）。"""
        print("[动作] 回到 HOME 位姿...")
        self.move_joints_rad([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], speed)

    # ------- 笛卡尔控制 (MOVE P) -------

    def move_pose_mm(self, x: float, y: float, z: float,
                      rx: float = 0, ry: float = 0, rz: float = 0,
                      speed: int = 50):
        """笛卡尔位姿控制（毫米 + 度）。"""
        self.set_mode(0, speed)
        self.piper.EndPoseCtrl(
            int(x * self.MM2POSE),
            int(y * self.MM2POSE),
            int(z * self.MM2POSE),
            int(rx * self.DEG2POSE),
            int(ry * self.DEG2POSE),
            int(rz * self.DEG2POSE),
        )

    # ------- 直线运动 (MOVE L) -------

    def move_linear_mm(self, x: float, y: float, z: float,
                        rx: float = 0, ry: float = 0, rz: float = 0,
                        speed: int = 50):
        """直线轨迹运动到目标位姿（毫米 + 度）。"""
        self.set_mode(2, speed)
        self.piper.EndPoseCtrl(
            int(x * self.MM2POSE),
            int(y * self.MM2POSE),
            int(z * self.MM2POSE),
            int(rx * self.DEG2POSE),
            int(ry * self.DEG2POSE),
            int(rz * self.DEG2POSE),
        )

    # ------- 夹爪控制 -------

    def gripper_open(self, width_mm: float = 50.0, effort: float = 1.0):
        """打开夹爪到指定宽度 (mm)。"""
        self.piper.GripperCtrl(
            int(width_mm * self.MM2GRIPPER),
            int(effort * 1000),
            gripper_code=0x01,
            set_zero=0x00,
        )
        print(f"[动作] 夹爪打开 → {width_mm:.1f} mm")

    def gripper_close(self, width_mm: float = 0.0, effort: float = 1.0):
        """闭合夹爪到指定宽度 (mm)。"""
        self.gripper_open(width_mm, effort)

    # ------- 急停 / 复位 -------

    def emergency_stop(self):
        if self.piper:
            self.piper.MotionCtrl_1(0x01, 0, 0)
            print("[警告] 急停!")

    def reset(self):
        if self.piper:
            self.piper.MotionCtrl_1(0x02, 0, 0)
            print("[信息] 已复位")

    # ------- 状态读取 -------

    def print_status(self):
        """打印当前状态摘要。"""
        joint = self.piper.GetArmJointMsgs().joint_state
        pose = self.piper.GetArmEndPoseMsgs().end_pose
        status = self.piper.GetArmStatus().arm_status
        gripper = self.piper.GetArmGripperMsgs().gripper_state

        print("=" * 56)
        j_deg = [f"{v/1000:.2f}°" for v in
                 [joint.joint_1, joint.joint_2, joint.joint_3,
                  joint.joint_4, joint.joint_5, joint.joint_6]]
        print(f"关节: {j_deg}")
        print(f"末端: X={pose.X_axis/1e6:.3f}m, Y={pose.Y_axis/1e6:.3f}m, Z={pose.Z_axis/1e6:.3f}m  "
              f"RX={pose.RX_axis/1000:.2f}° RY={pose.RY_axis/1000:.2f}° RZ={pose.RZ_axis/1000:.2f}°")
        g = gripper
        print(f"夹爪: 行程={g.grippers_angle/1000:.1f}mm  力矩={g.grippers_effort/1000:.2f}Nm  状态=0x{g._status_code:02X}")
        print(f"臂状态: ctrl={status.ctrl_mode} arm={status.arm_status} "
              f"err=0x{status._err_code:04X}")
        print("=" * 56)


# ════════════════════════ 主程序 ════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Piper 机械臂控制脚本")
    parser.add_argument("--can", default="can0", help="CAN 端口名 (默认: can0)")
    parser.add_argument("--demo", action="store_true", help="运行演示序列")
    args = parser.parse_args()

    arm = PiperController(args.can)

    try:
        if not arm.connect():
            sys.exit(1)
        if not arm.enable():
            sys.exit(1)

        arm.gripper_open(50.0)

        if args.demo:
            # 演示序列
            arm.go_home(speed=50)
            time.sleep(3)
            arm.print_status()

            arm.gripper_close(0.0)
            time.sleep(1)
            arm.gripper_open(50.0)
            time.sleep(1)

            arm.move_joints_deg([30, 20, -15, 10, 0, 0], speed=30)
            time.sleep(3)
            arm.print_status()

            arm.go_home(speed=30)
            time.sleep(3)
            arm.gripper_close(30.0)
            print("[完成] 演示结束")
        else:
            # 交互模式
            print("\n命令: j <j1>..<j6> 关节角(度) | p <x> <y> <z> 位姿(mm)"
                  " | g <开度mm> | s 状态 | home | stop | reset | q 退出\n")
            while True:
                try:
                    raw = input("> ").strip()
                except EOFError:
                    break
                if not raw:
                    continue
                parts = raw.split()
                cmd = parts[0].lower()

                if cmd == "q":
                    break
                elif cmd == "s":
                    arm.print_status()
                elif cmd == "home":
                    arm.go_home()
                elif cmd == "stop":
                    arm.emergency_stop()
                elif cmd == "reset":
                    arm.reset()
                    time.sleep(0.5)
                    arm.enable()
                elif cmd == "j" and len(parts) >= 7:
                    vals = [float(p) for p in parts[1:7]]
                    arm.move_joints_deg(vals)
                elif cmd == "p" and len(parts) >= 4:
                    xyz = [float(p) for p in parts[1:4]]
                    arm.move_pose_mm(*xyz)
                elif cmd == "g" and len(parts) >= 2:
                    arm.gripper_open(float(parts[1]))
                else:
                    print("未知命令或参数不足")

    except KeyboardInterrupt:
        print("\n[信息] 用户中断")
    finally:
        if arm.piper:
            arm.reset()
            arm.disable()
            arm.disconnect()


if __name__ == "__main__":
    main()

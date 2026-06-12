#!/usr/bin/env python3
"""机械臂定点控制脚本 —— 按预设路径点依次运动，等待到达后继续。

用法:
  python waypoint_control.py --demo                              # 内置演示序列
  python waypoint_control.py --pick-place                        # pick-and-place 示例
  python waypoint_control.py -w my_pts.json                      # 从 JSON 文件加载路径点
  python waypoint_control.py --demo --loop 3 --dwell 0.5         # 循环 3 次, 每步停 0.5s
  python waypoint_control.py --demo --no-home                    # 跳过初始 HOME
  python waypoint_control.py --demo --hold                       # 完成后保持连接
  python waypoint_control.py --demo --can can0 --speed 30        # 指定 CAN 端口
"""

import time
import sys
import os
import json
import argparse
import numpy as np

# 确保 SDK 在 sys.path 中
_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from piper_sdk import C_PiperInterface_V2


class WaypointController:
    """机械臂路径点控制器。"""

    RAD2JOINT = 57295.7795
    M2POSE = 1_000_000
    MM2POSE = 1000
    DEG2POSE = 1000
    MM2GRIPPER = 1000

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

    # ------- 连接 / 使能 / 断开 -------

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

    def disconnect(self):
        if self.piper:
            self.piper.DisconnectPort()
            print("[信息] 已断开连接")

    def reset(self):
        if self.piper:
            self.piper.MotionCtrl_1(0x02, 0, 0)

    # ------- 模式设置 -------

    def _set_mode(self, move_mode: int, speed: int = 50):
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=move_mode,
            move_spd_rate_ctrl=speed,
            is_mit_mode=0x00,
        )

    # ------- 关节运动 -------

    def move_joints_rad(self, joints: list[float], speed: int = 50):
        self._set_mode(0x01, speed)  # MOVE J
        cmd = [int(np.clip(j, *lim) * self.RAD2JOINT)
               for j, lim in zip(joints, self.JOINT_LIMITS)]
        self.piper.JointCtrl(*cmd)

    def move_joints_deg(self, joints: list[float], speed: int = 50):
        self.move_joints_rad([np.deg2rad(j) for j in joints], speed)

    def go_home(self, speed: int = 50):
        print("[动作] HOME")
        self.move_joints_rad([0.0] * 6, speed)

    # ------- 笛卡尔运动 -------

    def move_pose_mm(self, x: float, y: float, z: float,
                     rx: float = 0, ry: float = 0, rz: float = 0,
                     speed: int = 50):
        self._set_mode(0x00, speed)  # MOVE P
        self.piper.EndPoseCtrl(
            int(x * self.MM2POSE), int(y * self.MM2POSE), int(z * self.MM2POSE),
            int(rx * self.DEG2POSE), int(ry * self.DEG2POSE), int(rz * self.DEG2POSE),
        )

    def move_linear_mm(self, x: float, y: float, z: float,
                       rx: float = 0, ry: float = 0, rz: float = 0,
                       speed: int = 50):
        self._set_mode(0x02, speed)  # MOVE L
        self.piper.EndPoseCtrl(
            int(x * self.MM2POSE), int(y * self.MM2POSE), int(z * self.MM2POSE),
            int(rx * self.DEG2POSE), int(ry * self.DEG2POSE), int(rz * self.DEG2POSE),
        )

    # ------- 夹爪 -------

    def gripper_open(self, width_mm: float = 50.0, effort: float = 1.0):
        self.piper.GripperCtrl(
            int(width_mm * self.MM2GRIPPER), int(effort * 1000),
            gripper_code=0x01, set_zero=0x00,
        )

    # ------- 状态读取 -------

    def _read_pose(self) -> dict:
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return {
            "x": ep.X_axis / self.M2POSE,
            "y": ep.Y_axis / self.M2POSE,
            "z": ep.Z_axis / self.M2POSE,
            "rx": ep.RX_axis / self.DEG2POSE,
            "ry": ep.RY_axis / self.DEG2POSE,
            "rz": ep.RZ_axis / self.DEG2POSE,
        }

    def _read_joints_deg(self) -> list:
        j = self.piper.GetArmJointMsgs().joint_state
        return [v / 1000.0 for v in
                [j.joint_1, j.joint_2, j.joint_3,
                 j.joint_4, j.joint_5, j.joint_6]]

    def _read_status(self):
        return self.piper.GetArmStatus().arm_status

    def print_pose(self):
        p = self._read_pose()
        j = self._read_joints_deg()
        s = self._read_status()
        sys.stdout.write(
            f"末端: X={p['x']:.4f}m Y={p['y']:.4f}m Z={p['z']:.4f}m  "
            f"RX={p['rx']:.1f}° RY={p['ry']:.1f}° RZ={p['rz']:.1f}°\n"
            f"关节: [{', '.join(f'{v:.1f}°' for v in j)}]\n"
            f"状态: motion={s.motion_status.name} arm={s.arm_status.name}\n"
        )
        sys.stdout.flush()

    def motion_status(self) -> int:
        return self._read_status().motion_status

    def arm_status(self) -> int:
        return self._read_status().arm_status

    # ------- 到达等待 -------

    def wait_arrival(self, timeout: float = 60.0) -> bool:
        t0 = time.time()
        last_print = 0.0
        while time.time() - t0 < timeout:
            ms = self.motion_status()
            arm_st = self.arm_status()

            if ms == 0x00:  # REACH_TARGET_POS_SUCCESSFULLY
                return True

            now = time.time()
            if now - last_print > 1.0:
                p = self._read_pose()
                sys.stdout.write(
                    f"  ... 运动中 motion={ms.name} arm={arm_st.name} "
                    f"当前: ({p['x']:.4f}, {p['y']:.4f}, {p['z']:.4f})m\n"
                )
                sys.stdout.flush()
                last_print = now

            if arm_st == 0x01:   # EMERGENCY_STOP
                sys.stdout.write("  [错误] 急停!\n"); sys.stdout.flush()
                return False
            if arm_st == 0x04:   # TARGET_POS_EXCEEDS_LIMIT
                sys.stdout.write("  [错误] 目标位置超出限位!\n"); sys.stdout.flush()
                return False
            if arm_st in (0x02, 0x03, 0x05, 0x06, 0x07):
                sys.stdout.write(f"  [错误] 臂状态异常: {arm_st.name}\n"); sys.stdout.flush()
                return False

            time.sleep(0.1)

        sys.stdout.write(f"  [警告] 到达超时 ({timeout}s), motion={ms.name}\n")
        sys.stdout.flush()
        return False


# ════════════════════ 路径点执行 ════════════════════

def load_waypoints(filepath: str) -> list[dict]:
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "waypoints" in data:
        return data["waypoints"]
    raise ValueError("JSON 格式错误: 需要路径点数组或含 'waypoints' 键的对象")


def execute_waypoint(ctrl: WaypointController, wp: dict, dwell: float, move_timeout: float):
    """执行单个路径点，返回 True=成功 False=失败。"""
    kind = wp.get("type", "joint")

    if kind == "joint":
        joints = wp["joints"]
        speed = wp.get("speed", 50)
        unit = wp.get("unit", "deg")
        label = wp.get("label", "")
        sys.stdout.write(f"[路径点] {label}\n" if label else f"[路径点] 关节: {joints}\n")
        sys.stdout.flush()
        (ctrl.move_joints_deg(joints, speed) if unit != "rad"
         else ctrl.move_joints_rad(joints, speed))
        if not ctrl.wait_arrival(move_timeout):
            return False

    elif kind == "pose":
        x, y, z = wp["x"], wp["y"], wp["z"]
        # 未指定姿态时保持当前姿态
        cur = ctrl._read_pose()
        rx = wp.get("rx", cur["rx"])
        ry = wp.get("ry", cur["ry"])
        rz = wp.get("rz", cur["rz"])
        speed = wp.get("speed", 50)
        move_type = wp.get("move_type", "p")
        label = wp.get("label", "")
        sys.stdout.write(
            (f"[路径点] {label}\n" if label else "[路径点] ") +
            f"目标: ({x/1000:.4f}, {y/1000:.4f}, {z/1000:.4f})m  "
            f"姿态: RX={rx:.1f}° RY={ry:.1f}° RZ={rz:.1f}°  "
            f"模式={'MOVEL' if move_type == 'l' else 'MOVEP'}  速度={speed}%\n"
        )
        sys.stdout.flush()
        if move_type == "l":
            ctrl.move_linear_mm(x, y, z, rx, ry, rz, speed)
        else:
            ctrl.move_pose_mm(x, y, z, rx, ry, rz, speed)
        if not ctrl.wait_arrival(move_timeout):
            return False

    elif kind == "gripper":
        width, effort = wp.get("width", 50.0), wp.get("effort", 1.0)
        sys.stdout.write(f"[路径点] 夹爪 → {width:.1f} mm\n"); sys.stdout.flush()
        ctrl.gripper_open(width, effort)

    elif kind == "home":
        speed = wp.get("speed", 50)
        ctrl.go_home(speed)
        if not ctrl.wait_arrival(move_timeout):
            return False

    elif kind == "wait":
        seconds = wp.get("seconds", dwell)
        sys.stdout.write(f"[路径点] 等待 {seconds:.1f}s\n"); sys.stdout.flush()
        time.sleep(seconds)
        return True

    else:
        sys.stdout.write(f"[警告] 未知路径点类型: {kind}\n"); sys.stdout.flush()
        return False

    if dwell > 0:
        time.sleep(dwell)
    return True


def run_waypoints(ctrl: WaypointController, waypoints: list[dict],
                  dwell: float, loops: int, home_first: bool, move_timeout: float):
    if home_first:
        ctrl.go_home(speed=30)
        ctrl.wait_arrival(move_timeout)
        time.sleep(dwell)

    for loop in range(loops):
        if loops > 1:
            sys.stdout.write(f"\n--- 第 {loop + 1}/{loops} 轮 ---\n"); sys.stdout.flush()
        for i, wp in enumerate(waypoints):
            sys.stdout.write(f"\n  [{i+1}/{len(waypoints)}] "); sys.stdout.flush()
            ok = execute_waypoint(ctrl, wp, dwell, move_timeout)
            ctrl.print_pose()
            if not ok:
                sys.stdout.write("[中断] 路径点执行失败\n"); sys.stdout.flush()
                return


# ════════════════════ 内置示例 ════════════════════

def builtin_demo():
    return [
        {"type": "home", "speed": 30},
        {"type": "gripper", "width": 50.0, "label": "张开夹爪"},
        {"type": "joint", "joints": [30, 20, -15, 10, 0, 0], "speed": 30, "label": "关节位姿 A"},
        {"type": "pose", "x": 300, "y": 0, "z": 400, "speed": 30, "label": "笛卡尔位姿 B"},
        {"type": "gripper", "width": 10.0, "label": "闭合夹爪"},
        {"type": "pose", "x": 300, "y": 0, "z": 200, "speed": 20, "move_type": "l", "label": "直线下降"},
        {"type": "gripper", "width": 50.0, "label": "张开夹爪"},
        {"type": "pose", "x": 300, "y": 0, "z": 400, "speed": 20, "move_type": "l", "label": "直线上升"},
        {"type": "home", "speed": 30},
    ]


def builtin_pick_place():
    return [
        {"type": "home", "speed": 30},
        {"type": "gripper", "width": 60.0},
        {"type": "pose", "x": 250, "y": -100, "z": 300, "speed": 30, "label": "取料点上方"},
        {"type": "pose", "x": 250, "y": -100, "z": 150, "speed": 20, "move_type": "l", "label": "下降取料"},
        {"type": "gripper", "width": 5.0, "label": "夹取"},
        {"type": "pose", "x": 250, "y": -100, "z": 300, "speed": 20, "move_type": "l", "label": "抬升"},
        {"type": "pose", "x": 250, "y": 100, "z": 300, "speed": 30, "label": "放料点上方"},
        {"type": "pose", "x": 250, "y": 100, "z": 150, "speed": 20, "move_type": "l", "label": "下降放料"},
        {"type": "gripper", "width": 60.0, "label": "释放"},
        {"type": "pose", "x": 250, "y": 100, "z": 300, "speed": 20, "move_type": "l", "label": "抬升"},
        {"type": "home", "speed": 30},
    ]


# ════════════════════ 主程序 ════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Piper 机械臂定点控制 —— 按预设路径点依次运动，等待到达后继续",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --demo                   运行内置演示
  %(prog)s --pick-place             运行 pick-and-place 示例
  %(prog)s -w my_pts.json           从 JSON 文件加载路径点
  %(prog)s --demo --loop 3          演示序列循环 3 次
  %(prog)s --demo --dwell 0.5       每个路径点到达后额外停留 0.5s
  %(prog)s --demo --no-home         跳过初始 HOME
  %(prog)s -w pts.json --hold       序列完成后保持连接 (Ctrl+C 退出)

JSON 路径点格式:
  [
    {"type": "home", "speed": 30},
    {"type": "joint",  "joints": [10, 20, -30, 0, 0, 0], "speed": 50, "unit": "deg"},
    {"type": "pose",   "x": 300, "y": 0, "z": 400, "rx": 0, "ry": 0, "rz": 0, "speed": 30, "move_type": "p"},
    {"type": "gripper","width": 10.0, "effort": 1.0},
    {"type": "wait",   "seconds": 2.0}
  ]
  type: home | joint | pose | gripper | wait
  joint.unit: "deg" (默认) | "rad"
  pose 坐标单位: mm (x=98.1 即 0.0981m)
  pose.move_type: "p"=点到点 (默认) | "l"=直线
        """,
    )
    parser.add_argument("--can", default="can0", help="CAN 端口名 (默认: can0)")
    parser.add_argument("--demo", action="store_true", help="运行内置演示序列")
    parser.add_argument("--pick-place", action="store_true", help="运行 pick-and-place 示例")
    parser.add_argument("--waypoints", "-w", help="JSON 路径点文件路径")
    parser.add_argument("--dwell", "-d", type=float, default=0.5,
                        help="每个路径点到达后的额外停留时间 (秒, 默认 0.5)")
    parser.add_argument("--move-timeout", "-t", type=float, default=60.0,
                        help="单次运动到达超时 (秒, 默认 60)")
    parser.add_argument("--loop", "-n", type=int, default=1,
                        help="循环执行次数 (默认 1)")
    parser.add_argument("--no-home", action="store_true", help="跳过初始 HOME")
    parser.add_argument("--hold", action="store_true",
                        help="序列完成后保持连接不退出 (Ctrl+C 退出, 保持使能)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印路径点序列, 不执行")

    args = parser.parse_args()

    if args.waypoints:
        waypoints = load_waypoints(args.waypoints)
        print(f"[加载] {args.waypoints} ({len(waypoints)} 个路径点)")
    elif args.pick_place:
        waypoints = builtin_pick_place()
        print(f"[内置] pick-and-place 序列 ({len(waypoints)} 个路径点)")
    elif args.demo:
        waypoints = builtin_demo()
        print(f"[内置] 演示序列 ({len(waypoints)} 个路径点)")
    else:
        print("错误: 请指定 --demo / --pick-place / --waypoints")
        sys.exit(1)

    if args.dry_run:
        for i, wp in enumerate(waypoints):
            print(f"  {i+1}. {wp}")
        return

    ctrl = WaypointController(args.can)

    try:
        ctrl.connect()
        ctrl.enable()

        print(f"\n路径点数: {len(waypoints)} | 到达后停留: {args.dwell}s | "
              f"运动超时: {args.move_timeout}s | 循环: {args.loop} 次")
        print("按 Ctrl+C 中断\n")

        print("--- 初始状态 ---")
        ctrl.print_pose()

        run_waypoints(ctrl, waypoints, args.dwell, args.loop,
                      home_first=not args.no_home,
                      move_timeout=args.move_timeout)

        print("\n--- 最终状态 ---")
        ctrl.print_pose()
        print("[完成] 路径点序列执行完毕")

        if args.hold:
            print("\n[保持] 连接保持中, 机械臂维持当前位置 (Ctrl+C 退出)...")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[退出] (保持使能)")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        if ctrl.piper:
            ctrl.disconnect()


if __name__ == "__main__":
    main()

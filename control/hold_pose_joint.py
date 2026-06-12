#!/usr/bin/env python3
"""指定位姿控制 (关节角) —— IK 求解 → 下发关节角 → 保持, 防止卸力。"""

import time
import sys
import math
import argparse
import os

_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from piper_sdk import C_PiperInterface_V2
from piper_sdk.kinematics.piper_ik_pinocchio import C_PiperIKPinocchio

M2POSE = 1_000_000
DEG2POSE = 1000
RAD2JOINT = 57295.7795
JOINT_LIMITS = [
    (-2.62, 2.62), (0.0, 3.14), (-2.97, 0.0),
    (-1.75, 1.75), (-1.22, 1.22), (-2.09, 2.09),
]


def main():
    parser = argparse.ArgumentParser(description="IK 求解位姿, 下发关节角, 保持")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--speed", type=int, default=30, help="速度 0-100")
    parser.add_argument("--timeout", type=float, default=60.0, help="到达超时 (秒)")
    parser.add_argument("--keep", type=float, default=0.5, help="保持时重发指令间隔 (秒)")
    parser.add_argument("--x", type=float, required=True, help="X (m)")
    parser.add_argument("--y", type=float, required=True, help="Y (m)")
    parser.add_argument("--z", type=float, required=True, help="Z (m)")
    parser.add_argument("--rx", type=float, default=0.0, help="RX (deg)")
    parser.add_argument("--ry", type=float, default=0.0, help="RY (deg)")
    parser.add_argument("--rz", type=float, default=0.0, help="RZ (deg)")
    parser.add_argument("--position-only", action="store_true",
                        help="仅位置 IK (忽略姿态, 保持当前姿态)")
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.can)
    ik = C_PiperIKPinocchio()

    try:
        piper.ConnectPort()
        time.sleep(0.3)
        print(f"[信息] CAN {args.can} 已连接")

        while not piper.EnablePiper():
            time.sleep(0.01)
        print("[信息] 电机已使能")

        # 读取当前关节角作为 IK 初始值
        cur_j = piper.GetArmJointMsgs().joint_state
        cur_joints_rad = [v / 1000.0 for v in
                          [cur_j.joint_1, cur_j.joint_2, cur_j.joint_3,
                           cur_j.joint_4, cur_j.joint_5, cur_j.joint_6]]
        cur_joints_deg = [math.degrees(r) for r in cur_joints_rad]
        print(f"[当前关节] [{', '.join(f'{d:.1f}°' for d in cur_joints_deg)}]")

        # IK 求解
        if args.position_only:
            joints_rad = ik.solve_position(args.x, args.y, args.z, q0=cur_joints_rad)
        else:
            joints_rad = ik.solve_full(args.x, args.y, args.z,
                                       args.rx, args.ry, args.rz,
                                       q0=cur_joints_rad)
        joints_deg = [math.degrees(r) for r in joints_rad]
        joints_raw = [int(max(lim[0], min(lim[1], r)) * RAD2JOINT)
                      for r, lim in zip(joints_rad, JOINT_LIMITS)]

        print(f"[IK 目标关节] [{', '.join(f'{d:.1f}°' for d in joints_deg)}]")
        print(f"[IK 原始值  ] [{', '.join(f'{r:.4f}rad' for r in joints_rad)}]")
        print(f"[目标位姿  ] ({args.x:.4f}, {args.y:.4f}, {args.z:.4f})m  "
              f"RX={args.rx:.0f}° RY={args.ry:.0f}° RZ={args.rz:.0f}°")

        # 下发关节角 (MOVEJ)
        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                           move_spd_rate_ctrl=args.speed, is_mit_mode=0x00)
        piper.JointCtrl(*joints_raw)
        print(f"[指令] JointCtrl({' '.join(str(v) for v in joints_raw)})")

        # 等待到达: 比较实际关节角与目标关节角
        t0 = time.time()
        reached = False
        last_print = 0.0
        time.sleep(0.3)  # 等命令被处理
        while time.time() - t0 < args.timeout:
            j = piper.GetArmJointMsgs().joint_state
            actual_j = [j.joint_1/1000.0, j.joint_2/1000.0, j.joint_3/1000.0,
                        j.joint_4/1000.0, j.joint_5/1000.0, j.joint_6/1000.0]
            # 判断是否到达: 每个关节误差 < 0.5°
            errs = [abs(actual_j[i] - joints_deg[i]) for i in range(6)]
            if max(errs) < 0.5:
                reached = True
                break
            now = time.time()
            st = piper.GetArmStatus().arm_status
            if st.arm_status != 0x00:
                sys.stdout.write(f"\r  [错误] 臂状态: {st.arm_status.name}\n")
                sys.stdout.flush()
                break
            if now - last_print > 1.0:
                sys.stdout.write(
                    f"\r  ... 运动中 arm={st.arm_status.name}  "
                    f"关节: [{actual_j[0]:.1f} {actual_j[1]:.1f} {actual_j[2]:.1f} "
                    f"{actual_j[3]:.1f} {actual_j[4]:.1f} {actual_j[5]:.1f}]°  "
                    f"max_err={max(errs):.1f}°\n"
                )
                sys.stdout.flush()
                last_print = now
            time.sleep(0.1)

        if reached:
            j = piper.GetArmJointMsgs().joint_state
            actual_joints = [j.joint_1/1000.0, j.joint_2/1000.0, j.joint_3/1000.0,
                             j.joint_4/1000.0, j.joint_5/1000.0, j.joint_6/1000.0]
            ep = piper.GetArmEndPoseMsgs().end_pose
            actual_pose = (ep.X_axis / M2POSE, ep.Y_axis / M2POSE, ep.Z_axis / M2POSE,
                           ep.RX_axis / DEG2POSE, ep.RY_axis / DEG2POSE, ep.RZ_axis / DEG2POSE)

            # 用实际关节角重新跑 FK 对比 (Pinocchio FK 返回 m, rad)
            fk_result = ik.fk([math.radians(d) for d in actual_joints])
            fk_pose = (fk_result[0], fk_result[1], fk_result[2],
                       math.degrees(fk_result[3]), math.degrees(fk_result[4]), math.degrees(fk_result[5]))

            print(f"\n[到达]")
            print(f"  实际关节: [{', '.join(f'{d:.1f}°' for d in actual_joints)}]")
            print(f"  实际末端: ({actual_pose[0]:.4f}, {actual_pose[1]:.4f}, {actual_pose[2]:.4f})m  "
                  f"RX={actual_pose[3]:.1f}° RY={actual_pose[4]:.1f}° RZ={actual_pose[5]:.1f}°")
            print(f"  FK 验证 : ({fk_pose[0]:.4f}, {fk_pose[1]:.4f}, {fk_pose[2]:.4f})m  "
                  f"RX={fk_pose[3]:.1f}° RY={fk_pose[4]:.1f}° RZ={fk_pose[5]:.1f}°")
            print(f"  目标位姿: ({args.x:.4f}, {args.y:.4f}, {args.z:.4f})m  "
                  f"RX={args.rx:.0f}° RY={args.ry:.0f}° RZ={args.rz:.0f}°")
            if not args.position_only:
                err_pos = math.sqrt((actual_pose[0]-args.x)**2 + (actual_pose[1]-args.y)**2 + (actual_pose[2]-args.z)**2) * 1000
                print(f"  位姿误差: Δpos={err_pos:.1f}mm  "
                      f"ΔRX={actual_pose[3]-args.rx:.1f}° ΔRY={actual_pose[4]-args.ry:.1f}° ΔRZ={actual_pose[5]-args.rz:.1f}°")
        else:
            print(f"\n[警告] 到达超时 ({args.timeout}s)")

        # 保持: 循环重发关节角, 防止卸力
        print(f"\n[保持] 每 {args.keep}s 重发关节角, Ctrl+C 退出...")
        while True:
            piper.JointCtrl(*joints_raw)
            time.sleep(args.keep)

    except KeyboardInterrupt:
        print("\n[退出] (保持使能)")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()

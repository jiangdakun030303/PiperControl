#!/usr/bin/env python3
"""指定位姿控制 —— 移动到目标位姿后保持当前位置, 防止卸力。"""

import time
import sys
import argparse

from piper_sdk import C_PiperInterface_V2

M2POSE = 1_000_000
DEG2POSE = 1000


def main():
    parser = argparse.ArgumentParser(description="指定位姿控制, 到达后保持")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--speed", type=int, default=30, help="速度 0-100")
    parser.add_argument("--timeout", type=float, default=60.0, help="到达超时 (秒)")
    parser.add_argument("--x", type=float, required=True, help="X (m)")
    parser.add_argument("--y", type=float, required=True, help="Y (m)")
    parser.add_argument("--z", type=float, required=True, help="Z (m)")
    parser.add_argument("--rx", type=float, default=0.0, help="RX (deg)")
    parser.add_argument("--ry", type=float, default=0.0, help="RY (deg)")
    parser.add_argument("--rz", type=float, default=0.0, help="RZ (deg)")
    parser.add_argument("--keep", type=float, default=0.5, help="保持时重发指令间隔 (秒)")
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.can)
    try:
        piper.ConnectPort()
        time.sleep(0.3)
        print(f"[信息] CAN {args.can} 已连接")

        while not piper.EnablePiper():
            time.sleep(0.01)
        print("[信息] 电机已使能")
        # 下发目标位姿
        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00,
                           move_spd_rate_ctrl=args.speed, is_mit_mode=0x00)
        cmd = (int(args.x * M2POSE), int(args.y * M2POSE), int(args.z * M2POSE),
               int(args.rx * DEG2POSE), int(args.ry * DEG2POSE), int(args.rz * DEG2POSE))
        piper.EndPoseCtrl(*cmd)
        print(f"[指令] EndPoseCtrl X={cmd[0]} Y={cmd[1]} Z={cmd[2]} "
              f"RX={cmd[3]} RY={cmd[4]} RZ={cmd[5]}")
        print(f"[目标] ({args.x:.4f}, {args.y:.4f}, {args.z:.4f})m  "
              f"RX={args.rx:.0f}° RY={args.ry:.0f}° RZ={args.rz:.0f}°")
        # 等待到达
        t0 = time.time()
        reached = False
        last_print = 0.0
        while time.time() - t0 < args.timeout:
            st = piper.GetArmStatus().arm_status
            if st.motion_status == 0x00:
                reached = True
                break
            now = time.time()
            if now - last_print > 1.0:
                ep = piper.GetArmEndPoseMsgs().end_pose
                sys.stdout.write(
                    f"\r  ... 运动中 motion={st.motion_status.name}  "
                    f"当前: ({ep.X_axis/M2POSE:.4f}, {ep.Y_axis/M2POSE:.4f}, {ep.Z_axis/M2POSE:.4f})m\n"
                )
                sys.stdout.flush()
                last_print = now
            # 重发指令防止丢帧
            piper.EndPoseCtrl(*cmd)
            time.sleep(0.2)

        if reached:
            ep = piper.GetArmEndPoseMsgs().end_pose
            print(f"\n[到达] ({ep.X_axis/M2POSE:.4f}, {ep.Y_axis/M2POSE:.4f}, {ep.Z_axis/M2POSE:.4f})m  "
                  f"RX={ep.RX_axis/DEG2POSE:.1f}° RY={ep.RY_axis/DEG2POSE:.1f}° RZ={ep.RZ_axis/DEG2POSE:.1f}°")
        else:
            print(f"\n[警告] 到达超时 ({args.timeout}s)")

        # 保持: 循环重发指令, 防止卸力
        print(f"\n[保持] 每 {args.keep}s 重发指令, Ctrl+C 退出...")
        while True:
            piper.EndPoseCtrl(*cmd)
            time.sleep(args.keep)

    except KeyboardInterrupt:
        print("\n[退出] (保持使能)")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()

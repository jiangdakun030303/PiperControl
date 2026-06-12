#!/usr/bin/env python3
"""回零位脚本 —— 回到 HOME 后保持, 防止卸力。"""

import time
import sys
import argparse

from piper_sdk import C_PiperInterface_V2


def main():
    parser = argparse.ArgumentParser(description="机械臂回零位")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--speed", type=int, default=30, help="速度 0-100")
    parser.add_argument("--timeout", type=float, default=60.0, help="到达超时 (秒)")
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

        # HOME: MOVEJ 全零关节
        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                           move_spd_rate_ctrl=args.speed, is_mit_mode=0x00)
        piper.JointCtrl(0, 0, 0, 0, 0, 0)
        print("[指令] JointCtrl(0,0,0,0,0,0) → HOME")

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
                j = piper.GetArmJointMsgs().joint_state
                sys.stdout.write(
                    f"\r  ... 运动中 motion={st.motion_status.name}  "
                    f"关节: [{j.joint_1/1000:.0f} {j.joint_2/1000:.0f} {j.joint_3/1000:.0f} "
                    f"{j.joint_4/1000:.0f} {j.joint_5/1000:.0f} {j.joint_6/1000:.0f}]°\n"
                )
                sys.stdout.flush()
                last_print = now
            time.sleep(0.1)

        if reached:
            j = piper.GetArmJointMsgs().joint_state
            print(f"\n[HOME 到达] 关节: [{j.joint_1/1000:.0f} {j.joint_2/1000:.0f} {j.joint_3/1000:.0f} "
                  f"{j.joint_4/1000:.0f} {j.joint_5/1000:.0f} {j.joint_6/1000:.0f}]°")
        else:
            print(f"\n[警告] 到达超时 ({args.timeout}s)")

        # 保持, 防止卸力
        print(f"\n[保持] 每 {args.keep}s 重发 HOME 指令, Ctrl+C 退出...")
        while True:
            piper.JointCtrl(0, 0, 0, 0, 0, 0)
            time.sleep(args.keep)

    except KeyboardInterrupt:
        print("\n[退出] (保持使能)")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()

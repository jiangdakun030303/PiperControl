#!/usr/bin/env python3
"""关节轨迹播放 —— 加载处理后的 npz, 按时间戳逐帧下发 JointCtrl, 平滑回放。"""

import time
import sys
import os
import argparse
import numpy as np

_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from piper_sdk import C_PiperInterface_V2

RAD2JOINT = 57295.7795
M2POSE = 1_000_000
DEG2POSE = 1000
MM2GRIPPER = 1000

JOINT_LIMITS = [
    (-2.62, 2.62), (0.0, 3.14), (-2.97, 0.0),
    (-1.75, 1.75), (-1.22, 1.22), (-2.09, 2.09),
]


def joints_to_raw(joints_rad):
    """关节角 rad → 编码器原始值."""
    return [int(max(lim[0], min(lim[1], r)) * RAD2JOINT)
            for r, lim in zip(joints_rad, JOINT_LIMITS)]


def main():
    parser = argparse.ArgumentParser(description="关节轨迹播放")
    parser.add_argument("trajectory", help="处理后的 .npz 轨迹文件")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--home-speed", type=int, default=30, help="HOME 速度")
    parser.add_argument("--timeout", type=float, default=60.0, help="HOME 到达超时")
    parser.add_argument("--gripper", type=float, default=50.0, help="夹爪开度 (mm, 默认 50)")
    args = parser.parse_args()

    # 加载轨迹
    data = np.load(args.trajectory, allow_pickle=True)
    q_sampled = data["q_sampled"]       # (N, 6) rad
    t_sampled = data["t_sampled"]       # (N,)  s
    dt = t_sampled[1] - t_sampled[0]
    total_t = t_sampled[-1]
    n = len(t_sampled)
    print(f"[加载] {args.trajectory}")
    print(f"  采样点: {n}, 间隔: {dt*1000:.0f}ms, 总时长: {total_t:.1f}s")

    piper = C_PiperInterface_V2(args.can)

    try:
        piper.ConnectPort()
        time.sleep(0.3)
        print(f"[信息] CAN {args.can} 已连接")

        while not piper.EnablePiper():
            time.sleep(0.01)
        print("[信息] 电机已使能")

        # ---- 夹爪 ----
        piper.GripperCtrl(int(args.gripper * MM2GRIPPER), 1000, gripper_code=0x01, set_zero=0x00)
        print(f"[夹爪] 开 {args.gripper:.0f}mm")

        # ---- HOME ----
        print("\n[HOME] 回零位 ...")
        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                           move_spd_rate_ctrl=args.home_speed, is_mit_mode=0x00)
        piper.JointCtrl(0, 0, 0, 0, 0, 0)

        t0 = time.time()
        home_reached = False
        while time.time() - t0 < args.timeout:
            j = piper.GetArmJointMsgs().joint_state
            actual = [j.joint_1/1000.0, j.joint_2/1000.0, j.joint_3/1000.0,
                      j.joint_4/1000.0, j.joint_5/1000.0, j.joint_6/1000.0]
            errs = [abs(a) for a in actual]
            if max(errs) < 0.5:
                home_reached = True
                break
            time.sleep(0.1)
        if home_reached:
            print(f"  HOME 到达")
        else:
            print(f"  [警告] HOME 未完全到达")

        # ---- 预备: 移到第一个点 ----
        print(f"\n[预备] 移到轨迹起点 ...")
        q0_raw = joints_to_raw(q_sampled[0])
        piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                           move_spd_rate_ctrl=30, is_mit_mode=0x00)
        piper.JointCtrl(*q0_raw)
        q0_deg = np.degrees(q_sampled[0])
        t0_wait = time.time()
        while time.time() - t0_wait < args.timeout:
            j = piper.GetArmJointMsgs().joint_state
            actual = [j.joint_1/1000.0, j.joint_2/1000.0, j.joint_3/1000.0,
                      j.joint_4/1000.0, j.joint_5/1000.0, j.joint_6/1000.0]
            errs = [abs(actual[i] - q0_deg[i]) for i in range(6)]
            if max(errs) < 1.0:
                break
            time.sleep(0.1)
        print(f"  起点关节: [{', '.join(f'{d:.1f}deg' for d in q0_deg)}]")

        # ---- 播放 ----
        print(f"\n[播放] {n} 帧, {total_t:.1f}s, 按 Ctrl+C 中断")
        print(f"  {'时间':>6s}  {'J1':>7s}  {'J2':>7s}  {'J3':>7s}  {'J4':>7s}  {'J5':>7s}  {'J6':>7s}")
        sys.stdout.flush()

        t_start = time.time()
        frame = 0
        next_tick = t_start  # 绝对时间调度

        while frame < n:
            now = time.time()
            if now < next_tick:
                time.sleep(0.002)  # 2ms 粒度
                continue

            q = q_sampled[frame]
            raw = joints_to_raw(q)
            piper.JointCtrl(*raw)

            # 每秒打印一次进度
            if frame % max(1, int(1.0 / dt)) == 0:
                elapsed = now - t_start
                q_deg = np.degrees(q)
                sys.stdout.write(
                    f"\r  {elapsed:5.1f}s  "
                    f"{q_deg[0]:6.1f}  {q_deg[1]:6.1f}  {q_deg[2]:6.1f}  "
                    f"{q_deg[3]:6.1f}  {q_deg[4]:6.1f}  {q_deg[5]:6.1f}"
                )
                sys.stdout.flush()

            frame += 1
            next_tick = t_start + t_sampled[frame] if frame < n else float('inf')

        elapsed = time.time() - t_start
        print(f"\n[完成] {frame} 帧 / {elapsed:.1f}s (期望 {total_t:.1f}s)")

        # ---- 保持 ----
        print(f"\n[保持] 重发最后一帧 + 保持夹爪, Ctrl+C 退出...")
        last_raw = joints_to_raw(q_sampled[-1])
        gripper_raw = int(args.gripper * MM2GRIPPER)
        while True:
            piper.JointCtrl(*last_raw)
            piper.GripperCtrl(gripper_raw, 1000, gripper_code=0x01, set_zero=0x00)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[退出] (保持使能)")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()

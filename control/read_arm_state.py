#!/usr/bin/env python3
"""读取并打印 Piper 机械臂所有状态信息。"""

import time
import sys
from piper_sdk import C_PiperInterface_V2


class ArmStateReader:
    """读取机械臂所有反馈数据并格式化输出。"""

    MM2UNIT = 1e6   # 0.001mm → m
    DEG2UNIT = 1e3  # 0.001° → °
    RAD2DEG = 57.2957795
    TORQUE2NM = 1e3  # 0.001Nm → Nm
    CURRENT2A = 1e3   # 0.001A → A

    def __init__(self, can_name: str = "can0"):
        self.can_name = can_name
        self.piper = None

    def connect(self):
        try:
            self.piper = C_PiperInterface_V2(self.can_name)
        except Exception as e:
            raise ConnectionError(f"创建接口失败: {e}")
        self.piper.ConnectPort()
        time.sleep(0.5)  # 等待 CAN 读线程开始接收数据
        print(f"[信息] CAN {self.can_name} 已连接, FPS: {self.piper.GetCanFps():.0f}")

    def disconnect(self):
        if self.piper:
            self.piper.DisconnectPort()

    # -------------------------------------------------------
    def read_status(self):
        """读取机械臂状态。"""
        s = self.piper.GetArmStatus().arm_status
        print("\n" + "=" * 60)
        print("◆ 机械臂状态")
        print(f"  控制模式:    {s.ctrl_mode}")
        print(f"  臂状态:      {s.arm_status}")
        print(f"  当前模式:    {s.mode_feed}")
        print(f"  示教状态:    {s.teach_status}")
        print(f"  运动状态:    {s.motion_status}")
        print(f"  轨迹点序号:  {s.trajectory_num}")
        print(f"  故障码:      0x{s._err_code:04X}")
        es = s.err_status
        # 检查是否有异常
        angle_errors = []
        comm_errors = []
        for i, attr in enumerate([
            "joint_1_angle_limit", "joint_2_angle_limit", "joint_3_angle_limit",
            "joint_4_angle_limit", "joint_5_angle_limit", "joint_6_angle_limit",
        ], 1):
            if getattr(es, attr):
                angle_errors.append(f"J{i}")
        for i, attr in enumerate([
            "communication_status_joint_1", "communication_status_joint_2",
            "communication_status_joint_3", "communication_status_joint_4",
            "communication_status_joint_5", "communication_status_joint_6",
        ], 1):
            if getattr(es, attr):
                comm_errors.append(f"J{i}")
        if angle_errors:
            print(f"  [异常] 关节超限位: {', '.join(angle_errors)}")
        if comm_errors:
            print(f"  [异常] 通信异常:   {', '.join(comm_errors)}")
        if not angle_errors and not comm_errors:
            print("  故障详情: 正常")

    # -------------------------------------------------------
    def read_joints(self):
        """读取关节角度。"""
        j = self.piper.GetArmJointMsgs().joint_state
        deg = [v / self.DEG2UNIT for v in
               [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]]
        print("\n◆ 关节角度 (°)")
        for i, d in enumerate(deg, 1):
            print(f"  J{i}: {d: 8.2f}°  ({d * self.RAD2DEG / 180:.4f}π rad)")

    # -------------------------------------------------------
    def read_end_pose(self):
        """读取末端位姿。"""
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        print("\n◆ 末端位姿")
        print(f"  X:  {ep.X_axis / self.MM2UNIT:.3f} m  ({ep.X_axis / 1e3:.1f} mm)")
        print(f"  Y:  {ep.Y_axis / self.MM2UNIT:.3f} m  ({ep.Y_axis / 1e3:.1f} mm)")
        print(f"  Z:  {ep.Z_axis / self.MM2UNIT:.3f} m  ({ep.Z_axis / 1e3:.1f} mm)")
        print(f"  RX: {ep.RX_axis / self.DEG2UNIT:.2f}°")
        print(f"  RY: {ep.RY_axis / self.DEG2UNIT:.2f}°")
        print(f"  RZ: {ep.RZ_axis / self.DEG2UNIT:.2f}°")

    # -------------------------------------------------------
    def read_gripper(self):
        """读取夹爪状态。"""
        g = self.piper.GetArmGripperMsgs().gripper_state
        print("\n◆ 夹爪状态")
        print(f"  行程:    {g.grippers_angle / 1e3:.1f} mm")
        print(f"  力矩:    {g.grippers_effort / self.TORQUE2NM:.3f} Nm")
        fs = g.foc_status
        flags = []
        if fs.voltage_too_low:     flags.append("电压过低")
        if fs.motor_overheating:   flags.append("电机过温")
        if fs.driver_overcurrent:  flags.append("驱动器过流")
        if fs.driver_overheating:  flags.append("驱动器过温")
        if fs.sensor_status:       flags.append("传感器异常")
        if fs.driver_error_status: flags.append("驱动器错误")
        if fs.driver_enable_status: flags.append("[使能]")
        if fs.homing_status:       flags.append("[已回零]")
        print(f"  状态:     {', '.join(flags) if flags else '正常'}")

    # -------------------------------------------------------
    def read_motor_high_spd(self):
        """读取6个电机高速反馈（转速、电流、位置、力矩）。"""
        m = self.piper.GetArmHighSpdInfoMsgs()
        motors = [m.motor_1, m.motor_2, m.motor_3, m.motor_4, m.motor_5, m.motor_6]
        print("\n◆ 电机高速反馈")
        print(f"  {'电机':>5} {'转速(rad/s)':>13} {'电流(A)':>10} {'位置(rad)':>12} {'力矩(Nm)':>10}")
        for i, mot in enumerate(motors, 1):
            print(f"  M{i}:   {mot.motor_speed/1e3: 11.3f}  {mot.current/1e3: 9.3f}  "
                  f"{mot.pos/1e6: 11.4f}  {mot.effort: 9.4f}")

    # -------------------------------------------------------
    def read_motor_low_spd(self):
        """读取6个电机低速反馈（电压、温度、FOC状态）。"""
        m = self.piper.GetArmLowSpdInfoMsgs()
        motors = [m.motor_1, m.motor_2, m.motor_3, m.motor_4, m.motor_5, m.motor_6]
        print("\n◆ 电机低速反馈")
        print(f"  {'电机':>5} {'电压(V)':>9} {'FOC温度(°C)':>11} {'电机温度(°C)':>11} "
              f"{'母线电流(A)':>11} {'使能':>5}")
        for i, mot in enumerate(motors, 1):
            enabled = "✔" if mot.foc_status.driver_enable_status else "✘"
            print(f"  M{i}:   {mot.vol*0.1: 7.1f}  {mot.foc_temp: 10}  {mot.motor_temp: 10}  "
                  f"{mot.bus_current/1e3: 9.2f}  {enabled:>5}")

    # -------------------------------------------------------
    def read_enable_status(self):
        """读取各电机使能状态。"""
        status = self.piper.GetArmEnableStatus()
        print("\n◆ 使能状态")
        for i, en in enumerate(status, 1):
            print(f"  M{i}: {'使能' if en else '失能'}")

    # -------------------------------------------------------
    def read_fw_version(self):
        """读取固件版本。"""
        ver = self.piper.GetPiperFirmwareVersion()
        # 首次读取返回值通常是错误码,需要先发查询
        if isinstance(ver, int) and ver < 0:
            self.piper.SearchPiperFirmwareVersion()
            time.sleep(0.1)
            ver = self.piper.GetPiperFirmwareVersion()
        if isinstance(ver, int) and ver < 0:
            print(f"\n◆ 固件版本: 查询中...")
        else:
            print(f"\n◆ 固件版本: {ver}")

    # -------------------------------------------------------
    def read_fps(self):
        """读取CAN帧率。"""
        fps = self.piper.GetCanFps()
        if fps:
            print(f"\n◆ CAN FPS: {fps:.1f}")
        else:
            print("\n◆ CAN FPS: N/A")

    # -------------------------------------------------------
    def read_all(self):
        """读取全部状态。"""
        self.read_status()
        self.read_joints()
        self.read_end_pose()
        self.read_gripper()
        self.read_motor_high_spd()
        self.read_motor_low_spd()
        self.read_enable_status()
        self.read_fw_version()
        self.read_fps()


def clear_screen():
    sys.stdout.write("\033[2J\033[H")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="读取 Piper 机械臂所有状态信息")
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--loop", "-l", action="store_true", help="循环刷新模式")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="循环刷新间隔(s), 默认 1s")
    parser.add_argument("--mode", "-m", choices=["all", "joint", "pose", "gripper", "motor", "status"],
                        default="all", help="显示模式: all/joint/pose/gripper/motor/status")
    args = parser.parse_args()

    reader = ArmStateReader(args.can)

    try:
        reader.connect()

        if args.loop:
            interval = max(0.1, args.interval)
            print(f"[刷新模式] 间隔 {interval}s, 按 Ctrl+C 退出\n")
            while True:
                clear_screen()
                reader.read_all()
                time.sleep(interval)
        else:
            if args.mode == "all":
                reader.read_all()
            elif args.mode == "joint":
                reader.read_joints()
            elif args.mode == "pose":
                reader.read_end_pose()
            elif args.mode == "gripper":
                reader.read_gripper()
            elif args.mode == "motor":
                reader.read_motor_high_spd()
                reader.read_motor_low_spd()
                reader.read_enable_status()
            elif args.mode == "status":
                reader.read_status()
                reader.read_fw_version()
                reader.read_fps()
            print()

    except (KeyboardInterrupt, EOFError):
        print("\n[退出]")
    except Exception as e:
        print(f"[错误] {e}")
    finally:
        if reader.piper:
            reader.disconnect()


if __name__ == "__main__":
    main()

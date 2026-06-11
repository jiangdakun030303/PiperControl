# Piper 机械臂控制脚本说明

基于 `piper_sdk` (v0.6.1) 的三组控制/监控脚本,通过 CAN 总线与 AgileX Piper 6 轴机械臂通信。

## 环境准备

```bash
conda activate nero
# CAN 接口需先 up
sudo ip link set can0 up type can bitrate 1000000
```

## 脚本概览

| 脚本 | 用途 | 运行方式 |
|------|------|---------|
| `read_arm_state.py` | 读取全部状态 | 一次性或循环刷新 |
| `control_arm.py` | 关节/笛卡尔/夹爪控制 | 交互式或演示模式 |
| `jog_end_pose.py` | 末端位姿微调 | 按键 ±5mm 步进 |

---

## 1. read_arm_state.py — 状态读取

**读取内容:**

| 模块 | 数据 |
|------|------|
| 机械臂状态 | 控制模式、臂状态、运动状态、示教状态、故障码(含每关节限位/通信异常) |
| 关节角度 | J1~J6 角度 (° 和 rad) |
| 末端位姿 | X/Y/Z (m + mm), RX/RY/RZ (°) |
| 夹爪 | 行程(mm)、力矩(Nm)、FOC 状态 |
| 电机高速 | 6 电机 × 转速(rad/s)、电流(A)、位置(rad)、力矩(Nm) |
| 电机低速 | 6 电机 × 电压(V)、温度(°C)、使能状态、FOC 异常标志 |
| 其他 | 固件版本、CAN 帧率 |

**用法:**

```bash
# 一次性打印全部
python read_arm_state.py --can can0

# 持续刷新 (每秒)
python read_arm_state.py --can can0 --loop -i 1

# 只看关节角
python read_arm_state.py --can can0 -m joint
# 可选: joint / pose / gripper / motor / status / all
```

**数据通路:**

```
CAN 总线 ──→ C_PiperInterface_V2 ──→ GetArmJointMsgs().joint_state
              (异步读线程)            GetArmEndPoseMsgs().end_pose
                                     GetArmStatus().arm_status
                                     GetArmGripperMsgs().gripper_state
                                     GetArmHighSpdInfoMsgs().motor_[1-6]
                                     GetArmLowSpdInfoMsgs().motor_[1-6]
```

**单位换算:**

| 反馈量 | 原始单位 | 显示单位 | 换算 |
|--------|---------|---------|------|
| 关节角 | 0.001° | ° | `/ 1000` |
| 末端XYZ | 0.001mm | mm / m | `/ 1000` / `/ 1e6` |
| 末端RPY | 0.001° | ° | `/ 1000` |
| 电机转速 | 0.001 rad/s | rad/s | `/ 1000` |
| 电机电流 | 0.001 A | A | `/ 1000` |
| 电机电压 | 0.1 V | V | `* 0.1` |
| 夹爪行程 | 0.001mm | mm | `/ 1000` |
| 夹爪力矩 | 0.001 Nm | Nm | `/ 1000` |

**固件版本查询:** 首次调用 `GetPiperFirmwareVersion()` 返回错误码(`-0x4AF`),需先 `SearchPiperFirmwareVersion()` 发送查询,等待 ~100ms 后再读取。

---

## 2. control_arm.py — 运动控制

**工作流程:**

```
connect() → enable() → set_mode() → 发送指令 → reset() → disable() → disconnect()
```

**控制模式:**

| move_mode | 名称 | 说明 |
|-----------|------|------|
| 0 | MOVE P | 笛卡尔位姿控制 |
| 1 | MOVE J | 关节角控制 |
| 2 | MOVE L | 直线轨迹 |
| 3 | MOVE C | 圆弧轨迹(三点) |

**指令单位换算:**

| 指令量 | 输入单位 | CAN 单位 | 换算系数 |
|--------|---------|---------|---------|
| 关节角 | rad | 0.001° | `RAD2JOINT = 57295.7795` |
| 末端XYZ | mm | 0.001mm | `MM2POSE = 1000` |
| 末端RPY | ° | 0.001° | `DEG2POSE = 1000` |
| 夹爪开度 | mm | 0.001mm | `MM2GRIPPER = 1000` |
| 夹爪力矩 | Nm | 0.001 Nm | `* 1000` |

**用法:**

```bash
# 交互模式
python control_arm.py --can can0

# 自动演示 (HOME → 夹爪开合 → 关节运动 → HOME)
python control_arm.py --can can0 --demo
```

**交互命令:**

```
j <j1> <j2> <j3> <j4> <j5> <j6>   — 关节角控制 (°)
p <x> <y> <z>                       — 笛卡尔位姿 (mm)
g <开度>                             — 夹爪控制 (mm)
s                                    — 打印状态
home                                 — 回 HOME
stop                                 — 急停
reset                                — 复位
q                                    — 退出
```

**注意事项:**

- **HOME 位姿:** 当前脚本使用全零关节角作为 HOME (`[0, 0, 0, 0, 0, 0]`),J2 下限为 0.0、J3 上限为 0.0,恰在边界。实际 HOME 可能使臂处于奇异位形或不可达,按需修改 `go_home()` 中的目标值。
- **复位流程:** 急停后需 `reset()` (MotionCtrl_1 0x02) 清除错误,然后重新 `enable()` 使能电机。
- **MOVE P 与 MOVE L 的区别:** MOVE P 是点到点运动(路径不限),MOVE L 是直线轨迹。
- **速度参数:** `move_spd_rate_ctrl` 为百分比 0-100,并非绝对速度值。

---

## 3. jog_end_pose.py — 末端位姿微调

**工作原理:**

每按一次键(无需回车),读取当前末端位姿 → 叠加增量 → 发送 MOVE P 指令。terminal raw 模式,按键即时响应。

```
当前位姿 [X, Y, Z, RX, RY, RZ] → ±增量 → 新目标位姿 → EndPoseCtrl()
```

**键位:**

```
位置:   q / e  →  X轴 ±5mm
       a / d  →  Y轴 ±5mm
       w / s  →  Z轴 ±5mm

姿态:   u / j  →  RX ±1°
       i / k  →  RY ±1°
       o / l  →  RZ ±1°

p      →  打印当前位姿
r      →  复位
Ctrl+C →  退出 (保持使能, 不断开电机)
```

**用法:**

```bash
python jog_end_pose.py --can can0 --step 5 --rot-step 1
# --step     位置步进量 (mm), 默认 5
# --rot-step 姿态步进量 (°),  默认 1
```

**设计要点:**

- **无回车响应:** terminal raw 模式下按键即时生效,无需 Enter 确认
- **退出不掉使能:** Ctrl+C 只断开 CAN 连接,不 reset 不失能,机械臂保持当前姿态
- **绝对位姿控制:** 每次按键下发"当前值 + 增量"的绝对目标,连续按键自动从当前位置叠加

---

## CAN 通信架构

```
USB-to-CAN 适配器 (gs_usb / slcan)
        │
        ▼
C_PiperInterface_V2(can_name)
  ├─ C_STD_CAN         — 底层 CAN Socket 封装
  ├─ C_PiperParserV2   — CAN 帧解析
  ├─ 读线程 (daemon)   — 持续接收 CAN 帧,更新内部缓存
  └─ 写方法            — JointCtrl / EndPoseCtrl / GripperCtrl 等
```

**关键技术点:**

- **线程安全:** 所有 getter 方法用 `threading.Lock` 保护,读线程写缓存时不会被打断
- **帧率:** 正常 ~3040 FPS (单条臂,完整反馈周期约 0.3ms)
- **CAN ID 映射:**
  - 控制指令: 0x150-0x15F, 0x470-0x47F
  - 关节反馈: 0x2A5-0x2A7
  - 位姿反馈: 0x2A2-0x2A4
  - 状态反馈: 0x2A1
  - 电机高速: 0x251-0x256
  - 电机低速: 0x261-0x266
  - 夹爪反馈: 0x2A8

**故障码解析:** `err_code` 16 位:
- 低字节 bit[0-5]: J1-J6 通信异常
- 高字节 bit[0-5]: J1-J6 角度超限位

# PiperControl

基于 [piper_sdk](https://github.com/agilexrobotics/piper_sdk) 的 AgileX Piper 6 轴机械臂控制脚本集，通过 CAN 总线实现关节/笛卡尔/夹爪控制、状态读取、轨迹步进、手眼标定等功能。

## 环境准备

```bash
# 安装 SDK
pip3 install piper_sdk python-can numpy

# 激活 CAN 接口 (需 root 权限)
sudo ip link set can0 up type can bitrate 1000000
```

## 脚本一览

| 脚本 | 功能 | 使用方式 |
|------|------|---------|
| `read_arm_state.py` | 读取机械臂全部状态 | 持续刷新 / 单次打印 |
| `control_arm.py` | 关节/笛卡尔/夹爪交互控制 | 命令行交互 / 自动演示 |
| `jog_end_pose.py` | 末端位姿微调 | 无回车按键实时步进 |
| `waypoint_control.py` | 预设路径点依次运动 | JSON 路径点文件 |
| `trajectory_step.py` | 轨迹步进控制 | 从 .npz 加载，空格键步进 |
| `hand_eye_calibration.py` | 手眼标定 | 采集 → 标定 → 评估 |
| `multi_cam_calibration.py` | 多相机标定 | 棋盘格标定 |

详细说明见 [SCRIPTS_GUIDE.md](./SCRIPTS_GUIDE.md)。

---

## 1. 状态读取 — `read_arm_state.py`

```bash
# 一次性打印全部状态
python read_arm_state.py --can can0

# 持续刷新 (每秒)
python read_arm_state.py --can can0 --loop -i 1

# 只看关节角度
python read_arm_state.py --can can0 -m joint
# 可选: joint / pose / gripper / motor / status / all
```

输出内容包括关节角度 (°/rad)、末端位姿 (m/mm, °)、夹爪行程/力矩、电机高速/低速数据、固件版本、CAN 帧率等。

---

## 2. 运动控制 — `control_arm.py`

```bash
# 交互模式
python control_arm.py --can can0

# 自动演示 (HOME → 夹爪 → 关节运动 → HOME)
python control_arm.py --can can0 --demo
```

交互命令：

```
j <j1>..<j6>  — 关节角控制 (°)
p <x> <y> <z>  — 笛卡尔位姿 (mm)
g <开度>       — 夹爪控制 (mm)
s              — 打印当前状态
home           — 回 HOME 位姿
stop           — 急停
reset          — 复位
q              — 退出
```

---

## 3. 末端微调 — `jog_end_pose.py`

```bash
python jog_end_pose.py --can can0 --step 5 --rot-step 1
```

无需回车，按键即响应：

```
位置:  q/e → X±5mm   a/d → Y±5mm   w/s → Z±5mm
姿态:  u/j → RX±1°   i/k → RY±1°   o/l → RZ±1°
p → 打印位姿   r → 复位   Ctrl+C → 退出
```

退出不断使能，机械臂保持当前姿态。

---

## 4. 路径点控制 — `waypoint_control.py`

```bash
python waypoint_control.py --can can0 -f waypoints_target.json
```

路径点文件格式 (`waypoints_example.json`):

```json
[
  {"type": "joint", "target": [0, 0.5, -0.3, 0, 0.2, 0], "gripper": 20},
  {"type": "pose",  "target": [300, 0, 200, 0, 0, 0], "gripper": 30}
]
```

- `type: "joint"` → 关节角控制 (rad)
- `type: "pose"` → 笛卡尔位姿控制 (mm, °)
- `gripper` → 夹爪开度 (mm)，可选

---

## 5. 轨迹步进 — `trajectory_step.py`

```bash
python trajectory_step.py --can can0 -f trajectory_base_20260610_070851.npz
```

从 `.npz` 文件加载路径点，自动 HOME → 第一个目标点，之后每按一次空格走一步，按 `q` 退出。

---

## 6. 手眼标定 — `hand_eye_calibration.py`

```bash
python hand_eye_calibration.py --can can0
```

交互式手眼标定流程：采集标定板位姿 → 计算机器人基座到相机的变换矩阵 → 评估标定误差。

---

## 7. 多相机标定 — `multi_cam_calibration.py`

```bash
python multi_cam_calibration.py
```

使用棋盘格图像进行多相机内外参标定，数据存放于 `multi_cam_calib/` 目录。

---

## 注意事项

- CAN 接口必须先 `ip link set up` 并设置正确波特率 `1000000`
- 机械臂需处于从机模式才能读取反馈数据
- 关节限位 (rad): J1 ±2.62, J2 0~3.14, J3 -2.97~0, J4 ±1.75, J5 ±1.22, J6 ±2.09
- HOME 位姿使用全零关节角，J2/J3 恰在边界，按需修改
- 急停后需执行 `reset()` → `enable()` 流程恢复

## 相关链接

- [piper_sdk 官方仓库](https://github.com/agilexrobotics/piper_sdk)
- [Piper SDK UI](https://github.com/agilexrobotics/Piper_sdk_ui)
- [Discord 社区](https://discord.gg/wrKYTxwDBd)

#!/usr/bin/env python3
"""手部轨迹 → 平滑关节空间轨迹 转换脚本.

管线:
  原始 npz → 中值滤波 → Douglas-Peucker 精简 → 最小间距约束
  → Pinocchio IK → 关节空间三次样条 → 保存 npz
"""

import os
import sys
import math
import argparse
import numpy as np

_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from scipy.interpolate import CubicSpline
from scipy.signal import medfilt

from piper_sdk.kinematics.piper_ik_pinocchio import C_PiperIKPinocchio


# ======================================================================
#  ① 中值滤波
# ======================================================================

def median_filter(pts: np.ndarray, kernel: int = 5) -> np.ndarray:
    """对轨迹各维度独立做中值滤波, 消除高频抖动, 保留转角."""
    if len(pts) < kernel:
        return pts.copy()
    result = np.zeros_like(pts)
    for d in range(pts.shape[1]):
        result[:, d] = medfilt(pts[:, d], kernel)
    return result


# ======================================================================
#  ② Douglas-Peucker
# ======================================================================

def _dp_recurse(pts, start, end, epsilon):
    """递归 Douglas-Peucker."""
    if end - start <= 1:
        return [start]
    dmax = 0.0
    imax = start
    seg = pts[end] - pts[start]
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-9:
        return [start]

    for i in range(start + 1, end):
        cross = np.linalg.norm(np.cross(pts[i] - pts[start], seg)) / seg_len
        if cross > dmax:
            dmax = cross
            imax = i

    if dmax > epsilon:
        left = _dp_recurse(pts, start, imax, epsilon)
        right = _dp_recurse(pts, imax, end, epsilon)
        return left + right
    else:
        return [start]


def douglas_peucker(pts: np.ndarray, epsilon: float = 0.003) -> np.ndarray:
    """Douglas-Peucker 折线简化.

    Args:
        pts: (N, 3) 轨迹点 (m)
        epsilon: 允许的最大垂直距离 (m), 默认 3mm

    Returns:
        简化后的点
    """
    if len(pts) < 3:
        return pts.copy()
    indices = _dp_recurse(pts, 0, len(pts) - 1, epsilon)
    indices.append(len(pts) - 1)
    return pts[np.array(indices)]


# ======================================================================
#  ③ 最小弧长间距
# ======================================================================

def min_arc_spacing(pts: np.ndarray, min_dist: float = 0.005) -> np.ndarray:
    """合并间距过近的相邻点.

    累积弧长, 每隔 min_dist 保留一个点.
    """
    if len(pts) < 2:
        return pts.copy()

    kept = [pts[0]]
    accum = 0.0
    for i in range(1, len(pts)):
        accum += np.linalg.norm(pts[i] - pts[i - 1])
        if accum >= min_dist:
            kept.append(pts[i])
            accum = 0.0
    # 总是保留最后一个点
    if not np.array_equal(kept[-1], pts[-1]):
        kept.append(pts[-1])
    return np.array(kept)


# ======================================================================
#  ④ Pinocchio IK
# ======================================================================

def solve_ik_sequence(keypoints: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """对关键点序列依次做 IK, 返回 (M, 6) 关节角 (rad), 碰撞点索引, 无解点索引.

    每个点的初始值用上一个点的解, 保证连续性.
    """
    ik = C_PiperIKPinocchio()
    q_seq = np.zeros((len(keypoints), 6))
    collisions = []
    no_solutions = []

    # 第一个点从 HOME 出发
    q_prev = [0.0] * 6
    for i, p in enumerate(keypoints):
        sol = ik.solve_position(p[0], p[1], p[2], q0=q_prev)
        if sol is None:
            print(f"  [警告] 第 {i} 个关键点 ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})m IK 无解, 沿用上一解")
            sol = q_prev
            no_solutions.append(i)
        else:
            # 检查碰撞
            if ik.check_collision(sol):
                print(f"  [碰撞] 第 {i} 个关键点 ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})m 自碰撞, 尝试重解...")
                # 尝试轻微偏移目标位置
                offset = 0.002
                resolved = False
                for dx, dy, dz in [(0, 0, offset), (0, offset, 0), (offset, 0, 0),
                                   (0, 0, -offset), (0, -offset, 0), (-offset, 0, 0)]:
                    sol2 = ik.solve_position(p[0] + dx, p[1] + dy, p[2] + dz, q0=q_prev)
                    if sol2 is not None and not ik.check_collision(sol2):
                        sol = sol2
                        resolved = True
                        print(f"    偏移 ({dx*1000:.0f},{dy*1000:.0f},{dz*1000:.0f})mm 后找到无碰解")
                        break
                if not resolved:
                    collisions.append(i)
                    print(f"    无法找到无碰解, 保留碰撞位型")
        q_seq[i] = sol
        q_prev = sol
    return q_seq, collisions, no_solutions


# ======================================================================
#  ⑤ 关节空间三次样条
# ======================================================================

def fit_joint_spline(q_key: np.ndarray, key_t: np.ndarray,
                     sample_dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """在关节空间拟合 C² 连续三次样条, 等间隔采样.

    Args:
        q_key:   (M, 6) 关键关节角 (rad)
        key_t:   (M,)   关键点时间戳 (从 0 开始, 秒)
        sample_dt: 采样间隔 (秒)

    Returns:
        q_sampled: (N, 6) 采样关节角 (rad)
        t_sampled: (N,)   采样时间戳 (秒)
    """
    t_sampled = np.arange(0, key_t[-1], sample_dt)
    q_sampled = np.zeros((len(t_sampled), 6))

    for d in range(6):
        cs = CubicSpline(key_t, q_key[:, d], bc_type='natural')
        q_sampled[:, d] = cs(t_sampled)

    return q_sampled, t_sampled


# ======================================================================
#  主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="手部轨迹 → 关节空间平滑轨迹")
    parser.add_argument("input", help="输入 .npz 轨迹文件")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 .npz 文件 (默认 <input>_processed.npz)")
    parser.add_argument("--median-kernel", type=int, default=3,
                        help="中值滤波窗口 (默认 3)")
    parser.add_argument("--dp-epsilon", type=float, default=0.002,
                        help="Douglas-Peucker 垂直误差容限 (m, 默认 0.002)")
    parser.add_argument("--min-spacing", type=float, default=0.005,
                        help="关键点最小弧长间距 (m, 默认 0.005)")
    parser.add_argument("--sample-dt", type=float, default=0.02,
                        help="样条采样间隔 (s, 默认 0.02 = 50Hz)")
    parser.add_argument("--speed", type=float, default=0.05,
                        help="末端运动速度 (m/s, 默认 0.05)")
    args = parser.parse_args()

    # ---- 加载 ----
    data = np.load(args.input, allow_pickle=True)
    pts_raw = data["positions_base"]
    print(f"[加载] {args.input}  ({len(pts_raw)} 个点)")
    print(f"  范围: X [{pts_raw[:,0].min():.4f}, {pts_raw[:,0].max():.4f}]  "
          f"Y [{pts_raw[:,1].min():.4f}, {pts_raw[:,1].max():.4f}]  "
          f"Z [{pts_raw[:,2].min():.4f}, {pts_raw[:,2].max():.4f}]m")

    # ---- ① 中值滤波 ----
    pts_med = median_filter(pts_raw, args.median_kernel)
    print(f"[中值滤波] kernel={args.median_kernel} → {len(pts_med)} 点")

    # ---- ② Douglas-Peucker ----
    pts_dp = douglas_peucker(pts_med, args.dp_epsilon)
    print(f"[DP精简] epsilon={args.dp_epsilon*1000:.0f}mm → {len(pts_dp)} 个关键点")

    # ---- ③ 最小间距 ----
    pts_spaced = min_arc_spacing(pts_dp, args.min_spacing)
    print(f"[最小间距] {args.min_spacing*1000:.0f}mm → {len(pts_spaced)} 个关键点")

    # ---- ③b 极端点保真: DP 可能丢掉拐角最远点, 强制补回 ----
    for dim, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
        raw_min = pts_med[:, dim].min()
        raw_max = pts_med[:, dim].max()
        if pts_spaced[:, dim].min() > raw_min + 0.002:
            idx = np.argmin(pts_med[:, dim])
            print(f"  [保真] 补回{name}极小值点 {raw_min:.4f}m")
            pts_spaced = np.insert(pts_spaced, np.searchsorted(
                np.linalg.norm(pts_spaced - pts_spaced[0], axis=1),
                np.linalg.norm(pts_med[idx] - pts_spaced[0])), pts_med[idx], axis=0)
        if pts_spaced[:, dim].max() < raw_max - 0.002:
            idx = np.argmax(pts_med[:, dim])
            print(f"  [保真] 补回{name}极大值点 {raw_max:.4f}m")
            pts_spaced = np.insert(pts_spaced, np.searchsorted(
                np.linalg.norm(pts_spaced - pts_spaced[0], axis=1),
                np.linalg.norm(pts_med[idx] - pts_spaced[0])), pts_med[idx], axis=0)

    # ---- 压缩比 ----
    ratio = len(pts_spaced) / len(pts_raw) * 100
    print(f"[压缩] {len(pts_raw)} → {len(pts_spaced)} 点 (保留 {ratio:.0f}%)")

    # ---- 保真报告 ----
    print(f"[保真] 处理后范围:")
    print(f"  X [{pts_spaced[:,0].min():.4f}, {pts_spaced[:,0].max():.4f}]  "
          f"(原始 [{pts_raw[:,0].min():.4f}, {pts_raw[:,0].max():.4f}])")
    print(f"  Y [{pts_spaced[:,1].min():.4f}, {pts_spaced[:,1].max():.4f}]  "
          f"(原始 [{pts_raw[:,1].min():.4f}, {pts_raw[:,1].max():.4f}])")
    print(f"  Z [{pts_spaced[:,2].min():.4f}, {pts_spaced[:,2].max():.4f}]  "
          f"(原始 [{pts_raw[:,2].min():.4f}, {pts_raw[:,2].max():.4f}])")

    # ---- ④ IK ----
    print(f"[IK] 对 {len(pts_spaced)} 个关键点求解 (Pinocchio)...")
    q_key, collisions, no_solutions = solve_ik_sequence(pts_spaced)
    q_key_deg = np.degrees(q_key)
    print(f"  IK 完成, 关节范围:")
    for i in range(6):
        print(f"    J{i+1}: [{q_key_deg[:,i].min():.1f}, {q_key_deg[:,i].max():.1f}]°")
    if collisions:
        print(f"  [碰撞] {len(collisions)} 个关键点自碰撞: 索引 {collisions}")
    if no_solutions:
        print(f"  [无解] {len(no_solutions)} 个关键点 IK 无解: 索引 {no_solutions}")

    # ---- ⑤ 时间分配 (按弧长 + 速度) ----
    dists = np.linalg.norm(np.diff(pts_spaced, axis=0), axis=1)
    times = np.cumsum(np.concatenate([[0], dists / args.speed]))
    total_time = times[-1]
    print(f"[时间] 总弧长={dists.sum():.3f}m, 速度={args.speed}m/s → {total_time:.1f}s")

    # ---- ⑥ 样条插值 ----
    q_sampled, t_sampled = fit_joint_spline(q_key, times, args.sample_dt)
    print(f"[样条] {len(times)} 关键点 → {len(t_sampled)} 采样点 ({args.sample_dt*1000:.0f}ms)")

    # 样条后碰撞扫描 + 迭代修复
    ik = C_PiperIKPinocchio()
    for repair_round in range(3):
        spline_collisions = []
        for i in range(0, len(q_sampled), 5):
            if ik.check_collision(q_sampled[i].tolist()):
                spline_collisions.append(i)
        if not spline_collisions:
            print(f"  [碰撞] 样条轨迹无碰撞")
            break

        print(f"  [碰撞] 第{repair_round+1}轮: {len(spline_collisions)} 帧碰撞, 修复中...")

        # 找到碰撞帧对应的关键点索引
        t_key = np.linspace(0, t_sampled[-1], len(q_key))
        bad_key_indices = set()
        for ci in spline_collisions:
            t_c = t_sampled[ci]
            nearest = int(np.argmin(np.abs(t_key - t_c)))
            for offset in range(-1, 2):
                ni = nearest + offset
                if 0 <= ni < len(q_key):
                    bad_key_indices.add(ni)

        # 对每个问题关键点尝试替代 IK 解
        for ki in sorted(bad_key_indices):
            p = pts_spaced[ki]
            q_cur = q_key[ki]
            if not ik.check_collision(q_cur.tolist()):
                continue  # 实际没碰撞, 跳过

            # 策略1: 从不同初始值重解
            resolved = False
            for alt_q0 in [
                None,  # 从零位
                [0.5, 0.5, -0.5, 0.5, 0.0, 0.0],
                [-0.5, 0.3, -1.0, -0.5, 0.0, 0.5],
            ]:
                q0 = alt_q0 if alt_q0 else [0.0]*6
                sol = ik.solve_position(p[0], p[1], p[2], q0=q0)
                if sol is not None and not ik.check_collision(sol):
                    q_key[ki] = sol
                    resolved = True
                    break

            # 策略2: 微调目标位置
            if not resolved:
                for dx, dy, dz in [(0.005,0,0),(-0.005,0,0),(0,0.005,0),(0,-0.005,0),(0,0,0.005),(0,0,-0.005)]:
                    sol = ik.solve_position(p[0]+dx, p[1]+dy, p[2]+dz, q0=q_key[ki-1].tolist() if ki>0 else [0.0]*6)
                    if sol is not None and not ik.check_collision(sol):
                        q_key[ki] = sol
                        pts_spaced[ki] = p + np.array([dx, dy, dz])
                        resolved = True
                        break

            if not resolved:
                # 策略3: 用前后无碰关键点插值
                prev_safe = ki - 1
                while prev_safe >= 0 and ik.check_collision(q_key[prev_safe].tolist()):
                    prev_safe -= 1
                next_safe = ki + 1
                while next_safe < len(q_key) and ik.check_collision(q_key[next_safe].tolist()):
                    next_safe += 1
                if prev_safe >= 0 and next_safe < len(q_key):
                    alpha = (ki - prev_safe) / (next_safe - prev_safe)
                    q_key[ki] = q_key[prev_safe] * (1-alpha) + q_key[next_safe] * alpha
                    print(f"    关键点{ki}: 用安全点{prev_safe}-{next_safe}线性插值替代")

        # 更新关节空间样条
        q_sampled, t_sampled = fit_joint_spline(q_key, times, args.sample_dt)

    else:
        print(f"  [碰撞] 修复后仍有 {len(spline_collisions)} 帧碰撞, 请手动检查")
        # 仍记录碰撞帧
    spline_collisions_final = spline_collisions

    # ---- ⑦ 保存 ----
    out_path = args.output or args.input.replace('.npz', '_processed.npz')
    np.savez(
        out_path,
        positions_base=pts_spaced,
        positions_raw=pts_raw,
        q_key=q_key,
        q_key_deg=q_key_deg,
        q_sampled=q_sampled,
        q_sampled_deg=np.degrees(q_sampled),
        t_sampled=t_sampled,
        sample_dt=args.sample_dt,
        speed=args.speed,
        total_time=total_time,
        n_keypoints=len(pts_spaced),
        n_samples=len(t_sampled),
        median_kernel=args.median_kernel,
        dp_epsilon=args.dp_epsilon,
        min_spacing=args.min_spacing,
        collisions_key=np.array(collisions),
        collisions_spline=np.array(spline_collisions_final),
        no_solutions=np.array(no_solutions),
    )
    print(f"[保存] {out_path}")
    print(f"  keys: positions_base, q_key, q_sampled, t_sampled, ...")


if __name__ == "__main__":
    main()

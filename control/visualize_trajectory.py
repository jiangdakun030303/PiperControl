#!/usr/bin/env python3
"""轨迹可视化 —— 3D路径 + 关节曲线 + 统计面板。"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    parser = argparse.ArgumentParser(description="轨迹可视化")
    parser.add_argument("processed_npz", help="convert_trajectory.py 输出的 _processed.npz")
    parser.add_argument("-o", "--output", default=None,
                        help="保存到图片文件 (不指定则弹出窗口)")
    args = parser.parse_args()

    if args.output:
        matplotlib.use('Agg')

    data = np.load(args.processed_npz, allow_pickle=True)

    pts_raw = data.get("positions_raw")
    pts_key = data["positions_base"]
    q_key = data["q_key"]
    q_sampled = data["q_sampled"]
    t_sampled = data["t_sampled"]
    has_raw = pts_raw is not None

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[2.5, 1])

    # ---- 左上: 3D 任务空间路径 ----
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    if has_raw:
        ax3d.plot(pts_raw[:, 0], pts_raw[:, 1], pts_raw[:, 2],
                  'lightgray', linewidth=0.8, alpha=0.6, label=f'raw ({len(pts_raw)} pts)')
    ax3d.plot(pts_key[:, 0], pts_key[:, 1], pts_key[:, 2],
              'o-', color='steelblue', markersize=4, linewidth=1.5, label=f'key ({len(pts_key)} pts)')
    ax3d.scatter(*pts_key[0], c='green', s=80, zorder=5, label='start')
    ax3d.scatter(*pts_key[-1], c='red', s=80, zorder=5, label='end')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('Task Space Path', fontweight='bold')
    ax3d.legend(fontsize=7, loc='upper left')

    # ---- 右上: 统计面板 ----
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    text = (
        f"raw points:     {len(pts_raw) if has_raw else 'N/A'}\n"
        f"key points:     {len(pts_key)}\n"
        f"spline points:  {len(q_sampled)}\n"
        f"sample dt:      {t_sampled[1]-t_sampled[0]:.0f} ms\n"
        f"duration:       {t_sampled[-1]:.1f} s\n"
        f"speed:          {data.get('speed', 'N/A')} m/s\n"
        f"dp epsilon:     {data.get('dp_epsilon', 'N/A')}\n"
        f"min spacing:    {data.get('min_spacing', 'N/A')}\n"
        f"median kernel:  {data.get('median_kernel', 'N/A')}"
    )
    ax_info.text(0.05, 0.95, text, transform=ax_info.transAxes, fontsize=9,
                 family='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax_info.set_title('Statistics', fontweight='bold')

    # ---- 下行: 关节空间 3×2 ----
    q_key_deg = np.degrees(q_key)
    q_sampled_deg = np.degrees(q_sampled)
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    colors = plt.cm.tab10.colors
    t_key = np.linspace(0, t_sampled[-1], len(q_key))

    gs_joint = GridSpec(2, 3, figure=fig)
    gs_joint.update(top=0.47, bottom=0.05, left=0.05, right=0.97, hspace=0.4, wspace=0.25)

    for j in range(6):
        row, col = j // 3, j % 3
        ax_j = fig.add_subplot(gs_joint[row, col])
        ax_j.plot(t_sampled, q_sampled_deg[:, j], color=colors[j], linewidth=1.2, label='spline')
        ax_j.plot(t_key, q_key_deg[:, j], 'o', color=colors[j], markersize=3, alpha=0.6, label='key')
        ax_j.set_ylabel(f'{joint_names[j]} (deg)', fontsize=9)
        ax_j.set_xlabel('t (s)', fontsize=9)
        ax_j.set_title(joint_names[j], fontweight='bold', fontsize=10)
        ax_j.grid(True, alpha=0.3)
        if j == 0:
            ax_j.legend(fontsize=7)

    fig.suptitle(f'Trajectory: {args.processed_npz.split("/")[-1]}', fontsize=12, fontweight='bold')

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f'保存到: {args.output}')
    else:
        plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def draw_frame(ax, R, t, scale=0.15, alpha=1.0, lw=2):
    t = np.asarray(t).ravel()
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        end = t + R[:, i] * scale
        arrow = Arrow3D([t[0], end[0]], [t[1], end[1]], [t[2], end[2]],
                        mutation_scale=10, lw=lw, arrowstyle='-|>',
                        color=color, alpha=alpha)
        ax.add_artist(arrow)


def euler_zyx(R):
    euler = np.array([np.arctan2(R[2, 1], R[2, 2]),
                      np.arcsin(-R[2, 0]),
                      np.arctan2(R[1, 0], R[0, 0])])
    return np.degrees(euler)


def autolim(ax, pts):
    pts = np.asarray(pts)
    max_range = np.ptp(pts, axis=0).max() / 2 + 0.1
    mid = pts.mean(axis=0)
    for i, getter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        getter(mid[i] - max_range, mid[i] + max_range)


def main():
    # --- user's hardcoded pose ---
    R_user = np.array([[-0.88504736,  0.40697294, -0.22597387],
                       [ 0.46529101,  0.78799939, -0.40318885],
                       [ 0.01398032, -0.46198484, -0.88677763]])
    t_user = np.array([0.4268052, 0.44442098, 0.57029641])

    # --- cam1_to_base from calibration ---
    d1 = np.load('/home/jiang/my_ws/piper_sdk/control/multi_cam_calib/cam1_to_base.npz')
    R_cam1 = d1['R']; t_cam1 = d1['t']; samples1 = d1['samples'][:, :3, 3]

    # --- cam2_to_base from calibration ---
    d2 = np.load('/home/jiang/my_ws/piper_sdk/control/multi_cam_calib/cam2_to_base.npz')
    R_cam2 = d2['R']; t_cam2 = d2['t']; samples2 = d2['samples'][:, :3, 3]

    # --- trajectory ---
    traj = np.load('/home/jiang/my_ws/track_hand/outputs/trajectory_base_20260611_053108.npz')
    pos_base = traj['positions_base']       # (N, 3)
    T_cam1_base = traj['T_cam1_base']       # (4, 4)
    T_cam2_base = traj['T_cam2_base']       # (4, 4)
    # extract R/t from trajectory T matrices
    R_cam1_traj = T_cam1_base[:3, :3]; t_cam1_traj = T_cam1_base[:3, 3]
    R_cam2_traj = T_cam2_base[:3, :3]; t_cam2_traj = T_cam2_base[:3, 3]

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # ---- World origin ----
    draw_frame(ax, np.eye(3), [0, 0, 0], scale=0.2, alpha=0.3, lw=1.5)

    # ---- User pose (red) ----
    draw_frame(ax, R_user, t_user, scale=0.12, alpha=1.0, lw=2)
    ax.scatter(*t_user, c='red', s=60, depthshade=False, zorder=5)
    ax.text(t_user[0], t_user[1], t_user[2]+0.03, 'user pose', color='red', fontsize=9)

    # ---- cam1 -> base (blue) ----
    draw_frame(ax, R_cam1, t_cam1, scale=0.12, alpha=1.0, lw=2)
    ax.scatter(*t_cam1, c='blue', s=60, depthshade=False, zorder=5)
    ax.text(t_cam1[0], t_cam1[1], t_cam1[2]+0.03, 'cam1→base', color='blue', fontsize=9)
    ax.scatter(samples1[:, 0], samples1[:, 1], samples1[:, 2],
               c='dodgerblue', s=6, alpha=0.4, depthshade=False)

    # ---- cam2 -> base (green) ----
    draw_frame(ax, R_cam2, t_cam2, scale=0.12, alpha=1.0, lw=2)
    ax.scatter(*t_cam2, c='green', s=60, depthshade=False, zorder=5)
    ax.text(t_cam2[0], t_cam2[1], t_cam2[2]+0.03, 'cam2→base', color='green', fontsize=9)
    ax.scatter(samples2[:, 0], samples2[:, 1], samples2[:, 2],
               c='limegreen', s=6, alpha=0.4, depthshade=False)

    # ---- Cam frames from trajectory (dashed, to verify match) ----
    draw_frame(ax, R_cam1_traj, t_cam1_traj, scale=0.10, alpha=0.5, lw=1)
    draw_frame(ax, R_cam2_traj, t_cam2_traj, scale=0.10, alpha=0.5, lw=1)

    # ---- Trajectory path (orange gradient) ----
    pts = pos_base
    n = len(pts)
    for i in range(n - 1):
        frac = i / (n - 1)
        ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], pts[i:i+2, 2],
                color=plt.cm.Oranges(0.3 + 0.7*frac), lw=1.5, alpha=0.8)
    # start / end markers
    ax.scatter(*pts[0],  c='green', s=80, marker='o', depthshade=False, zorder=9,
               label=f'start [{pts[0,0]:.3f}, {pts[0,1]:.3f}, {pts[0,2]:.3f}]')
    ax.scatter(*pts[-1], c='red',   s=80, marker='X', depthshade=False, zorder=9,
               label=f'end   [{pts[-1,0]:.3f}, {pts[-1,1]:.3f}, {pts[-1,2]:.3f}]')

    # ---- Connect origins to camera positions ----
    for t, c in [(t_cam1, 'blue'), (t_cam2, 'green'), (t_user, 'red')]:
        ax.plot([0, t[0]], [0, t[1]], [0, t[2]], '--', color=c, alpha=0.2, lw=1)

    # ---- Print info ----
    for label, R, t in [('user pose  ', R_user, t_user),
                         ('cam1→base ', R_cam1, t_cam1),
                         ('cam2→base ', R_cam2, t_cam2)]:
        eu = euler_zyx(R)
        print(f"{label} t={np.round(t, 4)}  roll={eu[0]:.1f}° pitch={eu[1]:.1f}° yaw={eu[2]:.1f}°")
    print(f"Trajectory: {n} points, span X:[{pts[:,0].min():.4f}, {pts[:,0].max():.4f}] "
          f"Y:[{pts[:,1].min():.4f}, {pts[:,1].max():.4f}] Z:[{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]")

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Multi-Camera Pose + Trajectory ({n} waypoints)\n'
                 f'Trajectory: {n}pts  '
                 f'cam1 t=[{t_cam1[0]:.4f}, {t_cam1[1]:.4f}, {t_cam1[2]:.4f}]  '
                 f'cam2 t=[{t_cam2[0]:.4f}, {t_cam2[1]:.4f}, {t_cam2[2]:.4f}]')

    all_pts = np.vstack([[0, 0, 0], t_user, t_cam1, t_cam2, pts, samples1, samples2])
    autolim(ax, all_pts)
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    out = '/home/jiang/my_ws/piper_sdk/control/pose_visualization.png'
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == '__main__':
    main()

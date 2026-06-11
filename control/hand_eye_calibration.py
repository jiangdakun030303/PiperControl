#!/usr/bin/env python3
"""
Eye-in-hand hand-eye calibration for Piper robotic arm with drag teaching.

The camera is rigidly mounted on the robot's end-effector (eye-in-hand).
A checkerboard is placed at a fixed position in the workspace.
Drag the arm to different poses; at each pose the script captures:
  - end-effector pose in robot base frame (from forward kinematics via CAN)
  - checkerboard pose in camera frame (via solvePnP)

After collecting enough samples (>= 5 recommended), solves AX = XB to find
the camera-to-end-effector rigid transformation.

Requirements: pip install opencv-python numpy
"""

import numpy as np
import cv2
import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from piper_sdk import C_PiperInterface_V2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rpy_to_matrix(xyz, rpy):
    """Position + RPY Euler angles (rad) -> 4x4 homogeneous matrix.

    R = Rz(yaw) * Ry(pitch) * Rx(roll)  (extrinsic xyz convention).
    """
    xyz = np.asarray(xyz)
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def _matrix_to_rpy(T):
    """4x4 homogeneous matrix -> (xyz_m, rpy_rad) using xyz extrinsic convention."""
    R = T[:3, :3]
    if abs(R[2, 0]) > 0.99999:
        pitch = -np.pi / 2 if R[2, 0] < 0 else np.pi / 2
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        pitch = -np.arcsin(R[2, 0])
        cp = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    return T[:3, 3].copy(), np.array([roll, pitch, yaw])


def _solve_pnp(obj_points, img_points, camera_matrix, dist_coeffs):
    """PnP: return 4x4 target-in-camera transform, or None on failure."""
    ok, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _solve_pnp_with_rpe(obj_points, img_points, camera_matrix, dist_coeffs):
    """PnP + reprojection error. Returns (T_4x4, rpe_px) or (None, 0)."""
    ok, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, 0.0
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    rpe = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_points.reshape(-1, 2), axis=1))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, rpe


# ---------------------------------------------------------------------------
# load camera intrinsics (npz / OpenCV yaml)
# ---------------------------------------------------------------------------

def load_intrinsics(path):
    """Load camera intrinsics from .npz or OpenCV-format .yaml/.yml.

    Returns (camera_matrix_3x3, dist_coeffs_1xN) or (None, None).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        d = np.load(path)
        return d['camera_matrix'], d['dist_coeffs']

    if ext in (".yaml", ".yml"):
        return _load_intrinsics_yaml(path)

    print(f"[ERROR] Unsupported intrinsics format: {ext}")
    return None, None


def _load_intrinsics_yaml(path):
    """Parse an OpenCV-format calibration YAML without PyYAML.

    Looks for ``K:`` and ``dist:`` keys inside the ``intrinsics:`` block.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    def _parse_nested_list(start_idx, key_indent):
        """Parse ``- - val / - val`` rows until the next non-list key."""
        rows = []
        current_row = []
        for i in range(start_idx, len(lines)):
            s = lines[i].rstrip()
            stripped = s.lstrip()
            indent = len(s) - len(stripped)

            if stripped == '' or stripped.startswith('#'):
                continue

            if indent == key_indent:
                if stripped.startswith('- -'):
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                    current_row.append(float(stripped[4:].strip()))
                else:
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                    break
            elif indent > key_indent:
                if stripped.startswith('- '):
                    current_row.append(float(stripped[2:].strip()))
            else:
                if current_row:
                    rows.append(current_row)
                    current_row = []
                break

        if current_row:
            rows.append(current_row)
        return np.array(rows, dtype=np.float64) if rows else None

    K = None
    dist = None

    for i, line in enumerate(lines):
        s = line.rstrip()
        indent = len(s) - len(s.lstrip())
        stripped = s.lstrip()

        if stripped == 'K:' and indent >= 2:
            K = _parse_nested_list(i + 1, indent)
        elif stripped == 'dist:' and indent >= 2:
            dist = _parse_nested_list(i + 1, indent)

    if dist is not None and dist.ndim == 1:
        dist = dist.reshape(1, -1)

    if K is None:
        print("[ERROR] Could not find K: in YAML intrinsics block.")
        return None, None

    return K.astype(np.float64), (dist.astype(np.float64) if dist is not None
                                   else np.zeros((1, 5), dtype=np.float64))


# ---------------------------------------------------------------------------
# camera intrinsic calibration (interactive)
# ---------------------------------------------------------------------------

def calibrate_camera_interactive(chessboard_size, square_size,
                                  camera_id=0, save_path="camera_intrinsics.npz",
                                  fisheye=True):
    """Interactive monocular camera calibration with a checkerboard.

    Show the board in different poses, press SPACE to capture, 'q' to finish.
    Uses OpenCV fisheye model (set fisheye=False for pinhole).
    """
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points, img_points = [], []

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print(f"  Camera Calibration ({'FISHEYE' if fisheye else 'PINHOLE'} model)")
    print("=" * 60)
    print("  SPACE  capture   |   q  finish & calibrate")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if not found:
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        display = frame.copy()
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)

        cv2.putText(display, f"Captured: {len(obj_points)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"  [{len(obj_points)}] captured")
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 5:
        print("[ERROR] Need >= 5 images.")
        return None

    img_size = gray.shape[::-1]
    print(f"\n[INFO] Calibrating with {len(obj_points)} images ({'fisheye' if fisheye else 'pinhole'})...")

    if fisheye:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        obj_pts = [p.astype(np.float64).reshape(-1, 1, 3) for p in obj_points]
        img_pts = [p.astype(np.float64).reshape(-1, 1, 2) for p in img_points]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_pts, img_pts, img_size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        print(f"[INFO] RMS error: {rms:.4f}")
    else:
        ret, K, D, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
        print(f"[INFO] RMS error: {ret:.4f}")

    print(f"[INFO] Camera matrix:\n{K}")
    print(f"[INFO] Distortion: {D.ravel()}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path, camera_matrix=K, dist_coeffs=D,
             image_size=img_size, fisheye=fisheye)
    print(f"[INFO] Saved -> {save_path}")
    return K, D


def _undistort_fisheye(frame, K, D):
    """Undistort using fisheye model (D has 4 params) or pinhole model."""
    if D is None:
        return frame
    D_flat = np.asarray(D).ravel()
    if len(D_flat) <= 4:
        return cv2.fisheye.undistortImage(frame, K, D, Knew=K)
    else:
        return cv2.undistort(frame, K, D)


def _init_undistort_map(K, D, size):
    """Init remap for fisheye or pinhole model."""
    if D is None:
        return None
    D_flat = np.asarray(D).ravel()
    if len(D_flat) <= 4:
        return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)
    else:
        return cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_16SC2)


# ---------------------------------------------------------------------------
# HandEyeCalibrator
# ---------------------------------------------------------------------------

class HandEyeCalibrator:
    """Eye-in-hand calibration for the Piper arm (drag-teaching workflow)."""

    def __init__(self, can_name="can0", chessboard_size=(10, 7), square_size=0.025,
                 camera_matrix=None, dist_coeffs=None, save_dir="calibration_data"):
        self.can_name = can_name
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.save_dir = save_dir

        # pre-compute object points for solvePnP
        cols, rows = chessboard_size
        self._objp = np.zeros((cols * rows, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

        self.robot_poses = []   # list of 4x4:  T_ee_base
        self.camera_poses = []  # list of 4x4:  T_target_cam

        self._piper = None
        os.makedirs(save_dir, exist_ok=True)

    # -- arm ----------------------------------------------------------------

    def connect(self):
        """Connect to the arm and enter drag-teaching mode."""
        print(f"[INFO] Connecting to Piper arm via {self.can_name} ...")
        self._piper = C_PiperInterface_V2(can_name=self.can_name)
        self._piper.ConnectPort()
        if not self._piper.isOk():
            raise RuntimeError("Failed to connect to Piper arm.")
        print("[INFO] Connected. Enabling ...")
        self._piper.EnablePiper()
        time.sleep(0.3)
        # enter drag-teaching
        self._piper.MotionCtrl_1(0x00, 0x00, 0x01)
        print("[INFO] Drag-teaching ON - drag the arm freely.")

    def disconnect(self, safe=True):
        """Disconnect CAN port.  By default keeps the arm alive.

        Args:
            safe: if True, only close the CAN port - arm stays in current mode.
                  if False, exit drag-teaching and disable motors (arm goes limp).
        """
        if self._piper is None:
            return
        if not safe:
            self._piper.MotionCtrl_1(0x00, 0x00, 0x00)
            time.sleep(0.2)
            self._piper.DisablePiper()
            time.sleep(0.2)
        self._piper.DisconnectPort()
        print("[INFO] CAN port closed (arm state preserved).")

    def read_arm_pose(self):
        """Return 4x4 end-effector-in-base pose, or None."""
        try:
            ep = self._piper.GetArmEndPoseMsgs().end_pose
            xyz = [ep.X_axis / 1e6, ep.Y_axis / 1e6, ep.Z_axis / 1e6]
            rpy = np.radians([ep.RX_axis / 1000.0,
                              ep.RY_axis / 1000.0,
                              ep.RZ_axis / 1000.0])
            return _rpy_to_matrix(xyz, rpy)
        except Exception as e:
            print(f"[WARN] read arm pose: {e}")
            return None

    def read_arm_status(self):
        """Return a dict with current arm state for on-screen display."""
        info = {}
        try:
            # arm status
            s = self._piper.GetArmStatus().arm_status
            ctrl_modes = {0x00: "STANDBY", 0x01: "CAN_CTRL", 0x02: "REMOTE"}
            arm_states = {0x00: "OK", 0x01: "E-STOP", 0x02: "ERROR", 0x03: "COLLISION"}
            move_modes = {0x00: "MOVE_P", 0x01: "MOVE_J", 0x02: "MOVE_L", 0x03: "DRAG"}
            motion   = {0x00: "REACHED", 0x01: "MOVING", 0x02: "PAUSED"}
            info['ctrl_mode']  = ctrl_modes.get(s.ctrl_mode, f"0x{s.ctrl_mode:02X}")
            info['arm_state']  = arm_states.get(s.arm_status, f"0x{s.arm_status:02X}")
            info['move_mode']  = move_modes.get(s.mode_feed, f"0x{s.mode_feed:02X}")
            info['motion']     = motion.get(s.motion_status, f"0x{s.motion_status:02X}")
            info['err_code']   = f"0x{s._err_code:04X}"
            info['err_status'] = s.err_status  # detailed error bitfields
        except Exception:
            pass

        try:
            # joint angles
            j = self._piper.GetArmJointMsgs().joint_state
            info['joints_deg'] = np.array([
                j.joint_1 / 1000.0, j.joint_2 / 1000.0,
                j.joint_3 / 1000.0, j.joint_4 / 1000.0,
                j.joint_5 / 1000.0, j.joint_6 / 1000.0])
        except Exception:
            pass

        try:
            info['enabled'] = all(self._piper.GetArmEnableStatus())
        except Exception:
            pass

        return info

    # -- checkerboard -------------------------------------------------------

    def detect_board(self, frame, robust=False):
        """Return refined subpixel corners.

        Args:
            frame: BGR image
            robust: if True, try multiple detector+preprocess combos and pick
                    the one with lowest PnP reprojection error (slower but
                    more reliable, for sample capture).
                    if False, fast single-pass detection (for live display).
        """
        # Undistort first — fisheye distortion makes checkerboard undetectable
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            frame = _undistort_fisheye(frame, self.camera_matrix, self.dist_coeffs)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = self.chessboard_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if not robust:
            # Fast path: single attempt for live display
            try:
                ok, corners = cv2.findChessboardCornersSB(
                    gray, size, flags=cv2.CALIB_CB_NORMAL | cv2.CALIB_CB_ACCURACY)
                if ok and corners is not None:
                    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                pass
            try:
                ok, corners = cv2.findChessboardCorners(gray, size, None)
                if ok and corners is not None:
                    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                pass
            return None

        # Robust path: try 4 combos, keep best RPE (for sample capture)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_c = clahe.apply(gray)

        best_corners = None
        best_rpe = float('inf')

        for pre_name, g in [('SB+raw', gray), ('SB+clahe', gray_c),
                             ('CL+raw', gray), ('CL+clahe', gray_c)]:
            ok = False
            corners = None
            try:
                if pre_name.startswith('SB'):
                    ok, corners = cv2.findChessboardCornersSB(
                        g, size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
                else:
                    ok, corners = cv2.findChessboardCorners(g, size, None)
            except Exception:
                continue

            if not ok or corners is None:
                continue

            c_ref = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), criteria)
            # RPE computed on UNDISTORTED image with identity distortion
            ret, rv, tv = cv2.solvePnP(self._objp, c_ref,
                                        self.camera_matrix, None,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
            if not ret:
                continue
            proj, _ = cv2.projectPoints(self._objp, rv, tv,
                                         self.camera_matrix, None)
            rpe = np.mean(np.linalg.norm(proj.reshape(-1, 2) - c_ref.reshape(-1, 2), axis=1))
            if rpe < best_rpe:
                best_rpe = rpe
                best_corners = c_ref

        return best_corners

    # -- sample -------------------------------------------------------------

    def capture_sample(self, frame):
        """Capture one (T_ee_base, T_target_cam) pair."""
        if self.camera_matrix is None:
            print("[ERROR] Camera intrinsics not set.")
            return False

        corners = self.detect_board(frame, robust=True)
        if corners is None:
            print("[WARN] Checkerboard not found.")
            return False

        # Corners are in undistorted coords — PnP with zero distortion
        T_target_cam, rpe = _solve_pnp_with_rpe(self._objp, corners,
                                                  self.camera_matrix, None)
        if T_target_cam is None:
            print("[WARN] PnP failed.")
            return False

        T_ee_base = self.read_arm_pose()
        if T_ee_base is None:
            return False

        self.robot_poses.append(T_ee_base)
        self.camera_poses.append(T_target_cam)

        n = len(self.robot_poses)
        xyz, rpy = _matrix_to_rpy(T_ee_base)
        quality_flag = " *** BAD PnP" if rpe > 1.0 else ""
        print(f"[{n}] rpe={rpe:.2f}px  "
              f"pos=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}] m  "
              f"rpy=[{np.degrees(rpy[0]):.1f},{np.degrees(rpy[1]):.1f},{np.degrees(rpy[2]):.1f}] deg"
              f"{quality_flag}")

        # Save UNDISTORTED image (corners detected on undistorted frame)
        undistorted = _undistort_fisheye(frame, self.camera_matrix, self.dist_coeffs)
        annotated = cv2.drawChessboardCorners(
            undistorted, self.chessboard_size, corners, True)
        cv2.imwrite(os.path.join(self.save_dir, f"sample_{n:03d}.jpg"), annotated)
        return True

    # -- solve --------------------------------------------------------------

    def calibrate(self, method=cv2.CALIB_HAND_EYE_TSAI):
        """Solve AX=XB.  Returns (X_4x4, rot_err_rad, trans_err_m) or None."""
        n = len(self.robot_poses)
        if n < 3:
            print(f"[ERROR] Need >= 3 samples, have {n}.")
            return None

        # --- remove samples with inconsistent rotation angles ---
        keep_idx = self._filter_consistent(list(range(n)))
        if keep_idx is None:
            keep_idx = list(range(n))
        kept = len(keep_idx)
        removed = n - kept
        if removed > 0:
            removed_ids = [i+1 for i in range(n) if i not in keep_idx]
            print(f"\n[INFO] Removed {removed} inconsistent sample(s): {removed_ids}")
            print(f"[INFO] Running calibration with {kept} / {n} samples ...")
        else:
            print(f"\n[INFO] Hand-eye calibration with {n} samples ...")

        R_ee = [self.robot_poses[i][:3, :3] for i in keep_idx]
        t_ee = [self.robot_poses[i][:3, 3] for i in keep_idx]
        R_tc = [self.camera_poses[i][:3, :3] for i in keep_idx]
        t_tc = [self.camera_poses[i][:3, 3] for i in keep_idx]

        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            R_ee, t_ee, R_tc, t_tc, method=method)

        X = np.eye(4)
        X[:3, :3] = R_cam2ee
        X[:3, 3] = t_cam2ee.flatten()

        # --- quality metrics ---
        robot_sub = [self.robot_poses[i] for i in keep_idx]
        camera_sub = [self.camera_poses[i] for i in keep_idx]
        axxb_rot, axxb_trans = self._residual_axxb(R_cam2ee, t_cam2ee, robot_sub, camera_sub)
        board_pos_std, board_rot_std = self._board_consistency(X, robot_sub, camera_sub)

        print(f"\n[RESULT] Camera -> End-Effector  (X):")
        print(f"         Rotation:\n{R_cam2ee}")
        print(f"         Translation (m): {t_cam2ee.flatten()}")
        xyz, rpy = _matrix_to_rpy(X)
        print(f"         Euler RPY (deg): {np.degrees(rpy)}")
        print(f"  ---- Quality Metrics ({kept}/{n} samples) ----")
        print(f"  AX-XB  residual:  rot={np.degrees(axxb_rot):.4f} deg  |  "
              f"trans={axxb_trans * 1000:.3f} mm")
        print(f"  Board consistency: pos std={board_pos_std * 1000:.3f} mm  |  "
              f"rot std={np.degrees(board_rot_std):.4f} deg")
        self._rate_quality(board_pos_std * 1000, board_rot_std)

        return X, axxb_rot, axxb_trans

    def _filter_consistent(self, indices):
        """Remove samples whose rotation angles are inconsistent with others.

        For each pair, |angle(A)| should ≈ |angle(B)|. A sample that causes
        large mismatches with many other samples is likely an outlier.
        """
        n = len(indices)
        if n < 5:
            return None

        # Score each sample by how much its rotation magnitudes differ from others
        scores = []
        for idx in indices:
            diffs = []
            for jdx in indices:
                if idx == jdx:
                    continue
                A = np.linalg.inv(self.robot_poses[idx]) @ self.robot_poses[jdx]
                B = np.linalg.inv(self.camera_poses[idx]) @ self.camera_poses[jdx]
                ang_A = np.arccos(np.clip((np.trace(A[:3,:3]) - 1) / 2, -1, 1))
                ang_B = np.arccos(np.clip((np.trace(B[:3,:3]) - 1) / 2, -1, 1))
                # Only penalize large mismatches (small rotations are noisy anyway)
                if max(ang_A, ang_B) > np.radians(10):
                    diffs.append(abs(ang_A - ang_B))
            scores.append(np.mean(diffs) if diffs else 0.0)

        scores = np.array(scores)
        # Remove samples with > 3x median mismatch
        median = np.median(scores)
        if median < 1e-6:
            return None
        thresh = max(median * 3, np.radians(10))  # at least 10 deg threshold
        keep = [indices[i] for i in range(n) if scores[i] < thresh]
        if len(keep) < 3:
            return None
        if len(keep) == n:
            return None  # nothing to remove
        return keep

    def _residual_axxb(self, R_cam2ee, t_cam2ee,
                        robot_poses=None, camera_poses=None):
        """||AX - XB|| over all sample pairs (proper angular error)."""
        if robot_poses is None:
            robot_poses = self.robot_poses
        if camera_poses is None:
            camera_poses = self.camera_poses

        X = np.eye(4)
        X[:3, :3] = R_cam2ee
        X[:3, 3] = t_cam2ee.flatten()

        rot_errs, trans_errs = [], []
        for i in range(len(robot_poses)):
            for j in range(i + 1, len(robot_poses)):
                A = np.linalg.inv(robot_poses[i]) @ robot_poses[j]
                B = np.linalg.inv(camera_poses[i]) @ camera_poses[j]
                AX = A @ X
                XB = X @ B
                R_err = AX[:3, :3] @ XB[:3, :3].T
                trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
                rot_errs.append(np.arccos(trace))
                trans_errs.append(np.linalg.norm(AX[:3, 3] - XB[:3, 3]))
        return np.mean(rot_errs), np.mean(trans_errs)

    def _board_consistency(self, X, robot_poses=None, camera_poses=None):
        """Checkerboard should be at a fixed pose in the base frame.
        Compute std dev of T_target_base across all samples.
        """
        if robot_poses is None:
            robot_poses = self.robot_poses
        if camera_poses is None:
            camera_poses = self.camera_poses

        board_poses = []
        for T_ee, T_tc in zip(robot_poses, camera_poses):
            T_target_base = T_ee @ X @ T_tc
            board_poses.append(T_target_base)

        positions = np.array([T[:3, 3] for T in board_poses])
        pos_mean = positions.mean(axis=0)
        pos_std = np.mean([np.linalg.norm(p - pos_mean) for p in positions])

        # rotation std via quaternion spread
        quats = []
        for T in board_poses:
            R = T[:3, :3]
            qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            qx = np.sign(R[2, 1] - R[1, 2]) * np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
            qy = np.sign(R[0, 2] - R[2, 0]) * np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
            qz = np.sign(R[1, 0] - R[0, 1]) * np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
            quats.append([qx, qy, qz, qw])
        q_mean = np.mean(quats, axis=0)
        q_mean /= np.linalg.norm(q_mean)
        ang_devs = [2 * np.arccos(min(1, abs(np.dot(q[:4], q_mean)))) for q in quats]
        rot_std = np.mean(ang_devs)

        return pos_std, rot_std

    @staticmethod
    def _rate_quality(pos_std_mm, rot_std_rad):
        """Print a simple quality rating."""
        if pos_std_mm < 1.0 and rot_std_rad < 0.005:
            rating = "EXCELLENT"
        elif pos_std_mm < 3.0 and rot_std_rad < 0.015:
            rating = "GOOD"
        elif pos_std_mm < 8.0 and rot_std_rad < 0.04:
            rating = "FAIR"
        else:
            rating = "POOR (collect more / better-spread samples)"
        print(f"  Rating: {rating}")

    # -- io -----------------------------------------------------------------

    def save_result(self, X, filename=None):
        """Persist calibration matrix to .npz + .txt."""
        if filename is None:
            filename = os.path.join(self.save_dir, "hand_eye_result.npz")
        np.savez(filename, X=X, R_cam2ee=X[:3, :3], t_cam2ee=X[:3, 3],
                 robot_poses=np.array(self.robot_poses),
                 camera_poses=np.array(self.camera_poses))
        print(f"[INFO] Saved -> {filename}")

        txt = os.path.join(self.save_dir, "hand_eye_result.txt")
        xyz, rpy = _matrix_to_rpy(X)
        with open(txt, 'w') as f:
            f.write(f"Eye-in-Hand Calibration\n"
                    f"Date: {datetime.now().isoformat()}\n"
                    f"Samples: {len(self.robot_poses)}\n\n"
                    f"X (camera -> end-effector):\n{X}\n\n"
                    f"Position (m): {xyz}\n"
                    f"Euler RPY (deg): {np.degrees(rpy)}\n")
        print(f"[INFO] Saved -> {txt}")

    def save_samples(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_dir, "samples.npz")
        np.savez(filename,
                 robot_poses=np.array(self.robot_poses),
                 camera_poses=np.array(self.camera_poses))
        print(f"[INFO] Samples saved -> {filename}")

    def load_samples(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.robot_poses = list(data['robot_poses'])
        self.camera_poses = list(data['camera_poses'])
        print(f"[INFO] Loaded {len(self.robot_poses)} samples from {filename}")


# ---------------------------------------------------------------------------
# interactive loop
# ---------------------------------------------------------------------------

METHOD_MAP = {
    "tsai":       cv2.CALIB_HAND_EYE_TSAI,
    "park":       cv2.CALIB_HAND_EYE_PARK,
    "horaud":     cv2.CALIB_HAND_EYE_HORAUD,
    "andreff":    cv2.CALIB_HAND_EYE_ANDREFF,
    "daniiilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _live_loop(calib, camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Pre-compute undistortion map (fisheye or pinhole)
    K = calib.camera_matrix
    D = calib.dist_coeffs
    h, w = 480, 640
    map_result = _init_undistort_map(K, D, (w, h))
    use_undistort = map_result is not None
    if use_undistort:
        map1, map2 = map_result

    print("\n" + "=" * 60)
    print("  Eye-in-Hand Calibration  |  Drag Teaching")
    print("=" * 60)
    print("  SPACE  capture sample    c   calibrate")
    print("  d      delete last        s   save samples")
    print("  q      quit (arm stays alive)")
    print("=" * 60 + "\n")

    # Rate-limiting: heavy operations only run every N frames
    DETECT_EVERY = 5      # checkerboard detection (expensive)
    CAN_READ_EVERY = 3    # arm CAN reads

    _cached_corners = None
    _cached_pose = None
    _cached_status = {}
    _frame_cnt = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            _frame_cnt += 1
            # Undistort for display (fast remap) — raw frame goes to capture
            if use_undistort:
                display = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            else:
                display = frame.copy()
            n = len(calib.robot_poses)

            # --- expensive operations, throttled ---
            if _frame_cnt % DETECT_EVERY == 0:
                _cached_corners = calib.detect_board(frame)

            if _frame_cnt % CAN_READ_EVERY == 0:
                _cached_pose = calib.read_arm_pose()
                _cached_status = calib.read_arm_status()

            # --- fast overlay drawing ---
            corners = _cached_corners
            if corners is not None:
                cv2.drawChessboardCorners(display, calib.chessboard_size, corners, True)
                cv2.putText(display, f"Samples: {n}  [SPACE to capture]", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Samples: {n}  NO BOARD", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            y = 48
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            WHITE = (255, 255, 255)
            CYAN = (255, 255, 0)

            # end-effector pose
            T = _cached_pose
            if T is not None:
                xyz, rpy = _matrix_to_rpy(T)
                cv2.putText(display, f"EE Pos: [{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}] m",
                            (10, y), FONT, 0.45, WHITE, 1)
                y += 18
                cv2.putText(display, f"EE RPY: [{np.degrees(rpy[0]):.1f} "
                            f"{np.degrees(rpy[1]):.1f} {np.degrees(rpy[2]):.1f}] deg",
                            (10, y), FONT, 0.45, WHITE, 1)
                y += 22

            # arm status
            status = _cached_status
            enabled = status.get('enabled')
            if enabled is not None:
                clr = (0, 255, 0) if enabled else (0, 0, 255)
                cv2.putText(display, f"MOTORS: {'ON' if enabled else 'OFF'}", (10, y),
                            FONT, 0.5, clr, 2)
                y += 22

            ctrl  = status.get('ctrl_mode', '?')
            state = status.get('arm_state', '?')
            move  = status.get('move_mode', '?')
            motn  = status.get('motion', '?')
            err   = status.get('err_code', '?')
            cv2.putText(display, f"Ctrl:{ctrl} State:{state} Mode:{move} Motion:{motn}",
                        (10, y), FONT, 0.4, CYAN, 1)
            y += 16
            cv2.putText(display, f"ErrCode: {err}", (10, y), FONT, 0.4,
                        (0, 0, 255) if err != '0x0000' else (100, 255, 100), 1)
            y += 20

            joints = status.get('joints_deg')
            if joints is not None:
                cv2.putText(display, "Joints (deg):", (10, y), FONT, 0.4, WHITE, 1)
                y += 16
                cv2.putText(display,
                            f"  J1:{joints[0]:7.1f} J2:{joints[1]:7.1f} J3:{joints[2]:7.1f}",
                            (10, y), FONT, 0.35, CYAN, 1)
                y += 14
                cv2.putText(display,
                            f"  J4:{joints[3]:7.1f} J5:{joints[4]:7.1f} J6:{joints[5]:7.1f}",
                            (10, y), FONT, 0.35, CYAN, 1)

            h = display.shape[0]
            cv2.putText(display, "SPACE:capture  c:calib  d:del  s:save  q:quit",
                        (10, h - 8), FONT, 0.38, (180, 180, 180), 1)

            cv2.imshow("Hand-Eye Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                calib.capture_sample(frame)
            elif key == ord('c'):
                res = calib.calibrate(method=METHOD_MAP["tsai"])
                if res:
                    calib.save_result(res[0])
            elif key == ord('d'):
                if calib.robot_poses:
                    calib.robot_poses.pop()
                    calib.camera_poses.pop()
                    n = len(calib.robot_poses)
                    # Also delete saved image
                    img = os.path.join(calib.save_dir, f"sample_{n+1:03d}.jpg")
                    if os.path.exists(img):
                        os.remove(img)
                    print(f"[INFO] Deleted sample {n+1}, {n} remaining.")
                else:
                    print("[INFO] No samples to delete.")
            elif key == ord('s'):
                calib.save_samples()
            elif key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Eye-in-hand hand-eye calibration for Piper arm (drag-teaching)")

    parser.add_argument("--can", default="can0")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cols", type=int, default=10,
                        help="checkerboard inner corners (cols)")
    parser.add_argument("--rows", type=int, default=7,
                        help="checkerboard inner corners (rows)")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="square side length (m)")
    parser.add_argument("--save-dir", default="calibration_data")
    parser.add_argument("--load-camera", default=None,
                        help="load intrinsics from .npz / .yaml")
    parser.add_argument("--calib-camera", action="store_true",
                        help="run interactive camera calibration first")
    parser.add_argument("--load-samples", default=None,
                        help="load pre-collected samples (.npz)")
    parser.add_argument("--no-resume", action="store_true",
                        help="do NOT auto-load existing samples from save-dir")
    parser.add_argument("--offline", action="store_true",
                        help="calibrate from saved samples (no arm)")
    parser.add_argument("--method", default="tsai",
                        choices=list(METHOD_MAP.keys()))

    args = parser.parse_args()

    # --- camera intrinsics ---
    camera_matrix, dist_coeffs = None, None

    if args.calib_camera:
        r = calibrate_camera_interactive(
            chessboard_size=(args.cols, args.rows),
            square_size=args.square_size,
            camera_id=args.camera,
            save_path=os.path.join(args.save_dir, "camera_intrinsics.npz"))
        if r:
            camera_matrix, dist_coeffs = r

    if args.load_camera:
        camera_matrix, dist_coeffs = load_intrinsics(args.load_camera)
        if camera_matrix is not None:
            print(f"[INFO] Loaded camera intrinsics from {args.load_camera}")
            print(f"       K:\n{camera_matrix}")
            print(f"       dist: {dist_coeffs.ravel()}")

    calib = HandEyeCalibrator(
        can_name=args.can,
        chessboard_size=(args.cols, args.rows),
        square_size=args.square_size,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        save_dir=args.save_dir,
    )

    # --- offline -----------------------------------------------------------
    if args.offline:
        if args.load_samples:
            calib.load_samples(args.load_samples)
        if camera_matrix is None:
            print("[ERROR] Offline mode requires --load-camera")
            sys.exit(1)
        res = calib.calibrate(method=METHOD_MAP[args.method])
        if res:
            calib.save_result(res[0])
        return

    # --- live --------------------------------------------------------------
    if camera_matrix is None:
        print("[ERROR] Camera intrinsics required (--load-camera or --calib-camera).")
        sys.exit(1)

    # --- auto-resume previous samples ---
    auto_samples = os.path.join(args.save_dir, "samples.npz")
    if not args.no_resume and not args.load_samples and os.path.exists(auto_samples):
        calib.load_samples(auto_samples)
        print("[INFO] Auto-loaded previous samples. Use --no-resume to start fresh.")

    calib.connect()
    try:
        if args.load_samples:
            calib.load_samples(args.load_samples)
        _live_loop(calib, args.camera)
    finally:
        if calib.robot_poses:
            calib.save_samples()
        # safe disconnect: closes CAN port, arm stays in current mode (no sudden limp)
        calib.disconnect(safe=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Calibrate one fixed (eye-on-base) camera using the already-calibrated
eye-in-hand camera cam0.

Principle
---------
  cam0 is on the end-effector (eye-in-hand), hand-eye X = T_cam0_ee is known.
  camX is fixed in the workspace (eye-on-base), to be calibrated.

  All cameras observe the same static checkerboard.

  For each arm pose:
    1.  T_target_cam0  = solvePnP(board, cam0)
    2.  T_target_camX  = solvePnP(board, camX)
    3.  T_ee_base      = read arm FK
    4.  T_target_base  = T_ee_base * X * T_target_cam0
    5.  T_camX_base    = T_target_base * inv(T_target_camX)

  Collect N samples and average (rotation via quaternion mean).

Usage
-----
  # First: calibrate cam1
  python multi_cam_calibration.py \
      --load-hand-eye calibration_data/hand_eye_result.npz \
      --load-cam0 ./config/cam0_calibration.yaml \
      --load-cam ./config/cam1_calibration.yaml \
      --cam-id 1

  # Then: calibrate cam2 (run separately)
  python multi_cam_calibration.py \
      --load-hand-eye calibration_data/hand_eye_result.npz \
      --load-cam0 ./config/cam0_calibration.yaml \
      --load-cam ./config/cam2_calibration.yaml \
      --cam-id 2

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


# ---------------------------------------------------------------------------
# yaml intrinsics loader
# ---------------------------------------------------------------------------

def load_intrinsics(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        return d['camera_matrix'], d['dist_coeffs']
    if ext in (".yaml", ".yml"):
        return _load_intrinsics_yaml(path)
    print(f"[ERROR] Unsupported format: {ext}")
    return None, None


def _load_intrinsics_yaml(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    def _parse_nested_list(start_idx, key_indent):
        rows, current_row = [], []
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

    K, dist = None, None
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
        return None, None
    return K.astype(np.float64), (dist.astype(np.float64) if dist is not None
                                   else np.zeros((1, 5)))


# ---------------------------------------------------------------------------
# SingleExternalCameraCalibrator
# ---------------------------------------------------------------------------

class SingleExternalCameraCalibrator:
    """Calibrate ONE fixed camera using the known hand-eye result from cam0."""

    def __init__(self, X_cam0_ee, ext_cam_id, chessboard_size=(10, 7),
                 square_size=0.025, save_dir="multi_cam_calib"):
        self.X_cam0_ee = X_cam0_ee
        self.ext_cam_id = ext_cam_id   # 1 or 2
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.save_dir = save_dir

        cols, rows = chessboard_size
        self._objp = np.zeros((cols * rows, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

        self.K_cam0 = None
        self.D_cam0 = None
        self.K_ext  = None
        self.D_ext  = None

        self.T_camX_base_list = []
        self._raw_samples = []  # (T_ee_base, T_target_cam0, T_target_camX, corners_img)
        self.sample_count = 0

        self._piper = None
        os.makedirs(save_dir, exist_ok=True)

    # -- arm ----------------------------------------------------------------

    def connect(self):
        print(f"[INFO] Connecting to Piper arm ...")
        self._piper = C_PiperInterface_V2(can_name="can0")
        self._piper.ConnectPort()
        if not self._piper.isOk():
            raise RuntimeError("Failed to connect to Piper arm.")
        print("[INFO] Connected. Enabling ...")
        self._piper.EnablePiper()
        time.sleep(0.3)
        self._piper.MotionCtrl_1(0x00, 0x00, 0x01)
        print("[INFO] Drag-teaching ON - drag the arm freely.")

    def disconnect(self, safe=True):
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
        info = {}
        try:
            s = self._piper.GetArmStatus().arm_status
            ctrl_modes = {0x00: "STANDBY", 0x01: "CAN_CTRL", 0x02: "REMOTE"}
            arm_states = {0x00: "OK", 0x01: "E-STOP", 0x02: "ERROR", 0x03: "COLLISION"}
            info['ctrl_mode'] = ctrl_modes.get(s.ctrl_mode, f"0x{s.ctrl_mode:02X}")
            info['arm_state'] = arm_states.get(s.arm_status, f"0x{s.arm_status:02X}")
            info['err_code']  = f"0x{s._err_code:04X}"
        except Exception:
            pass
        try:
            info['enabled'] = all(self._piper.GetArmEnableStatus())
        except Exception:
            pass
        return info

    # -- detection ----------------------------------------------------------

    def detect_board(self, frame, robust=False):
        """Detect checkerboard. Caller should undistort the frame first."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = self.chessboard_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if not robust:
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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_c = clahe.apply(gray)

        for g in [gray, gray_c]:
            for detector in ['SB', 'CL']:
                try:
                    if detector == 'SB':
                        ok, corners = cv2.findChessboardCornersSB(
                            g, size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
                    else:
                        ok, corners = cv2.findChessboardCorners(g, size, None)
                    if ok and corners is not None:
                        return cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), criteria)
                except Exception:
                    continue
        return None

    def pnp(self, corners, K, D):
        return _solve_pnp(self._objp, corners, K, D)

    # -- sample -------------------------------------------------------------

    def capture_sample(self, frame_cam0, frame_ext):
        """Capture one sample from both cameras.

        Returns True on success.
        """
        if self.K_cam0 is None or self.K_ext is None:
            print("[ERROR] Camera intrinsics not set.")
            return False

        T_ee_base = self.read_arm_pose()
        if T_ee_base is None:
            return False

        # Undistort before detection (fisheye)
        if self.K_cam0 is not None and self.D_cam0 is not None:
            frame_cam0 = cv2.undistort(frame_cam0, self.K_cam0, self.D_cam0)
        if self.K_ext is not None and self.D_ext is not None:
            frame_ext = cv2.undistort(frame_ext, self.K_ext, self.D_ext)

        corners0 = self.detect_board(frame_cam0, robust=True)
        cornersX = self.detect_board(frame_ext, robust=True)

        if corners0 is None:
            print("[WARN] cam0: checkerboard not found.")
            return False
        if cornersX is None:
            print("[WARN] cam{}: checkerboard not found.".format(self.ext_cam_id))
            return False

        # Corners in undistorted coords → PnP with zero distortion
        T_target_cam0 = self.pnp(corners0, self.K_cam0, None)
        T_target_camX = self.pnp(cornersX, self.K_ext, None)

        if T_target_cam0 is None or T_target_camX is None:
            print("[WARN] PnP failed.")
            return False

        # T_target_base via eye-in-hand chain
        T_target_base = T_ee_base @ self.X_cam0_ee @ T_target_cam0

        # T_camX_base
        T_camX_base = T_target_base @ np.linalg.inv(T_target_camX)
        self.T_camX_base_list.append(T_camX_base)
        self._raw_samples.append((T_ee_base, T_target_cam0, T_target_camX, cornersX))
        self.sample_count += 1

        n = self.sample_count
        xyz, rpy = _matrix_to_rpy(T_ee_base)
        print(f"\n[Sample {n}] EE pos=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}] m")
        p, r = _matrix_to_rpy(T_camX_base)
        print(f"  cam{self.ext_cam_id} -> base: "
              f"pos=[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}] m  "
              f"rpy=[{np.degrees(r[0]):.2f},{np.degrees(r[1]):.2f},{np.degrees(r[2]):.2f}] deg")

        # Save annotated images
        ann0 = cv2.drawChessboardCorners(
            frame_cam0.copy(), self.chessboard_size, corners0, True)
        annX = cv2.drawChessboardCorners(
            frame_ext.copy(), self.chessboard_size, cornersX, True)
        cv2.imwrite(os.path.join(self.save_dir, f"sample_{n:03d}_cam0.jpg"), ann0)
        cv2.imwrite(os.path.join(self.save_dir, f"sample_{n:03d}_cam{self.ext_cam_id}.jpg"), annX)

        return True

    # -- solve --------------------------------------------------------------

    def calibrate(self):
        T_list = self.T_camX_base_list
        n = len(T_list)
        if n < 3:
            print(f"[ERROR] Need >= 3 samples, have {n}.")
            return None

        # Average translation
        t_mean = np.mean([T[:3, 3] for T in T_list], axis=0)

        # Average rotation via quaternion
        quats = []
        for T in T_list:
            R = T[:3, :3]
            qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            qx = np.sign(R[2, 1] - R[1, 2]) * np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
            qy = np.sign(R[0, 2] - R[2, 0]) * np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
            qz = np.sign(R[1, 0] - R[0, 1]) * np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
            quats.append([qx, qy, qz, qw])

        q_mean = np.mean(quats, axis=0)
        q_mean /= np.linalg.norm(q_mean)
        qx, qy, qz, qw = q_mean

        R_mean = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
        ])

        T_mean = np.eye(4)
        T_mean[:3, :3] = R_mean
        T_mean[:3, 3] = t_mean

        # Std deviation across samples (proper angular error for rotation)
        t_devs = [np.linalg.norm(T[:3, 3] - t_mean) for T in T_list]
        r_devs = []
        for T in T_list:
            R_err = T[:3, :3] @ R_mean.T
            trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
            r_devs.append(np.arccos(trace))
        t_std = np.mean(t_devs)
        r_std = np.mean(r_devs)

        # Reprojection error: project checkerboard into camX using T_mean
        # For each sample: P_cam = inv(T_mean) * T_target_base * P_board
        # Then project with K and compare with detected corners
        rpe = self._reprojection_error(T_mean)

        p, r = _matrix_to_rpy(T_mean)
        print(f"\n{'='*60}")
        print(f"  RESULT: cam{self.ext_cam_id} -> base  ({n} samples)")
        print(f"{'='*60}")
        print(f"  Position (m):     [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
        print(f"  Euler RPY (deg):  [{np.degrees(r[0]):.2f}, {np.degrees(r[1]):.2f}, {np.degrees(r[2]):.2f}]")
        print(f"  Rotation:\n{R_mean}")
        print(f"  Translation: {t_mean}")
        print(f"  ---- Quality Metrics ----")
        print(f"  Sample spread:  trans std={t_std*1000:.3f} mm  |  "
              f"rot std={np.degrees(r_std):.4f} deg")
        if rpe is not None:
            print(f"  Reprojection:   mean={rpe:.3f} px")
        self._rate_quality(t_std * 1000, r_std, rpe)
        print(f"{'='*60}")

        return T_mean

    def _reprojection_error(self, T_camX_base):
        """Mean reprojection error (px) using the solved T_camX_base.

        For each sample, projects the checkerboard into camX and compares
        with the actual detected corners.
        """
        if not self._raw_samples or self.K_ext is None:
            return None

        K = self.K_ext
        D = self.D_ext
        objp = self._objp

        errors = []
        for T_ee, T_tc0, T_tcX, corners_img in self._raw_samples:
            # Board in base frame via cam0 chain
            T_board_base = T_ee @ self.X_cam0_ee @ T_tc0
            # Board pose in camX frame using the MEAN T_camX_base
            T_board_camX = np.linalg.inv(T_camX_base) @ T_board_base
            rvec, _ = cv2.Rodrigues(T_board_camX[:3, :3])
            tvec = T_board_camX[:3, 3].reshape(3, 1)

            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
            proj = proj.reshape(-1, 2)
            corners_img_2d = corners_img.reshape(-1, 2)
            errors.append(np.mean(np.linalg.norm(proj - corners_img_2d, axis=1)))

        return np.mean(errors)

    @staticmethod
    def _rate_quality(pos_std_mm, rot_std_rad, rpe_px):
        if rpe_px is not None:
            if pos_std_mm < 1.0 and rot_std_rad < 0.005 and rpe_px < 0.5:
                rating = "EXCELLENT"
            elif pos_std_mm < 3.0 and rot_std_rad < 0.015 and rpe_px < 1.5:
                rating = "GOOD"
            elif pos_std_mm < 8.0 and rot_std_rad < 0.04 and rpe_px < 4.0:
                rating = "FAIR"
            else:
                rating = "POOR (collect more / better-spread samples)"
        else:
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

    def save_result(self, T):
        npz = os.path.join(self.save_dir, f"cam{self.ext_cam_id}_to_base.npz")
        np.savez(npz, T=T, R=T[:3, :3], t=T[:3, 3],
                 samples=np.array(self.T_camX_base_list))
        print(f"[INFO] Saved -> {npz}")

        txt = os.path.join(self.save_dir, f"cam{self.ext_cam_id}_to_base.txt")
        p, r = _matrix_to_rpy(T)
        with open(txt, 'w') as f:
            f.write(f"Camera {self.ext_cam_id} -> Base Calibration\n"
                    f"Date: {datetime.now().isoformat()}\n"
                    f"Samples: {self.sample_count}\n\n"
                    f"T_cam{self.ext_cam_id}_base:\n{T}\n\n"
                    f"Position (m): {p}\n"
                    f"Euler RPY (deg): {np.degrees(r)}\n")
        print(f"[INFO] Saved -> {txt}")

    def save_samples(self):
        if not self.T_camX_base_list:
            return
        f = os.path.join(self.save_dir, f"cam{self.ext_cam_id}_samples.npz")
        np.savez(f, T_list=np.array(self.T_camX_base_list),
                 sample_count=self.sample_count)
        print(f"[INFO] Samples saved -> {f}")

    def load_samples(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.T_camX_base_list = list(data['T_list'])
        self.sample_count = int(data['sample_count'])
        print(f"[INFO] Loaded {self.sample_count} samples from {filename}")


# ---------------------------------------------------------------------------
# live loop (2 cameras: cam0 + one external)
# ---------------------------------------------------------------------------

def _open_camera(device_id, width=640, height=480):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    for _ in range(5):
        cap.read()
    return cap


def _live_loop(calib, cam0_dev, ext_dev):
    cap0 = _open_camera(cam0_dev)
    capX = _open_camera(ext_dev)

    if cap0 is None:
        print("[ERROR] Cannot open cam0.")
        return
    if capX is None:
        print("[ERROR] Cannot open external camera.")
        cap0.release()
        return

    DETECT_EVERY = 5
    CAN_READ_EVERY = 3
    _cached_corners0 = None
    _cached_cornersX = None
    _cached_status = {}
    _frame_cnt = 0

    # Pre-compute undistortion maps for both cameras
    h, w = 480, 640
    map0 = mapX = None
    if calib.K_cam0 is not None and calib.D_cam0 is not None:
        map0 = cv2.initUndistortRectifyMap(calib.K_cam0, calib.D_cam0, None,
                                            calib.K_cam0, (w, h), cv2.CV_16SC2)
    if calib.K_ext is not None and calib.D_ext is not None:
        mapX = cv2.initUndistortRectifyMap(calib.K_ext, calib.D_ext, None,
                                            calib.K_ext, (w, h), cv2.CV_16SC2)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED   = (0, 0, 255)

    print("\n" + "=" * 60)
    print(f"  Calibrating cam{calib.ext_cam_id} (eye-on-base)")
    print("=" * 60)
    print("  SPACE  capture sample       c   calibrate")
    print("  d      delete last          s   save samples")
    print("  q      quit (arm stays alive)")
    print("=" * 60 + "\n")

    try:
        while True:
            ret0, frame0 = cap0.read()
            retX, frameX = capX.read()
            if not ret0 or not retX:
                time.sleep(0.05)
                continue

            _frame_cnt += 1
            n = calib.sample_count

            # Undistort for display + detection
            raw0, rawX = frame0, frameX  # keep raw for capture
            if map0 is not None:
                frame0 = cv2.remap(frame0, map0[0], map0[1], cv2.INTER_LINEAR)
            if mapX is not None:
                frameX = cv2.remap(frameX, mapX[0], mapX[1], cv2.INTER_LINEAR)

            if _frame_cnt % DETECT_EVERY == 0:
                _cached_corners0 = calib.detect_board(frame0)
                _cached_cornersX = calib.detect_board(frameX)

            if _frame_cnt % CAN_READ_EVERY == 0:
                _cached_status = calib.read_arm_status()

            # Build side-by-side display
            h, w = frame0.shape[:2]
            disp0 = frame0.copy()
            dispX = frameX.copy()

            for disp, corners, label in [
                (disp0, _cached_corners0, "cam0 (hand)"),
                (dispX, _cached_cornersX, f"cam{calib.ext_cam_id} (base)"),
            ]:
                if corners is not None:
                    cv2.drawChessboardCorners(disp, calib.chessboard_size, corners, True)
                    cv2.putText(disp, f"{label} OK", (5, 18), FONT, 0.45, GREEN, 2)
                else:
                    cv2.putText(disp, f"{label} --", (5, 18), FONT, 0.45, RED, 2)

            panel = np.hstack([disp0, dispX])

            # Bottom bar
            bar = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
            status = _cached_status
            enabled = status.get('enabled')
            emsg = "ON" if enabled else "OFF" if enabled is not None else "?"
            info = (f"Samples: {n}  Motors: {emsg}  "
                    f"Ctrl: {status.get('ctrl_mode', '?')}  "
                    f"State: {status.get('arm_state', '?')}  "
                    f"Err: {status.get('err_code', '?')}")
            cv2.putText(bar, info, (5, 18), FONT, 0.4, WHITE, 1)
            cv2.putText(bar, "SPACE:capture  c:calib  d:del  s:save  q:quit",
                        (5, 34), FONT, 0.35, (160, 160, 160), 1)
            panel = np.vstack([panel, bar])

            cv2.imshow("Multi-Camera Calibration", panel)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                calib.capture_sample(raw0, rawX)
            elif key == ord('c'):
                T = calib.calibrate()
                if T is not None:
                    calib.save_result(T)
            elif key == ord('d'):
                if calib.T_camX_base_list:
                    calib.T_camX_base_list.pop()
                    calib._raw_samples.pop()
                    calib.sample_count -= 1
                    n = calib.sample_count
                    for suffix in [f"cam0.jpg", f"cam{calib.ext_cam_id}.jpg"]:
                        img = os.path.join(calib.save_dir, f"sample_{n+1:03d}_{suffix}")
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
        cap0.release()
        capX.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ONE eye-on-base camera via eye-in-hand cam0")

    parser.add_argument("--load-hand-eye", required=True,
                        help="hand-eye result .npz (cam0 calibration output)")
    parser.add_argument("--load-cam0", required=True,
                        help="cam0 intrinsics (.yaml)")
    parser.add_argument("--load-cam", required=True,
                        help="external camera intrinsics (.yaml)")
    parser.add_argument("--cam-id", type=int, required=True, choices=[1, 2],
                        help="external camera ID (1 or 2)")
    parser.add_argument("--cam0-dev", type=int, default=4, help="cam0 device id")
    parser.add_argument("--cam-dev", type=int, default=3, help="external camera device id")
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--square-size", type=float, default=0.025)
    parser.add_argument("--save-dir", default="multi_cam_calib")
    parser.add_argument("--no-resume", action="store_true",
                        help="do NOT auto-load previous samples")
    parser.add_argument("--offline", action="store_true",
                        help="calibrate from saved samples (no arm)")

    args = parser.parse_args()

    # --- load hand-eye ---
    he = np.load(args.load_hand_eye)
    X = he['X']
    print(f"[INFO] Hand-eye X (cam0 -> ee):")
    print(f"       Rotation:\n{X[:3, :3]}")
    print(f"       Translation: {X[:3, 3]}")

    # --- load intrinsics ---
    K0, D0 = load_intrinsics(args.load_cam0)
    KX, DX = load_intrinsics(args.load_cam)
    if K0 is None or KX is None:
        print("[ERROR] Failed to load intrinsics.")
        sys.exit(1)
    print(f"[INFO] cam0 K:\n{K0}")
    print(f"[INFO] cam{args.cam_id} K:\n{KX}")

    # --- calibrator ---
    calib = SingleExternalCameraCalibrator(
        X_cam0_ee=X,
        ext_cam_id=args.cam_id,
        chessboard_size=(args.cols, args.rows),
        square_size=args.square_size,
        save_dir=args.save_dir,
    )
    calib.K_cam0 = K0
    calib.D_cam0 = D0
    calib.K_ext  = KX
    calib.D_ext  = DX

    if args.offline:
        # TODO: load pre-saved samples and run calibrate
        print("[ERROR] Offline mode not implemented yet.")
        sys.exit(1)

    # --- auto-resume ---
    auto_samples = os.path.join(args.save_dir, f"cam{args.cam_id}_samples.npz")
    if not args.no_resume and os.path.exists(auto_samples):
        calib.load_samples(auto_samples)
        print("[INFO] Auto-loaded previous samples. Use --no-resume to start fresh.")

    calib.connect()
    try:
        _live_loop(calib, args.cam0_dev, args.cam_dev)
    finally:
        calib.disconnect(safe=True)
        calib.save_samples()


if __name__ == "__main__":
    main()

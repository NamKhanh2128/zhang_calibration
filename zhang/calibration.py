"""
Calibration pipeline:

    detect corners ─► Zhang closed form (K, board poses, no distortion)
                   ─► non-linear refinement of K, distortion and poses (Levenberg-Marquardt,
                      cv2.calibrateCameraExtended started from the closed-form K)
                   ─► reject images whose RMS error is far above the median, refine again

Distortion model (OpenCV, 5 coefficients): with x, y normalised coordinates and r^2 = x^2 + y^2

    x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
    y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y

k3 is held at 0 by default. The r^6 term only matters near the image corners; when the board
never reaches them, k3 is not constrained by the data and the fit extrapolates (on the iPhone
data: k3 = -2.4 predicts a 237 px shift at the corners, against 22 px with k3 = 0, while the
cross-validated error improves by only 0.04 px).

Author: Nguyen Duc Nam Khanh
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .closed_form import calibrate_closed_form
from .corners import Board, detect_corners
from .images import load_rgb

INTRINSIC_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]


@dataclass
class Detection:
    path: str
    corners: np.ndarray            # (N, 2) sub-pixel image coordinates

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Calibration:
    image_size: Tuple[int, int]    # (width, height)
    K: np.ndarray                  # 3x3, OpenCV pixel convention (pixel centres at integers)
    dist: np.ndarray               # (5,) k1, k2, p1, p2, k3
    std: Dict[str, float]          # 1-sigma uncertainty of every intrinsic parameter
    K_closed_form: np.ndarray      # Zhang's linear estimate (with skew, no distortion)
    detections: List[Detection]    # images used in the final solution
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    rejected: List[str] = field(default_factory=list)       # removed as outliers
    not_detected: List[str] = field(default_factory=list)   # board not found

    def residuals(self, board: Board) -> List[np.ndarray]:
        """Detected minus reprojected corner position, (N, 2) per image."""
        out = []
        for det, rvec, tvec in zip(self.detections, self.rvecs, self.tvecs):
            projected, _ = cv2.projectPoints(board.object_points(), rvec, tvec, self.K, self.dist)
            out.append(det.corners - projected.reshape(-1, 2))
        return out

    def per_image_rms(self, board: Board) -> np.ndarray:
        return np.array([np.sqrt(np.mean(np.sum(r**2, axis=1))) for r in self.residuals(board)])

    def rms(self, board: Board) -> float:
        r = np.vstack(self.residuals(board))
        return float(np.sqrt(np.mean(np.sum(r**2, axis=1))))


def detect_all(paths: List[str], board: Board, log=print) -> Tuple[List[Detection], List[str], Tuple[int, int]]:
    """Detect the board in every image. All images must share one size."""
    detections, failed, size = [], [], None
    for k, path in enumerate(paths, 1):
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            raise ValueError(f"{os.path.basename(path)} is {w}x{h}, expected {size[0]}x{size[1]}")
        corners = detect_corners(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), board)
        status = "ok" if corners is not None else "board not found"
        log(f"  [{k:2d}/{len(paths)}] {os.path.basename(path):<16s} {status}")
        if corners is None:
            failed.append(os.path.basename(path))
        else:
            detections.append(Detection(path, corners))
    return detections, failed, size


def _refine(detections: List[Detection], board: Board, size: Tuple[int, int], K_init: np.ndarray, fit_k3: bool):
    """Joint Levenberg-Marquardt refinement of K, distortion and all board poses."""
    obj = [board.object_points()] * len(detections)
    img = [d.corners.astype(np.float32).reshape(-1, 1, 2) for d in detections]
    K0 = K_init.copy()
    K0[0, 1] = 0.0                                       # OpenCV's model has zero skew
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (0 if fit_k3 else cv2.CALIB_FIX_K3)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    _, K, dist, rvecs, tvecs, std_intr, _, _ = cv2.calibrateCameraExtended(
        obj, img, size, K0, np.zeros(5), flags=flags, criteria=criteria)
    std = dict(zip(INTRINSIC_NAMES, std_intr.ravel()[:9].tolist()))
    return K, dist.ravel()[:5], list(rvecs), list(tvecs), std


def calibrate(detections: List[Detection], board: Board, size: Tuple[int, int], outlier_factor: float = 3.0,
              fit_k3: bool = False, not_detected: List[str] = None) -> Calibration:
    """
    Closed form + refinement; then images with RMS > outlier_factor x median RMS are removed
    (a blurred or mis-detected board) and the refinement is repeated once.
    """
    if len(detections) < 3:
        raise RuntimeError(f"The board was found in only {len(detections)} images; at least 3 are needed.")
    object_xy = board.object_points()[:, :2].astype(np.float64)
    K_cf, _ = calibrate_closed_form(object_xy, [d.corners for d in detections])

    K, dist, rvecs, tvecs, std = _refine(detections, board, size, K_cf, fit_k3)
    result = Calibration(size, K, dist, std, K_cf, detections, rvecs, tvecs, [], list(not_detected or []))

    rms = result.per_image_rms(board)
    keep = rms <= outlier_factor * np.median(rms)
    if not keep.all():
        kept = [d for d, k in zip(detections, keep) if k]
        K, dist, rvecs, tvecs, std = _refine(kept, board, size, K_cf, fit_k3)
        result = Calibration(size, K, dist, std, K_cf, kept, rvecs, tvecs,
                             [d.name for d, k in zip(detections, keep) if not k], result.not_detected)
    return result


def save_json(result: Calibration, board: Board, path: str, camera_name: str) -> Dict:
    """Write the calibration in the format read by the Bundle Adjustment project."""
    K, d = result.K, result.dist
    rms_per_image = result.per_image_rms(board)
    data = {
        "camera": camera_name,
        "image_width": int(result.image_size[0]),
        "image_height": int(result.image_size[1]),
        "camera_matrix_K": K.tolist(),
        "distortion_model": "OpenCV [k1, k2, p1, p2, k3]" + ("" if result.std["k3"] > 0 else ", k3 fixed at 0"),
        "distortion_coefficients": d.tolist(),
        "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "std_deviation": result.std,
        "rms_reprojection_error_px": result.rms(board),
        "closed_form_K": result.K_closed_form.tolist(),
        "board": {"inner_corners": list(board.inner_corners), "square_size_m": board.square_size},
        "images_used": {det.name: round(float(e), 4) for det, e in zip(result.detections, rms_per_image)},
        "images_rejected": result.rejected,
        "images_without_board": result.not_detected,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data

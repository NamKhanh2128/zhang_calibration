"""
Checkerboard geometry and corner detection.

Detection runs in two stages so that 24-megapixel photos stay fast:
  1. cv2.findChessboardCornersSB on a down-scaled copy finds the grid;
  2. cv2.cornerSubPix refines every corner on the full-resolution image, with a
     search window proportional to the size of one square in the photo.

Author: Nguyen Duc Nam Khanh
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

DETECTION_MAX_SIDE = 1600   # longest image side used for the coarse grid search (px)


@dataclass(frozen=True)
class Board:
    """Planar checkerboard: `inner_corners` = (columns, rows) of inner corners, `square_size` in metres."""
    inner_corners: Tuple[int, int]
    square_size: float

    @property
    def num_corners(self) -> int:
        return self.inner_corners[0] * self.inner_corners[1]

    def object_points(self) -> np.ndarray:
        """Corner coordinates (X, Y, 0) on the board plane, row by row, shape (N, 3), float32."""
        cols, rows = self.inner_corners
        xy = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size
        return np.hstack([xy, np.zeros((len(xy), 1))]).astype(np.float32)


def detect_corners(gray: np.ndarray, board: Board) -> Optional[np.ndarray]:
    """Sub-pixel corner positions (N, 2) in full-resolution pixels, or None if the board is not found."""
    scale = min(1.0, DETECTION_MAX_SIDE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, coarse = cv2.findChessboardCornersSB(small, board.inner_corners,
                                                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        return None
    corners = coarse.reshape(-1, 2) / scale

    # Search window: a quarter of the smallest square side, so it never reaches a neighbouring corner.
    cols, rows = board.inner_corners
    grid = corners.reshape(rows, cols, 2)
    side = min(np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
               np.linalg.norm(np.diff(grid, axis=1), axis=2).min())
    half_window = int(np.clip(0.25 * side, 3, 30))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    refined = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2).astype(np.float32),
                               (half_window, half_window), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float64)

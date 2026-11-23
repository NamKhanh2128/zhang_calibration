"""
Zhang's closed-form calibration (Z. Zhang, "A flexible new technique for camera
calibration", IEEE TPAMI 2000).

For a planar board (Z = 0) every image gives a homography  s [u v 1]^T = H [X Y 1]^T
with  H = [h1 h2 h3] = lambda K [r1 r2 t].  Because r1 and r2 are orthonormal,

    h1^T B h2 = 0,      h1^T B h1 = h2^T B h2,        B = K^-T K^-1 (symmetric),

two linear equations in the six unknowns b = [B11 B12 B22 B13 B23 B33] per image.
Stacking them gives V b = 0, solved by SVD; K follows from B (Zhang, Appendix B),
and each pose from K^-1 H.

Pixel coordinates are first normalised (centred, scaled to ~1) so that the linear
systems are well conditioned; the normalisation is undone at the end.

Author: Nguyen Duc Nam Khanh
"""

from typing import List, Tuple

import numpy as np


def normalization_matrix(points: np.ndarray) -> np.ndarray:
    """Similarity T that moves the centroid to 0 and the mean distance to sqrt(2) (Hartley)."""
    center = points.mean(axis=0)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(points - center, axis=1))
    return np.array([[scale, 0.0, -scale * center[0]],
                     [0.0, scale, -scale * center[1]],
                     [0.0, 0.0, 1.0]])


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Normalised Direct Linear Transform: H with  dst ~ H src  for (N, 2) point sets, N >= 4.
    Each correspondence gives two rows of A h = 0; h is the right singular vector of the
    smallest singular value.
    """
    T_src, T_dst = normalization_matrix(src), normalization_matrix(dst)
    s = (_to_homogeneous(src) @ T_src.T)[:, :2]
    d = (_to_homogeneous(dst) @ T_dst.T)[:, :2]
    zeros, ones = np.zeros(len(s)), np.ones(len(s))
    rows_u = np.stack([s[:, 0], s[:, 1], ones, zeros, zeros, zeros,
                       -d[:, 0] * s[:, 0], -d[:, 0] * s[:, 1], -d[:, 0]], axis=1)
    rows_v = np.stack([zeros, zeros, zeros, s[:, 0], s[:, 1], ones,
                       -d[:, 1] * s[:, 0], -d[:, 1] * s[:, 1], -d[:, 1]], axis=1)
    A = np.vstack([rows_u, rows_v])
    H_normalized = np.linalg.svd(A)[2][-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_normalized @ T_src
    return H / H[2, 2]


def _v(H: np.ndarray, i: int, j: int) -> np.ndarray:
    """Row vector v_ij with  h_i^T B h_j = v_ij . b  (h_i = column i of H)."""
    hi, hj = H[:, i], H[:, j]
    return np.array([hi[0] * hj[0],
                     hi[0] * hj[1] + hi[1] * hj[0],
                     hi[1] * hj[1],
                     hi[2] * hj[0] + hi[0] * hj[2],
                     hi[2] * hj[1] + hi[1] * hj[2],
                     hi[2] * hj[2]])


def intrinsics_from_homographies(homographies: List[np.ndarray]) -> np.ndarray:
    """K (3x3, with skew) from >= 3 board homographies."""
    if len(homographies) < 3:
        raise ValueError("Zhang's method needs at least 3 views of the board")
    V = np.vstack([np.vstack([_v(H, 0, 1), _v(H, 0, 0) - _v(H, 1, 1)]) for H in homographies])
    b = np.linalg.svd(V)[2][-1]
    if b[0] < 0:                       # B = K^-T K^-1 is positive definite up to the sign of b
        b = -b
    B11, B12, B22, B13, B23, B33 = b

    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha**2 / lam
    return np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])


def pose_from_homography(K: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Board pose (R, t) from  H = lambda K [r1 r2 t]:  lambda = 1 / ||K^-1 h1||, r3 = r1 x r2.
    The sign is chosen so the board lies in front of the camera (t_z > 0), and R is projected
    onto SO(3) with an SVD because noise makes [r1 r2 r3] only approximately orthonormal.
    """
    M = np.linalg.inv(K) @ H
    lam = 1.0 / np.linalg.norm(M[:, 0])
    if M[2, 2] * lam < 0:
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    return U @ Vt, t


def calibrate_closed_form(object_xy: np.ndarray, image_points: List[np.ndarray]):
    """
    Zhang's linear solution, without lens distortion.
    object_xy: (N, 2) board coordinates; image_points: list of (N, 2) detected corners.
    Returns K and the list of board poses (R, t).

    The pixel coordinates are pre-scaled by a common similarity N (image-independent), so
    the estimated K_n relates to the true K by K = N^-1 K_n.
    """
    N = normalization_matrix(np.vstack(image_points))
    homographies = [N @ homography_dlt(object_xy, pts) for pts in image_points]
    K = np.linalg.inv(N) @ intrinsics_from_homographies(homographies)
    K /= K[2, 2]
    poses = [pose_from_homography(K, np.linalg.inv(N) @ H) for H in homographies]
    return K, poses

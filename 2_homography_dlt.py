"""
Bước 2 (phụ): DLT – Direct Linear Transform.
Tính ma trận Homography H (3×3) từ N cặp điểm tương ứng.
Lý thuyết: Hartley & Zisserman, Chương 4.
"""

import numpy as np


def _normalize_points(pts):
    """
    Chuẩn hóa điểm trước khi DLT để cải thiện điều kiện số.
    Trả về pts_norm và ma trận biến đổi T.
    """
    centroid = pts.mean(axis=0)
    shifted  = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-10:
        mean_dist = 1.0
    scale = math.sqrt(2) / mean_dist

    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [    0, scale, -scale * centroid[1]],
        [    0,     0,                   1 ],
    ])
    ones    = np.ones((pts.shape[0], 1))
    pts_h   = np.hstack([pts, ones])            # (N, 3)
    pts_n   = (T @ pts_h.T).T[:, :2]           # (N, 2)
    return pts_n, T


import math   # noqa: E402  (import nhỏ để _normalize_points dùng math.sqrt)


def compute_homography(M_points, m_points):
    """
    Tính H dùng DLT chuẩn hóa (Normalized DLT).

    Tham số:
      M_points : (N, 2) – tọa độ mặt phẳng 3D (Z=0)
      m_points : (N, 2) – tọa độ pixel 2D trên ảnh

    Trả về H (3, 3), đã chuẩn hóa H[2,2] = 1.
    """
    assert len(M_points) == len(m_points) >= 4, \
        "Cần ít nhất 4 cặp điểm tương ứng"
    N = len(M_points)

    # Chuẩn hóa
    M_n, T_M = _normalize_points(np.array(M_points, dtype=np.float64))
    m_n, T_m = _normalize_points(np.array(m_points, dtype=np.float64))

    # Lập hệ phương trình A h = 0
    A = []
    for i in range(N):
        X, Y = M_n[i, 0], M_n[i, 1]
        u, v = m_n[i, 0], m_n[i, 1]
        A.append([-X, -Y, -1,  0,  0,  0, u*X, u*Y, u])
        A.append([ 0,  0,  0, -X, -Y, -1, v*X, v*Y, v])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    # Khử chuẩn hóa: H = T_m^-1 @ H_n @ T_M
    H = np.linalg.inv(T_m) @ H_n @ T_M

    # Chuẩn hóa H[2,2] = 1
    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]

    return H

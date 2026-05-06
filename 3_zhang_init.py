"""
Bước 4: Nghiệm đóng của Zhang – Closed-Form Initialization.
Lý thuyết: Zhang (2000), Mục 3.1 & Phụ lục B.

Quy trình:
  1. Từ mỗi H, lập 2 phương trình ràng buộc cho B = K^{-T} K^{-1}.
  2. Gom N ảnh → giải Vb = 0 bằng SVD.
  3. Rút trích K từ B bằng phân tích đại số (công thức Zhang).
  4. Tính R, t cho từng ảnh.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lập vector v_ij (công thức Zhang, Phụ lục B)
# ─────────────────────────────────────────────────────────────────────────────

def _v_ij(H, i, j):
    """
    Vector 6 chiều v_{ij} từ cột i và j của H (0-indexed).
    B được biểu diễn là vec b = [B11, B12, B22, B13, B23, B33].
    Ràng buộc: h_i^T B h_j = v_{ij}^T b.
    """
    hi = H[:, i]
    hj = H[:, j]
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rút trích K từ vector b = [B11, B12, B22, B13, B23, B33]
# ─────────────────────────────────────────────────────────────────────────────

def _extract_intrinsics(b):
    """
    Rút trích K từ vector b theo đúng công thức đại số của Zhang (2000).
    Nếu B11 < 0, đảo dấu toàn bộ để đảm bảo B xác định dương.
    """
    B11, B12, B22, B13, B23, B33 = b

    # B phải xác định dương → B11 > 0
    if B11 < 0:
        B11, B12, B22, B13, B23, B33 = -B11, -B12, -B22, -B13, -B23, -B33

    denom = B11 * B22 - B12 ** 2
    if abs(denom) < 1e-12:
        raise ValueError("B11*B22 - B12^2 ≈ 0: hệ phương trình suy biến.")

    v0    = (B12 * B13 - B11 * B23) / denom
    lam   = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(abs(lam / B11))
    beta  = np.sqrt(abs(lam * B11 / denom))
    gamma = -B12 * alpha ** 2 * beta / lam
    u0    = gamma * v0 / beta - B13 * alpha ** 2 / lam

    K = np.array([
        [alpha, gamma, u0],
        [    0,  beta, v0],
        [    0,     0,  1],
    ])
    return K


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tách R, t từ H và K
# ─────────────────────────────────────────────────────────────────────────────

def _extract_extrinsics(H, K):
    """
    Từ H = [h1 h2 h3] và K, tính R = [r1 r2 r3], t.
    Chuẩn hóa bằng SVD để ép R thành ma trận quay hợp lệ.
    """
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    # Tỉ lệ λ
    lam = 1.0 / np.linalg.norm(K_inv @ h1)

    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    t  = lam * (K_inv @ h3)

    # Ép R về SO(3) bằng SVD
    R_raw = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    # Đảm bảo det = +1 (không phải phản xạ)
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt

    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hàm công khai
# ─────────────────────────────────────────────────────────────────────────────

def zhang_init(H_list):
    """
    Nghiệm đóng của Zhang.

    Tham số:
      H_list : list of (3,3) homography matrices (N ảnh)

    Trả về:
      K       : (3,3) ma trận nội
      R_list  : list of (3,3)
      t_list  : list of (3,)
    """
    if len(H_list) < 3:
        raise ValueError("Cần ít nhất 3 ảnh để xác định duy nhất K.")

    # ── 4.1 Lập hệ Vb = 0 ──
    rows = []
    for H in H_list:
        v12 = _v_ij(H, 0, 1)
        v11 = _v_ij(H, 0, 0)
        v22 = _v_ij(H, 1, 1)
        rows.append(v12)            # h1^T B h2 = 0
        rows.append(v11 - v22)      # h1^T B h1 = h2^T B h2

    V = np.array(rows)              # (2N, 6)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]                      # vector b ứng với singular value nhỏ nhất

    # ── 4.2 Rút trích K ──
    K = _extract_intrinsics(b)

    # ── 4.3 Rút trích R, t ──
    R_list, t_list = [], []
    for H in H_list:
        R, t = _extract_extrinsics(H, K)
        R_list.append(R)
        t_list.append(t)

    return K, R_list, t_list

"""
Bước 5 & 6: Tối ưu hóa Levenberg-Marquardt + Khử méo ảnh.

Bước 5 – Bundle Adjustment:
  Mô hình méo Brown: x_dist = x(1 + k1*r² + k2*r⁴)
  Hàm mục tiêu: tổng bình phương reprojection error trên tất cả ảnh.
  Dùng scipy.optimize.least_squares(method='lm').

Bước 6 – Undistortion (Inverse Mapping):
  Với mỗi pixel (u', v') trên ảnh đích trống:
    1. x_n = (u' - cx) / fx,  y_n = (v' - cy) / fy
    2. r² = x_n² + y_n²
    3. x_d = x_n (1 + k1 r² + k2 r⁴)   ← ánh xạ ngược
    4. u_src = fx * x_d + cx,  v_src = fy * y_d + cy
    5. Nội suy song tuyến tính tại (u_src, v_src) trên ảnh gốc bị méo.
  Dùng cv2.remap để vectorize bước nội suy.
"""

import numpy as np
import cv2
import os
import glob
from scipy.optimize import least_squares


# ─────────────────────────────────────────────────────────────────────────────
# ĐÓNG GÓI / MỞ GÓI THAM SỐ
# ─────────────────────────────────────────────────────────────────────────────

def pack_params(K, R_list, t_list, k1=0.0, k2=0.0):
    """
    Đóng gói tất cả tham số vào một vector 1-D để truyền cho LM.

    Layout: [fx, fy, cx, cy, k1, k2,
             rvec0(3), t0(3),
             rvec1(3), t1(3), ...]
    """
    params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2]
    for R, t in zip(R_list, t_list):
        rvec, _ = cv2.Rodrigues(R)
        params.extend(rvec.flatten().tolist())
        params.extend(np.array(t).flatten().tolist())
    return np.array(params, dtype=np.float64)


def unpack_params(params, num_images):
    """Mở gói vector tham số → K, D=[k1,k2,...], R_list, t_list."""
    fx, fy, cx, cy, k1, k2 = params[:6]
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ])
    D = np.array([k1, k2, 0.0, 0.0, 0.0])

    R_list, t_list = [], []
    off = 6
    for _ in range(num_images):
        rvec = params[off:off+3]
        t    = params[off+3:off+6]
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        R_list.append(R)
        t_list.append(t.reshape(3, 1))
        off += 6

    return K, D, R_list, t_list


# ─────────────────────────────────────────────────────────────────────────────
# CHIẾU ĐIỂM (PROJECTION)
# ─────────────────────────────────────────────────────────────────────────────

def project_points(M_pts, K, R, t, k1, k2):
    """
    Chiếu N điểm 3D (Z=0) lên ảnh có méo Brown.

    Tham số:
      M_pts : (N, 2) – tọa độ thế giới (X, Y, Z=0)
      K, R, t, k1, k2 : tham số camera

    Trả về (N, 2) – tọa độ pixel dự đoán.
    """
    N = M_pts.shape[0]
    # Gắn Z=0, chuyển sang tọa độ camera
    M_3d = np.hstack([M_pts, np.zeros((N, 1))])     # (N, 3)
    P_cam = (R @ M_3d.T + t.reshape(3, 1))           # (3, N)

    # Tọa độ chuẩn hóa
    z = P_cam[2, :]
    x_n = P_cam[0, :] / z
    y_n = P_cam[1, :] / z

    # Méo Brown (xuyên tâm)
    r2 = x_n ** 2 + y_n ** 2
    dist = 1.0 + k1 * r2 + k2 * r2 ** 2
    x_d = x_n * dist
    y_d = y_n * dist

    # Áp dụng K (skew = 0 để đơn giản; nếu cần thêm gamma vào K)
    u = K[0, 0] * x_d + K[0, 1] * y_d + K[0, 2]
    v =                  K[1, 1] * y_d + K[1, 2]

    return np.vstack([u, v]).T                        # (N, 2)


# ─────────────────────────────────────────────────────────────────────────────
# HÀM MỤC TIÊU (RESIDUALS)
# ─────────────────────────────────────────────────────────────────────────────

def _residuals(params, M_squares, m_points_list):
    """
    Trả về vector residuals = [m_observed - m_projected] cho tất cả ảnh.
    scipy.least_squares sẽ tối thiểu hóa tổng bình phương vector này.
    """
    num_images = len(m_points_list)
    K, D, R_list, t_list = unpack_params(params, num_images)
    k1, k2 = D[0], D[1]

    errors = []
    for i in range(num_images):
        m_proj = project_points(M_squares, K, R_list[i], t_list[i], k1, k2)
        errors.append((m_points_list[i] - m_proj).flatten())

    return np.concatenate(errors)


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: KHỬ MÉO – INVERSE MAPPING + BILINEAR INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_undistort_maps(K, k1, k2, img_shape):
    """
    Xây dựng bảng ánh xạ (map_x, map_y) theo phương pháp Inverse Mapping:

    Với mỗi pixel (u', v') trên ảnh phẳng đích:
      1. Chuẩn hóa: x_n = (u' - cx)/fx,  y_n = (v' - cy)/fy
      2. r² = x_n² + y_n²
      3. x_d = x_n(1 + k1 r² + k2 r⁴)   ← áp dụng méo ngược
      4. u_src = fx * x_d + cx
         v_src = fy * y_d + cy
    """
    h, w = img_shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Lưới pixel đích
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))

    # Tọa độ chuẩn hóa của ảnh phẳng đích
    x_n = (u_grid - cx) / fx
    y_n = (v_grid - cy) / fy

    # Áp dụng méo Brown để tìm vị trí tương ứng trên ảnh gốc bị méo
    r2   = x_n ** 2 + y_n ** 2
    dist = 1.0 + k1 * r2 + k2 * r2 ** 2
    x_d  = x_n * dist
    y_d  = y_n * dist

    # Tọa độ pixel nguồn (trên ảnh bị méo)
    map_x = (fx * x_d + cx).astype(np.float32)
    map_y = (fy * y_d + cy).astype(np.float32)

    return map_x, map_y


def undistort_images(K, k1, k2, input_dir, output_dir):
    """
    Khử méo tất cả ảnh trong input_dir và lưu vào output_dir.
    Dùng Bilinear Interpolation (cv2.INTER_LINEAR) qua cv2.remap.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(input_dir, "*.*"))

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        map_x, map_y = _build_undistort_maps(K, k1, k2, img.shape)

        # cv2.remap thực hiện bilinear interpolation (INTER_LINEAR)
        undistorted = cv2.remap(img, map_x, map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, undistorted)
        print(f"  Đã khử méo: {os.path.basename(img_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# HÀM CÔNG KHAI CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

def optimize_and_undistort(K_init, R_list, t_list,
                           M_squares, points_2d_list,
                           input_dir, output_dir):
    """
    Tối ưu hóa LM toàn cục + khử méo ảnh.

    Tham số:
      K_init        : (3,3) ma trận nội khởi tạo (từ Zhang closed-form)
      R_list, t_list: danh sách ma trận ngoại
      M_squares     : (N, 2) tọa độ thế giới của góc vuông
      points_2d_list: list of (N, 2) tọa độ pixel quan sát
      input_dir     : thư mục ảnh gốc để khử méo
      output_dir    : thư mục xuất ảnh đã khử méo

    Trả về:
      K_opt, R_opt_list, t_opt_list, D_opt, rmse
    """
    # ── Bước 5a: Khởi tạo vector tham số ──
    params0 = pack_params(K_init, R_list, t_list, k1=0.0, k2=0.0)

    # ── Bước 5b: Tối ưu LM ──
    print("  Bắt đầu Levenberg-Marquardt...")
    result = least_squares(
        _residuals, params0,
        method='lm',
        args=(M_squares, points_2d_list),
        max_nfev=2000,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )

    K_opt, D_opt, R_opt_list, t_opt_list = unpack_params(result.x, len(R_list))
    k1_opt, k2_opt = D_opt[0], D_opt[1]

    # ── Bước 5c: Tính RMSE ──
    err_vec = _residuals(result.x, M_squares, points_2d_list)
    rmse    = float(np.sqrt(np.mean(err_vec ** 2)))

    print(f"  LM hội tụ sau {result.nfev} lần đánh giá | RMSE = {rmse:.4f} px")

    # ── Bước 6: Khử méo ──
    print("  Đang khử méo ảnh...")
    undistort_images(K_opt, k1_opt, k2_opt, input_dir, output_dir)

    return K_opt, R_opt_list, t_opt_list, D_opt, rmse

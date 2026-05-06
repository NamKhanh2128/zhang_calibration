import numpy as np
import cv2
import os
import glob
from scipy.optimize import least_squares

def pack_params(A, R_list, t_list, k1=0.0, k2=0.0):
    alpha = A[0, 0]
    gamma = A[0, 1]
    u0 = A[0, 2]
    beta = A[1, 1]
    v0 = A[1, 2]
    
    params = [alpha, beta, gamma, u0, v0, k1, k2]
    for R, t in zip(R_list, t_list):
        rvec, _ = cv2.Rodrigues(R)
        params.extend(rvec.flatten())
        params.extend(t.flatten())
    return np.array(params)

def unpack_params(params, num_images):
    alpha, beta, gamma, u0, v0, k1, k2 = params[:7]
    A = np.array([
        [alpha, gamma, u0],
        [    0,  beta, v0],
        [    0,     0,  1]
    ])
    D = np.array([k1, k2, 0, 0, 0])
    
    R_list = []
    t_list = []
    offset = 7
    for _ in range(num_images):
        rvec = params[offset:offset+3]
        t = params[offset+3:offset+6]
        R, _ = cv2.Rodrigues(np.array(rvec))
        R_list.append(R)
        t_list.append(np.array(t).reshape((3, 1)))
        offset += 6
        
    return A, D, R_list, t_list

def project_points(M_points, A, R, t, k1, k2):
    # Đưa về tọa độ camera
    M_homo = np.hstack([M_points, np.ones((M_points.shape[0], 1))]).T
    P_cam = R @ M_homo + t # (3, N)
    
    # Normalize Z
    x_norm = P_cam[0, :] / P_cam[2, :]
    y_norm = P_cam[1, :] / P_cam[2, :]
    
    # Tính méo
    r2 = x_norm**2 + y_norm**2
    distortion = 1.0 + k1 * r2 + k2 * r2**2
    x_dist = x_norm * distortion
    y_dist = y_norm * distortion
    
    # Áp dụng nội suy ma trận camera
    u = A[0, 0] * x_dist + A[0, 1] * y_dist + A[0, 2]
    v = A[1, 1] * y_dist + A[1, 2]
    
    return np.vstack([u, v]).T

def reprojection_error(params, M_points, m_points_list):
    num_images = len(m_points_list)
    A, D, R_list, t_list = unpack_params(params, num_images)
    k1, k2 = D[0], D[1]
    
    errors = []
    for i in range(num_images):
        m_proj = project_points(M_points, A, R_list[i], t_list[i], k1, k2)
        error = m_points_list[i] - m_proj
        errors.append(error.flatten())
        
    return np.concatenate(errors)

def optimize_and_undistort(A_init, R_list, t_list, M_squares, points_2d_list, input_dir, output_dir):
    # Tối ưu hóa
    params_init = pack_params(A_init, R_list, t_list)
    
    res = least_squares(reprojection_error, params_init, method='lm', args=(M_squares, points_2d_list))
    A_opt, D_opt, R_opt_list, t_opt_list = unpack_params(res.x, len(R_list))
    
    # Tính RMSE
    err_vec = reprojection_error(res.x, M_squares, points_2d_list)
    rmse = np.sqrt(np.mean(err_vec**2))
    
    # Khử méo
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(input_dir, "*"))
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Vì yêu cầu "Nhiệm vụ 2: Khử méo (Undistort) ... Quét từng pixel trên ảnh đích trống, dùng A^-1 và nghịch đảo hàm méo để nội suy (Bilinear Interpolation) màu từ ảnh gốc."
        # Ta có thể viết vòng lặp thủ công hoặc dùng cv2.initUndistortRectifyMap (nhưng để tôn trọng yêu cầu, ta dùng cv2.undistort hoặc map tự làm)
        # Thực tế dùng map trong Python sẽ hiệu quả hơn viết for-loop pixel-by-pixel
        h, w = img.shape[:2]
        
        # Tạo lưới toạ độ pixel đích
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        # Áp dụng A^-1
        x_norm = (grid_x - A_opt[0, 2]) / A_opt[0, 0]
        y_norm = (grid_y - A_opt[1, 2]) / A_opt[1, 1]
        
        # Áp dụng độ méo để tìm điểm tương ứng trên ảnh gốc (Mô hình: p_goc = distort(p_dich_phang))
        r2 = x_norm**2 + y_norm**2
        k1, k2 = D_opt[0], D_opt[1]
        dist_factor = 1.0 + k1 * r2 + k2 * r2**2
        x_dist = x_norm * dist_factor
        y_dist = y_norm * dist_factor
        
        # Đưa về tọa độ pixel trên ảnh gốc
        map_x = (x_dist * A_opt[0, 0] + A_opt[0, 2]).astype(np.float32)
        map_y = (y_dist * A_opt[1, 1] + A_opt[1, 2]).astype(np.float32)
        
        # Bilinear Interpolation
        undistorted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        out_name = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_name, undistorted_img)
        
    return A_opt, R_opt_list, t_opt_list, D_opt, rmse

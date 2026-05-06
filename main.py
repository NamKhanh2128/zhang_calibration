import numpy as np
import cv2
import os
import glob

import importlib

feature_extractor = importlib.import_module("1_feature_extractor")
process_image = feature_extractor.process_image

homography_dlt = importlib.import_module("2_homography_dlt")
compute_homography = homography_dlt.compute_homography

zhang_init_mod = importlib.import_module("3_zhang_init")
zhang_init = zhang_init_mod.zhang_init

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

optimizer = importlib.import_module("4_optimizer")
optimize_and_undistort = optimizer.optimize_and_undistort

def generate_3d_points(square_size, circle_distance, board_cols=11, board_rows=8):
    # Tạo M_circles (9 điểm tròn)
    # Gốc P0 là (0,0)
    # Trục X từ trái sang phải, Y từ trên xuống
    M_circles = []
    for i in range(3):      # 3 hàng
        for j in range(3):  # 3 cột
            M_circles.append([j * circle_distance, i * circle_distance])
    M_circles = np.array(M_circles, dtype=np.float32)
    
    # Tạo M_squares
    # Quét từ trái sang phải, trên xuống dưới
    M_squares = []
    for i in range(board_rows):
        for j in range(board_cols):
            M_squares.append([j * square_size, i * square_size])
    M_squares = np.array(M_squares, dtype=np.float32)
    
    return M_circles, M_squares

def main():
    square_size = 20.0
    circle_distance = 60.0
    
    M_circles, M_squares = generate_3d_points(square_size, circle_distance)
    
    input_dir = os.path.join('dataset', 'input', 'boardchecker_img')
    pointed_dir = os.path.join('dataset', 'output', 'pointed_boardchecker')
    norm_img_dir = os.path.join('dataset', 'input', 'norm_img')
    uncalib_dir = os.path.join('dataset', 'output', 'uncalib_img')
    
    image_paths = glob.glob(os.path.join(input_dir, '*.*'))
    
    H_list = []
    points_2d_list = []
    valid_images = []
    
    print(f"Đang xử lý {len(image_paths)} ảnh...")
    
    for img_path in image_paths:
        # Bước 1 & 2: Trích xuất đặc trưng và dự đoán góc vuông
        board_size = (11, 8)
        refined_corners = process_image(img_path, M_circles, M_squares, compute_homography, pointed_dir, board_size)
        if refined_corners is not None:
            points_2d_list.append(refined_corners)
            valid_images.append(img_path)
            
            # Tính Homography chính xác H từ M_squares và corners
            H = compute_homography(M_squares, refined_corners)
            H_list.append(H)
            
    if not H_list:
        print("Không tìm thấy dữ liệu hợp lệ trên bất kỳ ảnh nào!")
        return
        
    print(f"Trích xuất thành công trên {len(valid_images)} ảnh.")
    
    # Bước 4: Lời giải đóng (Khởi tạo Zhang)
    A_init, R_init_list, t_init_list = zhang_init(H_list)
    
    # Bước 5: Tối ưu hóa LM và Khử méo
    print("Đang chạy tối ưu hóa Levenberg-Marquardt...")
    A_opt, R_opt_list, t_opt_list, D_opt, rmse = optimize_and_undistort(
        A_init, R_init_list, t_init_list, M_squares, points_2d_list, norm_img_dir, uncalib_dir
    )
    
    print("\n--- KẾT QUẢ CALIBRATION ---")
    print("Ma trận Nội K (A_opt):")
    print(A_opt)
    print("\nHệ số méo D (k1, k2):")
    print(D_opt[:2])
    print("\nSai số Reprojection RMSE:")
    print(rmse)
    
    print("\nMa trận Ngoại (R, t) cho ảnh đầu tiên:")
    print("R1:")
    print(R_opt_list[0])
    print("t1:")
    print(t_opt_list[0])
    
if __name__ == "__main__":
    main()

"""
main.py – Điều phối toàn bộ pipeline Zhang Calibration (6 bước).

╔══════════════════════════════════════════════════════════════╗
║          CẤU HÌNH THAM SỐ BẢN HIỆU CHUẨN HYBRID             ║
╚══════════════════════════════════════════════════════════════╝

NHÓM 1 – Lưới Caro (Checkerboard)
  square_size  : cạnh ô vuông (mm)
  board_cols   : số cột góc vuông nội  (= số ô ngang - 1)
  board_rows   : số hàng góc vuông nội (= số ô dọc   - 1)

NHÓM 2 – Điểm tròn (Circular Markers)
  circle_pitch_x : khoảng cách tâm–tâm theo chiều X (mm)
  circle_pitch_y : khoảng cách tâm–tâm theo chiều Y (mm)
                   (bằng nhau nếu lưới tròn là hình vuông)

NHÓM 3 – Offset
  Gốc tọa độ (0, 0) = tâm điểm tròn P0 (đặc biệt nhất).
  Góc vuông đầu tiên của lưới caro (top-left) nằm cách P0 một khoảng:
  offset_x : theo chiều X (mm)
  offset_y : theo chiều Y (mm)

  ** Bạn bổ sung các giá trị thực đo được vào đây. **
"""

import numpy as np
import cv2
import os
import glob
import sys
import io
import importlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Import các module pipeline ──
feature_extractor   = importlib.import_module("1_feature_extractor")
process_image       = feature_extractor.process_image

homography_dlt      = importlib.import_module("2_homography_dlt")
compute_homography  = homography_dlt.compute_homography

zhang_init_mod      = importlib.import_module("3_zhang_init")
zhang_init          = zhang_init_mod.zhang_init

optimizer_mod       = importlib.import_module("4_optimizer")
optimize_and_undistort = optimizer_mod.optimize_and_undistort


# ══════════════════════════════════════════════════════════════════════════════
# THAM SỐ CẦN ĐIỀN
# ══════════════════════════════════════════════════════════════════════════════

# Nhóm 1: Lưới Caro
SQUARE_SIZE  = 20.0   # mm  ← đổi sang kích thước thực đo
BOARD_COLS   = 11     # số góc vuông theo chiều ngang
BOARD_ROWS   = 8      # số góc vuông theo chiều dọc

# Nhóm 2: Điểm tròn
CIRCLE_PITCH_X = 60.0   # mm  ← đổi sang khoảng cách thực đo
CIRCLE_PITCH_Y = 60.0   # mm

# Nhóm 3: Offset từ P0 đến góc vuông đầu tiên
OFFSET_X = 0.0   # mm  ← BỔ SUNG SAU
OFFSET_Y = 0.0   # mm  ← BỔ SUNG SAU

# ── Thư mục dữ liệu ──
INPUT_BOARD_DIR  = os.path.join('dataset', 'input',  'boardchecker_img')
OUTPUT_POINTED   = os.path.join('dataset', 'output', 'pointed_boardchecker')
INPUT_NORM_DIR   = os.path.join('dataset', 'input',  'norm_img')
OUTPUT_UNDIST    = os.path.join('dataset', 'output', 'uncalib_img')


# ══════════════════════════════════════════════════════════════════════════════
# TẠO TỌA ĐỘ 3D
# ══════════════════════════════════════════════════════════════════════════════

def generate_world_points(square_size, circle_pitch_x, circle_pitch_y,
                          offset_x, offset_y, board_cols, board_rows):
    """
    Tạo hai mảng tọa độ thế giới thực (Z=0).

    M_circles (9 điểm, 3×3):
      Gốc P0 = (0, 0).
      P0..P2: hàng trên  (Y = 0)
      P3..P5: hàng giữa  (Y = pitch_y)
      P6..P8: hàng dưới  (Y = 2*pitch_y)

    M_squares (board_rows × board_cols điểm):
      Góc vuông đầu tiên = (offset_x, offset_y).
      Tăng dần theo X rồi Y.
    """
    # ── Điểm tròn ──
    M_circles = np.array(
        [[j * circle_pitch_x, i * circle_pitch_y]
         for i in range(3) for j in range(3)],
        dtype=np.float32,
    )

    # ── Góc vuông ──
    M_squares = np.array(
        [[offset_x + j * square_size, offset_y + i * square_size]
         for i in range(board_rows) for j in range(board_cols)],
        dtype=np.float32,
    )

    return M_circles, M_squares


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def main():
    M_circles, M_squares = generate_world_points(
        square_size    = SQUARE_SIZE,
        circle_pitch_x = CIRCLE_PITCH_X,
        circle_pitch_y = CIRCLE_PITCH_Y,
        offset_x       = OFFSET_X,
        offset_y       = OFFSET_Y,
        board_cols     = BOARD_COLS,
        board_rows     = BOARD_ROWS,
    )

    image_paths = sorted(glob.glob(os.path.join(INPUT_BOARD_DIR, '*.*')))
    print(f"Tìm thấy {len(image_paths)} ảnh trong '{INPUT_BOARD_DIR}'.")

    H_list          = []
    points_2d_list  = []
    valid_images    = []

    # ── Bước 1–3: Trích xuất đặc trưng ──
    print("\n[Bước 1-3] Trích xuất đặc trưng & chốt góc vuông sub-pixel...")
    for img_path in image_paths:
        refined_corners = process_image(
            img_path, M_circles, M_squares,
            compute_homography,
            OUTPUT_POINTED,
            board_size=(BOARD_COLS, BOARD_ROWS),
        )
        if refined_corners is None:
            continue

        # H chính xác từ M_squares ↔ refined_corners (dùng cho Zhang)
        H = compute_homography(M_squares, refined_corners)
        H_list.append(H)
        points_2d_list.append(refined_corners)
        valid_images.append(img_path)

    print(f"  → Thành công trên {len(valid_images)} / {len(image_paths)} ảnh.")

    if len(H_list) < 3:
        print("Lỗi: Cần ít nhất 3 ảnh hợp lệ để hiệu chuẩn.")
        return

    # ── Bước 4: Nghiệm đóng Zhang ──
    print("\n[Bước 4] Tính nghiệm đóng Zhang (Closed-Form)...")
    K_init, R_init_list, t_init_list = zhang_init(H_list)
    print("  K khởi tạo:")
    print(K_init)

    # ── Bước 5 & 6: Tối ưu LM + Khử méo ──
    print("\n[Bước 5-6] Tối ưu Levenberg-Marquardt & Khử méo ảnh...")
    K_opt, R_opt_list, t_opt_list, D_opt, rmse = optimize_and_undistort(
        K_init, R_init_list, t_init_list,
        M_squares, points_2d_list,
        INPUT_NORM_DIR, OUTPUT_UNDIST,
    )

    # ── Kết quả ──
    print("\n" + "=" * 55)
    print("KẾT QUẢ HIỆU CHUẨN")
    print("=" * 55)
    print("\nMa trận Nội K (tối ưu):")
    print(K_opt)
    print(f"\nHệ số méo:  k1 = {D_opt[0]:.6f},  k2 = {D_opt[1]:.6f}")
    print(f"\nReprojection RMSE = {rmse:.4f} pixel")
    print("\nMa trận Ngoại ảnh đầu tiên:")
    print("  R =")
    print(R_opt_list[0])
    print("  t =", t_opt_list[0].flatten())


if __name__ == "__main__":
    main()

import os
import sys
import io

# Mục đích: Đảm bảo terminal Windows hiển thị được tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

from calibrate_helper import Calibrator


def main():
    # Mục đích: Cấu hình các đường dẫn thư mục lưu trữ ảnh
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pic_dir = os.path.join(base_dir, "pic")

    img_calib_dir = os.path.join(pic_dir, "RGB_camera_calib_img")
    img_corners_dir = os.path.join(pic_dir, "RGB_camera_calib_img_corners")
    img_normal_dir = os.path.join(pic_dir, "RGB_camera_normal_img")
    img_normal_undist_dir = os.path.join(pic_dir, "RGB_camera_normal_img_undistorted")

    # Mục đích: Tạo các thư mục nếu chưa tồn tại
    for d in [img_calib_dir, img_corners_dir, img_normal_dir, img_normal_undist_dir]:
        os.makedirs(d, exist_ok=True)

    # Mục đích: Định nghĩa thông số bàn cờ calibration
    shape_inner = (11, 8)
    square_size = 0.017

    # Mục đích: Khởi tạo đối tượng Calibrator
    calibrator = Calibrator(
        img_dir=img_calib_dir,
        shape_inner_corner=shape_inner,
        size_grid=square_size,
        visualization=False
    )

    # Mục đích: Thực hiện Calibration và lưu ảnh corners
    print("\n[START] BẮT ĐẦU QUÁ TRÌNH CALIBRATION...")
    calibrator.calibrate_camera(save_corners_dir=img_corners_dir)

    # Mục đích: Undistort ảnh thường
    if os.path.exists(img_normal_dir) and os.listdir(img_normal_dir):
        calibrator.undistort_images(img_normal_dir, img_normal_undist_dir)
    else:
        print(f"\n[WARN] Thư mục {img_normal_dir} trống hoặc không tồn tại. Bỏ qua undistort ảnh thường.")


if __name__ == "__main__":
    main()
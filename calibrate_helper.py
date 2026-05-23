import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Calibrator:

    # ==========================================================
    def __init__(
        self,
        img_dir,
        shape_inner_corner,
        size_grid,
        visualization=False,
        auto_resize=True
    ):

        self.img_dir = img_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.auto_resize = auto_resize

        self.mat_intri = None
        self.coff_dis = None
        self.v_rot = None
        self.v_trans = None

        self.used_img_paths = []

        # Mục đích: Tính toán tọa độ thế giới (Object points)
        # Công thức: P_w = (X, Y, 0) với X,Y theo số hàng/cột * kích thước ô
        w, h = shape_inner_corner

        self.cp_world = np.zeros((w * h, 3), np.float32)

        self.cp_world[:, :2] = (
            np.mgrid[0:w, 0:h]
            .T
            .reshape(-1, 2)
        )

        self.cp_world *= size_grid

        # Mục đích: Load đường dẫn ảnh

        self.img_paths = []

        for ext in ["jpg", "jpeg", "png", "bmp"]:

            self.img_paths.extend(
                glob.glob(
                    os.path.join(img_dir, f"*.{ext}")
                )
            )

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh trong {img_dir}"
            )

    # ==========================================================
    @staticmethod
    def section(txt):

        print("\n" + "=" * 72)
        print(" ", txt)
        print("=" * 72)

    # ==========================================================
    def detect_corners(self, gray):

        w, h = self.shape_inner_corner

        # Mục đích: Tìm vị trí góc trên bàn cờ với độ chính xác pixel
        flags = (
            cv2.CALIB_CB_EXHAUSTIVE |
            cv2.CALIB_CB_ACCURACY
        )

        ret, corners = cv2.findChessboardCornersSB(
            gray,
            (w, h),
            flags
        )

        if not ret:
            return False, None

        # Mục đích: Tinh chỉnh vị trí góc đạt độ chính xác sub-pixel

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        return True, corners

    # ==========================================================
    def calibrate_camera(
        self,
        save_corners_dir=None,
        draw_text=True,
        outlier_threshold=3.0
    ):

        w, h = self.shape_inner_corner

        points_world = []
        points_pixel = []

        self.used_img_paths = []

        img_size = None

        self.section("STEP 1: DETECT CHECKERBOARD")

        print(f"Images : {len(self.img_paths)}")
        print(f"Pattern: {w} x {h}")
        print(f"Grid   : {self.size_grid*1000:.1f} mm")

        # ======================================================
        # Detect checkerboard

        for path in self.img_paths:

            img = cv2.imread(path)

            if img is None:
                continue

            # ==================================================
            # First image defines target size

            if img_size is None:

                h0, w0 = img.shape[:2]
                img_size = (w0, h0)

            else:

                if (
                    img.shape[1] != img_size[0] or
                    img.shape[0] != img_size[1]
                ):

                    if self.auto_resize:

                        print(
                            f"[WARN] resize: "
                            f"{os.path.basename(path)} "
                            f"{img.shape[1]}x{img.shape[0]}"
                            f" -> "
                            f"{img_size[0]}x{img_size[1]}"
                        )

                        img = cv2.resize(
                            img,
                            img_size
                        )

                    else:

                        print(
                            f"[FAIL] skip size mismatch: "
                            f"{os.path.basename(path)}"
                        )

                        continue

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            t0 = time.time()

            ret, corners = self.detect_corners(gray)

            dt = (time.time() - t0) * 1000

            if ret:

                points_world.append(self.cp_world)
                points_pixel.append(corners)

                self.used_img_paths.append(path)

                print(
                    f"[OK] {os.path.basename(path):30s} "
                    f"{len(corners):3d} corners "
                    f"{dt:.1f} ms"
                )

            else:

                print(
                    f"[FAIL] {os.path.basename(path):30s} "
                    f"pattern not found"
                )

        # ======================================================
        if len(points_world) == 0:

            raise RuntimeError(
                "Không detect được checkerboard."
            )

        # ======================================================
        self.section("STEP 2: INITIAL CALIBRATION")

        t0 = time.time()

        rms, K, dist, rvecs, tvecs = (
            cv2.calibrateCamera(
                points_world,
                points_pixel,
                img_size,
                None,
                None
            )
        )

        print(f"Initial RMS: {rms:.4f} px")
        print(f"Time       : {time.time()-t0:.2f} s")

        # Mục đích: Loại bỏ ảnh có reprojection error trung bình cao hơn outlier_threshold (Outlier rejection)
        self.section("STEP 3: OUTLIER REJECTION")

        good_world = []
        good_pixel = []
        good_paths = []

        for i in range(len(points_world)):

            reproj, _ = cv2.projectPoints(
                points_world[i],
                rvecs[i],
                tvecs[i],
                K,
                dist
            )

            err = np.linalg.norm(
                points_pixel[i] - reproj,
                axis=2
            ).ravel()

            mean_err = np.mean(err)
            rms_err = np.sqrt(np.mean(err**2))
            max_err = np.max(err)

            fname = os.path.basename(
                self.used_img_paths[i]
            )

            print(
                f"{fname:30s} "
                f"mean={mean_err:.3f} "
                f"rms={rms_err:.3f} "
                f"max={max_err:.3f}"
            )

            if mean_err <= outlier_threshold:

                good_world.append(points_world[i])
                good_pixel.append(points_pixel[i])
                good_paths.append(
                    self.used_img_paths[i]
                )

            else:

                print(f"   -> rejected")

        if len(good_world) == 0:
            print("\n[WARN] CẢNH BÁO: Tất cả các ảnh đều bị loại bỏ vì sai số quá lớn (lớn hơn outlier_threshold).")
            print("Khôi phục lại toàn bộ ảnh để tiếp tục (Bỏ qua Outlier Rejection).")
            good_world = points_world
            good_pixel = points_pixel
            good_paths = self.used_img_paths

        # ======================================================
        self.section("STEP 4: FINAL CALIBRATION")

        rms, K, dist, rvecs, tvecs = (
            cv2.calibrateCamera(
                good_world,
                good_pixel,
                img_size,
                None,
                None
            )
        )

        self.mat_intri = K
        self.coff_dis = dist
        self.v_rot = rvecs
        self.v_trans = tvecs

        self.used_img_paths = good_paths

        # ======================================================
        print("\nCamera Matrix K:")
        print(K)

        print("\nDistortion:")
        print(dist.ravel())

        print(f"\nFinal RMS: {rms:.4f} px")

        print("\nIntrinsic:")
        print(f"fx={K[0,0]:.3f}")
        print(f"fy={K[1,1]:.3f}")
        print(f"cx={K[0,2]:.3f}")
        print(f"cy={K[1,2]:.3f}")

        # Mục đích: Tính toán và hiển thị sai số chiếu lại (Reprojection Error) cho từng ảnh
        # Công thức: Err = || P_pixel - Project(P_world, K, D, R, T) ||_2
        self.section("STEP 5: REPROJECTION ERROR")

        all_errors = []
        img_errors = [] # Lưu sai số trung bình mỗi ảnh cho đồ thị
        all_dx_dy = []  # Lưu sai số vector (dx, dy) cho scatter plot

        for i in range(len(good_world)):

            reproj, _ = cv2.projectPoints(
                good_world[i],
                rvecs[i],
                tvecs[i],
                K,
                dist
            )
            
            diff = good_pixel[i] - reproj
            dx = diff[:, 0, 0]
            dy = diff[:, 0, 1]
            all_dx_dy.append((dx, dy))

            err = np.linalg.norm(
                diff,
                axis=2
            ).ravel()

            all_errors.extend(err)

            mean_img_err = np.mean(err)
            img_errors.append(mean_img_err)

            print(
                f"{os.path.basename(good_paths[i]):30s} "
                f"mean={mean_img_err:.4f} "
                f"rms={np.sqrt(np.mean(err**2)):.4f} "
                f"max={np.max(err):.4f}"
            )

        all_errors = np.array(all_errors)

        print("\nTOTAL")
        print(f"Points = {len(all_errors)}")
        print(f"Mean   = {np.mean(all_errors):.4f}")
        print(f"RMS    = {np.sqrt(np.mean(all_errors**2)):.4f}")
        print(f"Max    = {np.max(all_errors):.4f}")

        # ======================================================
        bins = [0,0.1,0.2,0.5,1,2,5,np.inf]

        labels = [
            "<0.1",
            "0.1-0.2",
            "0.2-0.5",
            "0.5-1",
            "1-2",
            "2-5",
            ">5"
        ]

        hist, _ = np.histogram(
            all_errors,
            bins=bins
        )

        print("\nDistribution:")

        for lbl, cnt in zip(labels, hist):

            print(f"{lbl:10s}: {cnt}")

        # Mục đích: Vẽ đồ thị Reprojection Error (Hàm Loss) bằng Matplotlib 
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#1c1c1c')
        ax1.set_facecolor('#111111')
        ax2.set_facecolor('#111111')

        # --- 1. Bar Chart: Mean Reprojection Error per Image ---
        ax1.bar(range(1, len(img_errors) + 1), img_errors, color='deepskyblue', edgecolor='white', linewidth=0.5)
        overall_mean = np.mean(all_errors)
        ax1.axhline(y=overall_mean, color='gold', linestyle='--', label=f'Overall Mean Error: {overall_mean:.2f} pixels')
        ax1.set_title('Mean Reprojection Error per Image', fontweight='bold', color='lightgray')
        ax1.set_xlabel('Images', fontweight='bold', color='lightgray')
        ax1.set_ylabel('Mean Error in Pixels', fontweight='bold', color='lightgray')
        ax1.set_xticks(range(1, len(img_errors) + 1))
        ax1.legend(loc='lower center', frameon=True, facecolor='#1c1c1c', edgecolor='white')

        # --- 2. Scatter Plot: Reprojection Errors in Pixels ---
        for i, (dx, dy) in enumerate(all_dx_dy):
            ax2.scatter(dx, dy, marker='+', s=50, label=str(i+1), linewidth=1.5)
        
        ax2.set_title('Reprojection Errors in Pixels', fontweight='bold', color='lightgray')
        ax2.set_xlabel('X', fontweight='bold', color='lightgray')
        ax2.set_ylabel('Y', fontweight='bold', color='lightgray')
        if len(good_world) <= 30: # Only show legend if not too many images
            ax2.legend(loc='upper right', prop={'size': 9}, frameon=True, facecolor='#1c1c1c', edgecolor='white')
        ax2.grid(False)

        plt.tight_layout()
        print("Đang hiển thị đồ thị biểu diễn sai số...")
        plt.show()

        # Mục đích: Vẽ trực quan hóa kết quả calibration
        if save_corners_dir:

            self.section("STEP 6: SAVE VISUALIZATION")

            os.makedirs(
                save_corners_dir,
                exist_ok=True
            )

            axis_len = 3 * self.size_grid

            for i, path in enumerate(good_paths):

                img = cv2.imread(path)

                if img.shape[1] != img_size[0] or \
                   img.shape[0] != img_size[1]:

                    img = cv2.resize(
                        img,
                        img_size
                    )

                # ==============================================
                cv2.drawChessboardCorners(
                    img,
                    (w, h),
                    good_pixel[i],
                    True
                )

                cv2.drawFrameAxes(
                    img,
                    K,
                    dist,
                    rvecs[i],
                    tvecs[i],
                    axis_len,
                    3
                )

                # ==============================================
                if draw_text:

                    for idx, p in enumerate(
                        good_pixel[i]
                    ):

                        u, v = p.ravel()

                        txt = (
                            f"{idx}:"
                            f"({u:.0f},{v:.0f})"
                        )

                        cv2.putText(
                            img,
                            txt,
                            (int(u)+5, int(v)-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0,0,255),
                            1
                        )

                # ==============================================
                note = (
                    "Oxyz: "
                    "X(red) "
                    "Y(green) "
                    "Z(blue)"
                )

                cv2.putText(
                    img,
                    note,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2
                )

                out_path = os.path.join(
                    save_corners_dir,
                    os.path.basename(path)
                )

                cv2.imwrite(out_path, img)

                print(
                    f"[OK] {os.path.basename(path)}"
                )

                if self.visualization:

                    cv2.imshow(
                        "Calibration",
                        img
                    )

                    cv2.waitKey(300)

            cv2.destroyAllWindows()

        self.section("CALIBRATION DONE")

        return K, dist

    # ==========================================================
    def undistort_images(
        self,
        img_dir,
        save_dir,
        alpha=0
    ):

        if self.mat_intri is None:

            raise RuntimeError(
                "Run calibrate_camera() first."
            )

        paths = []

        for ext in ["jpg", "jpeg", "png", "bmp"]:

            paths.extend(
                glob.glob(
                    os.path.join(
                        img_dir,
                        f"*.{ext}"
                    )
                )
            )

        if len(paths) == 0:

            print("[WARN] no images")
            return

        os.makedirs(save_dir, exist_ok=True)

        sample = cv2.imread(paths[0])

        h, w = sample.shape[:2]

        newK, roi = (
            cv2.getOptimalNewCameraMatrix(
                self.mat_intri,
                self.coff_dis,
                (w, h),
                alpha,
                (w, h)
            )
        )

        self.section("UNDISTORT")

        print(f"ROI = {roi}")

        x, y, rw, rh = roi

        for path in paths:

            img = cv2.imread(path)

            dst = cv2.undistort(
                img,
                self.mat_intri,
                self.coff_dis,
                None,
                newK
            )

            # Mục đích: Cắt phần viền đen sau khi undistort
            dst = dst[
                y:y+rh,
                x:x+rw
            ]

            out = os.path.join(
                save_dir,
                os.path.basename(path)
            )

            cv2.imwrite(out, dst)

            print(
                f"✔ {os.path.basename(path)}"
            )

        print("\nDone.")
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
        
        save_plots_dir = None
        if save_corners_dir:
            save_plots_dir = os.path.dirname(save_corners_dir)
            os.makedirs(save_plots_dir, exist_ok=True)
            save_path = os.path.join(save_plots_dir, "reprojection_error.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✔ Đã lưu đồ thị biểu diễn sai số vào: {save_path}")
            
            # Đồng thời lưu vào thư mục gốc để dễ theo dõi
            root_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reprojection_error.png")
            try:
                plt.savefig(root_save_path, dpi=300, bbox_inches='tight')
                print(f"✔ Đã cập nhật đồ thị biểu diễn sai số ở thư mục gốc: {root_save_path}")
            except Exception:
                pass
                
        if self.visualization:
            plt.show()
        else:
            plt.close()

        # Mục đích: Vẽ trực quan hóa reprojection error trace-back lên ảnh & heatmap bàn cờ
        self.section("STEP 5b: REPROJECTION TRACE-BACK VISUALIZATION")

        # --- Thu thập dữ liệu reproj cho trace-back ---
        reproj_data = []
        for i in range(len(good_world)):
            reproj_pts, _ = cv2.projectPoints(
                good_world[i], rvecs[i], tvecs[i], K, dist
            )
            diff = good_pixel[i] - reproj_pts
            err  = np.linalg.norm(diff, axis=2).ravel()
            reproj_data.append({
                "path"       : good_paths[i],
                "detected"   : good_pixel[i],   # shape (N,1,2)
                "reproj"     : reproj_pts,        # shape (N,1,2)
                "err"        : err,               # shape (N,)
            })

        self.visualize_reprojection_on_image(reproj_data, img_size, save_dir=save_plots_dir)
        self.visualize_board_heatmap(reproj_data, save_dir=save_plots_dir)

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
    def visualize_reprojection_on_image(
        self,
        reproj_data,
        img_size,
        max_display=6,
        error_cap=None,
        save_dir=None
    ):
        """
        Vẽ trace-back reprojection error trực tiếp lên ảnh gốc bằng vector Quiver.

        Với mỗi góc bàn cờ:
          - Dấu cộng xanh (+) = vị trí reproj (điểm lý thuyết)
          - Mũi tên (→) = vector error phóng đại, màu thay đổi theo magnitude
            (xanh lá = nhỏ, vàng = trung bình, đỏ = lớn)
          - Có Quiver Key chú giải độ dài tỉ lệ thực tế.

        Chỉ hiển thị tối đa `max_display` ảnh trong plot chung,
        nhưng lưu toàn bộ ảnh đơn lẻ chất lượng cao cho TẤT CẢ các ảnh vào
        thư mục `pic/reprojection_traceback_individual/`.
        """

        # Sắp xếp ảnh từ error cao nhất → thấp nhất
        sorted_data = sorted(
            reproj_data,
            key=lambda d: np.mean(d["err"]),
            reverse=True
        )
        display_data = sorted_data[:max_display]

        # Xác định ngưỡng màu chung
        all_err = np.concatenate([d["err"] for d in reproj_data])
        vmax = error_cap if error_cap else np.percentile(all_err, 95)
        vmax = max(vmax, 1e-6)

        cmap = plt.cm.RdYlGn_r  # xanh lá → vàng → đỏ

        # --- PHẦN 1: LƯU TOÀN BỘ CÁC ẢNH ĐƠN LẺ CHO TẤT CẢ FILE CALIB ---
        if save_dir:
            indiv_dir = os.path.join(save_dir, "reprojection_traceback_individual")
            os.makedirs(indiv_dir, exist_ok=True)
            print(f"\n[INFO] Đang tạo và lưu ảnh trực quan đơn lẻ cho toàn bộ {len(reproj_data)} ảnh calib...")
            
            for data in reproj_data:
                fig_indiv, ax_indiv = plt.subplots(figsize=(12, 10))
                fig_indiv.patch.set_facecolor('#0d0d0d')
                ax_indiv.set_facecolor('#111111')
                
                img_bgr = cv2.imread(data["path"])
                if img_bgr is not None:
                    if img_bgr.shape[1] != img_size[0] or img_bgr.shape[0] != img_size[1]:
                        img_bgr = cv2.resize(img_bgr, img_size)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax_indiv.imshow(img_rgb, origin='upper')
                    
                    det_pts = data["detected"][:, 0, :]
                    rep_pts = data["reproj"][:, 0, :]
                    err_pts = data["err"]
                    
                    rx_i = rep_pts[:, 0]
                    ry_i = rep_pts[:, 1]
                    dx_i = det_pts[:, 0] - rx_i
                    dy_i = det_pts[:, 1] - ry_i
                    
                    # 1. Dấu cộng lý thuyết (xanh dương mảnh)
                    ax_indiv.scatter(
                        rx_i, ry_i, marker='+', s=16,
                        color='deepskyblue', linewidths=0.7, label='Reproj point'
                    )
                    
                    # 2. Vẽ quiver phóng đại lỗi (scale=0.03 nghĩa là 1 px lỗi = 33.3 px trên ảnh)
                    scale_val = 0.03
                    q_indiv = ax_indiv.quiver(
                        rx_i, ry_i, dx_i, dy_i, err_pts,
                        cmap=cmap, norm=plt.Normalize(0, vmax),
                        angles='xy', scale_units='xy', scale=scale_val,
                        width=0.003, headwidth=3.5, headlength=5
                    )
                    
                    # 3. Quiver Key chú thích tỉ lệ
                    ax_indiv.quiverkey(
                        q_indiv, X=0.88, Y=0.04, U=1.0,
                        label='Lệch 1.0 Pixel (phóng đại 33x)', labelpos='E',
                        coordinates='axes', color='tomato',
                        fontproperties={'size': 9, 'weight': 'bold'}
                    )
                    
                    fname_i = os.path.basename(data["path"])
                    mean_e_i = np.mean(err_pts)
                    max_e_i = np.max(err_pts)
                    ax_indiv.set_title(
                        f"{fname_i}\nMean: {mean_e_i:.3f} px | Max: {max_e_i:.3f} px",
                        fontsize=12, color='lightgray', fontweight='bold'
                    )
                    ax_indiv.axis('off')
                    
                    # Lưu file ảnh đơn lẻ
                    base_name_no_ext = os.path.splitext(fname_i)[0]
                    save_path_indiv = os.path.join(indiv_dir, f"{base_name_no_ext}_traceback.png")
                    plt.savefig(save_path_indiv, dpi=300, bbox_inches='tight')
                plt.close(fig_indiv)
            print(f"✔ Đã xuất xong toàn bộ ảnh đơn lẻ vào thư mục: {indiv_dir}")

        # --- PHẦN 2: TẠO HÌNH GHÉP TỔNG HỢP TOP 6 ---
        n = len(display_data)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols

        plt.style.use('dark_background')
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 7, rows * 5)
        )
        fig.patch.set_facecolor('#0d0d0d')
        fig.suptitle(
            'Reprojection Error — Trace-back trên ảnh calib (Top Lỗi Cao Nhất)\n'
            '+ xanh = vị trí reproj (lý thuyết) | → mũi tên = vector error (phóng đại)',
            fontsize=11, color='lightgray', y=1.01
        )

        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])

        for ax_row in axes:
            for ax in (ax_row if hasattr(ax_row, '__iter__') else [ax_row]):
                ax.axis('off')
                ax.set_facecolor('#111111')

        q_last = None
        for idx, data in enumerate(display_data):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.axis('on')

            # Đọc & resize ảnh
            img_bgr = cv2.imread(data["path"])
            if img_bgr is None:
                continue
            if img_bgr.shape[1] != img_size[0] or img_bgr.shape[0] != img_size[1]:
                img_bgr = cv2.resize(img_bgr, img_size)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ax.imshow(img_rgb, origin='upper')

            detected = data["detected"][:, 0, :]   # (N, 2)
            reproj   = data["reproj"][:, 0, :]     # (N, 2)
            err      = data["err"]                 # (N,)

            rx = reproj[:, 0]
            ry = reproj[:, 1]
            dx = detected[:, 0] - rx
            dy = detected[:, 1] - ry

            # 1. Điểm reproj lý thuyết (dấu cộng nhỏ mảnh)
            ax.scatter(rx, ry, marker='+', s=8, color='deepskyblue', linewidths=0.5)

            # 2. Vẽ quiver phóng đại lỗi (scale=0.04 nghĩa là 1 px lỗi = 25 px trên ảnh)
            q_scale = 0.04
            q = ax.quiver(
                rx, ry, dx, dy, err,
                cmap=cmap, norm=plt.Normalize(0, vmax),
                angles='xy', scale_units='xy', scale=q_scale,
                width=0.0035, headwidth=3.5, headlength=5
            )
            q_last = q

            # 3. Quiver Key chú thích tỉ lệ lỗi cho từng subplot
            ax.quiverkey(
                q, X=0.88, Y=0.05, U=1.0,
                label='1 px error', labelpos='E',
                coordinates='axes', color='tomato',
                fontproperties={'size': 7, 'weight': 'bold'}
            )

            fname = os.path.basename(data["path"])
            mean_e = np.mean(err)
            max_e  = np.max(err)
            ax.set_title(
                f"{fname}\nmean={mean_e:.3f}px  max={max_e:.3f}px",
                fontsize=8, color='lightgray'
            )
            ax.tick_params(colors='gray', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        # Colorbar biểu thị giá trị lỗi tương ứng với màu sắc
        if q_last is not None:
            cbar = fig.colorbar(
                q_last, ax=axes.ravel().tolist(),
                shrink=0.6, pad=0.02
            )
            cbar.set_label('Reprojection error (pixels)', color='lightgray')
            cbar.ax.yaxis.set_tick_params(color='lightgray')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='lightgray')

        plt.tight_layout()
        print("Đang hiển thị trace-back reprojection error trên ảnh...")
        if save_dir:
            save_path = os.path.join(save_dir, "reprojection_traceback.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✔ Đã lưu ảnh ghép trực quan hóa trace-back vào: {save_path}")
        if self.visualization:
            plt.show()
        else:
            plt.close()

    # ==========================================================
    def visualize_board_heatmap(self, reproj_data, save_dir=None):
        """
        Vẽ heatmap error theo vị trí góc (col, row) trên bàn cờ.

        Mỗi ô trong grid = một góc nội (inner corner) trên bàn cờ.
        Màu = mean reprojection error tại góc đó, tổng hợp qua
        tất cả ảnh calibration.

        Giúp phát hiện vùng bàn cờ/ảnh thường xuyên có lỗi cao
        (vd: góc ngoài cùng, vùng bị blur, hay bị che khuất).
        """

        w, h = self.shape_inner_corner   # (cols, rows) của inner corners
        N = w * h

        # Tích lũy error theo từng góc qua tất cả ảnh
        # cp_world được xây dựng với np.mgrid[0:w, 0:h].T → corner idx = row*w + col
        acc_err   = np.zeros(N, dtype=np.float64)
        acc_count = np.zeros(N, dtype=np.int32)

        for data in reproj_data:
            acc_err   += data["err"]
            acc_count += 1

        mean_err_per_corner = np.where(
            acc_count > 0,
            acc_err / acc_count,
            0.0
        )

        # cp_world layout: mgrid[0:w, 0:h].T.reshape(-1,2)
        # → index k = row*w + col  (row ∈ [0,h), col ∈ [0,w))
        heatmap = mean_err_per_corner.reshape(h, w)

        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0d0d0d')
        fig.suptitle(
            'Heatmap Reprojection Error theo vị trí góc bàn cờ\n'
            '(tổng hợp qua tất cả ảnh calibration)',
            fontsize=12, color='lightgray'
        )

        # --- Panel trái: heatmap màu ---
        ax1 = axes[0]
        ax1.set_facecolor('#111111')
        im = ax1.imshow(
            heatmap, cmap='hot', aspect='equal',
            origin='upper', interpolation='nearest'
        )
        fig.colorbar(im, ax=ax1, label='Mean reprojection error (px)')
        ax1.set_title('Error Heatmap (hot colormap)', color='lightgray')
        ax1.set_xlabel(f'Column (0 → {w-1})', color='lightgray')
        ax1.set_ylabel(f'Row (0 → {h-1})', color='lightgray')
        ax1.tick_params(colors='gray')

        # Vẽ số lên từng ô
        for row in range(h):
            for col in range(w):
                val = heatmap[row, col]
                text_color = 'black' if val > heatmap.max() * 0.6 else 'white'
                ax1.text(
                    col, row, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=max(5, int(90 / max(w, h))),
                    color=text_color
                )

        # Vẽ grid lines để phân tách ô
        ax1.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax1.grid(which='minor', color='#333333', linewidth=0.8)
        ax1.tick_params(which='minor', bottom=False, left=False)
        ax1.set_xticks(range(w))
        ax1.set_yticks(range(h))

        # --- Panel phải: phân phối error theo row và col ---
        ax2 = axes[1]
        ax2.set_facecolor('#111111')

        row_mean = heatmap.mean(axis=1)   # mean theo cột cho từng hàng
        col_mean = heatmap.mean(axis=0)   # mean theo hàng cho từng cột

        x_col = np.arange(w)
        x_row = np.arange(h)

        ax2.bar(
            x_col - 0.2, col_mean, width=0.4,
            color='deepskyblue', label='Mean error by Column'
        )
        ax2.bar(
            x_row + 0.2, row_mean, width=0.4,
            color='salmon', label='Mean error by Row'
        )
        ax2.set_title(
            'Mean Error by Row vs Column',
            color='lightgray'
        )
        ax2.set_xlabel('Index', color='lightgray')
        ax2.set_ylabel('Mean error (px)', color='lightgray')
        ax2.tick_params(colors='gray')
        ax2.legend(frameon=True, facecolor='#1c1c1c', edgecolor='white')

        for spine in ax1.spines.values():
            spine.set_edgecolor('#444')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444')

        plt.tight_layout()
        print("Đang hiển thị heatmap error trên bàn cờ...")
        if save_dir:
            save_path = os.path.join(save_dir, "chessboard_error_heatmap.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✔ Đã lưu ảnh heatmap reprojection error vào: {save_path}")
        if self.visualization:
            plt.show()
        else:
            plt.close()

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
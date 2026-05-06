"""
Bước 1+2: Trích xuất đặc trưng & giải mã hướng bàn cờ.

Sơ đồ tư duy:
  - Phân ngưỡng Otsu → tìm contours với RETR_TREE để có hierarchy.
  - Lọc contour tròn (circularity, aspect-ratio, area).
  - Mỗi vòng tròn trên bảng hybrid có thiết kế viền khác nhau:
      * Vòng đặc biệt nhất (P0 = gốc tọa độ): viền đôi → nhiều child contour nhất.
      * Vòng định trục X (P3 = hàng trên, cột phải): viền nét đứt → chu vi/diện tích lớn.
      * Các vòng còn lại: thường.
  - Sau khi giải mã, P0, P3, P6 (cột trái) → hướng Y; P0, P1, P2 (hàng trên) → hướng X.
  - Dùng H_draft để chiếu M_squares → predicted corners.
  - Chốt sub-pixel bằng Hessian saddle-point.
"""

import cv2
import numpy as np
import os
import math


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: PHÁT HIỆN & GIẢI MÃ 9 VÒNG TRÒN
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_centroid(contour):
    """Tính tâm bằng spatial moments (chuẩn hơn bounding-rect center)."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def _count_children(idx, hierarchy):
    """Đếm số contour con trực tiếp của contour idx."""
    count = 0
    child = hierarchy[0][idx][2]          # first child
    while child != -1:
        count += 1
        child = hierarchy[0][child][0]    # next sibling
    return count


def extract_9_circles(img_gray):
    """
    Trả về list 9 dict:
      {'center': (x,y), 'area': float, 'n_children': int, 'circularity': float}
    Danh sách CHƯA được sắp xếp theo hướng — decode_orientation() sẽ làm việc đó.
    Trả về [] nếu không đủ 9 vòng tròn.
    """
    # Phân ngưỡng Otsu (tự tính threshold, mạnh hơn threshold cố định)
    _, thresh = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # RETR_TREE để lấy hierarchy (cần đếm child contours cho mỗi vòng tròn)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    candidates = []
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 200:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)
        # Circularity lý thuyết ≤ 1.0; cho 5% sai số ảnh thực
        if not (0.75 < circularity < 1.05):
            continue

        # Bounding rect phải gần vuông (hình tròn)
        x, y, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) == 0:
            continue
        if min(bw, bh) / max(bw, bh) < 0.70:
            continue

        center = _spatial_centroid(c)
        if center is None:
            continue

        n_children = _count_children(idx, hierarchy)
        candidates.append({
            'center': center,
            'area': area,
            'circularity': circularity,
            'n_children': n_children,
            'idx': idx,
        })

    if len(candidates) < 9:
        return []

    # Lọc outlier kích thước so với median
    areas = np.array([c['area'] for c in candidates])
    med = float(np.median(areas))
    candidates = [c for c in candidates if 0.25 * med < c['area'] < 4.0 * med]

    if len(candidates) < 9:
        return []

    # Lấy 9 cái gần median nhất (ổn định hơn top-N lớn nhất)
    candidates = sorted(candidates, key=lambda c: abs(c['area'] - med))[:9]
    return candidates


def decode_orientation(candidates):
    """
    Gán thứ tự chuẩn P0…P8 cho 9 vòng tròn dựa trên thiết kế viền.

    Quy ước hệ tọa độ thế giới (xem main.py):
      P0 = gốc (0, 0)   → viền đặc biệt nhất (nhiều child nhất = viền đôi)
      P1 = (pitch_x, 0) → cùng hàng P0, kề phải
      P2 = (2*pitch_x,0)
      P3 = (0, pitch_y)
      ...
      P8 = (2*pitch_x, 2*pitch_y)

    Thuật toán:
      1. Sắp xếp 9 candidates theo vị trí hình học (y rồi x) → lưới 3×3 thô.
      2. Trong 4 góc của lưới thô, chọn P0 = góc có n_children nhiều nhất.
      3. Từ P0, xác định hướng X và Y bằng tọa độ tương đối.
      4. Sắp xếp lại toàn bộ theo hệ trục P0→X, P0→Y.

    Trả về mảng np (9, 2) theo đúng thứ tự P0…P8 trong hệ tọa độ thế giới.
    Nếu không giải mã được, trả về None.
    """
    if len(candidates) != 9:
        return None

    # ── Bước 1: Sắp xếp hình học thô (row-major, top→bottom, left→right) ──
    geo_sorted = sorted(candidates, key=lambda c: c['center'][1])
    rows_geo = [
        sorted(geo_sorted[0:3], key=lambda c: c['center'][0]),
        sorted(geo_sorted[3:6], key=lambda c: c['center'][0]),
        sorted(geo_sorted[6:9], key=lambda c: c['center'][0]),
    ]
    # Tọa độ pixel 4 góc trong lưới thô
    corners_geo = [
        rows_geo[0][0],  # TL
        rows_geo[0][2],  # TR
        rows_geo[2][0],  # BL
        rows_geo[2][2],  # BR
    ]

    # ── Bước 2: Tìm P0 = góc có nhiều child contours nhất ──
    p0_candidate = max(corners_geo, key=lambda c: c['n_children'])
    p0_center = np.array(p0_candidate['center'])

    # ── Bước 3: Xác định hướng trục X và Y từ P0 ──
    # Trong 4 góc, xác định P0 nằm ở vị trí nào trong lưới thô
    # rồi căn chỉnh lại hướng.
    tl = np.array(rows_geo[0][0]['center'])
    tr = np.array(rows_geo[0][2]['center'])
    bl = np.array(rows_geo[2][0]['center'])

    # Vector hàng và cột trong lưới thô
    vec_row = tr - tl   # hướng X thô
    vec_col = bl - tl   # hướng Y thô

    # Xác định P0 là góc nào
    corner_positions = {'TL': tl, 'TR': tr,
                        'BL': bl, 'BR': np.array(rows_geo[2][2]['center'])}
    min_dist = float('inf')
    p0_name = 'TL'
    for name, pos in corner_positions.items():
        d = np.linalg.norm(pos - p0_center)
        if d < min_dist:
            min_dist = d
            p0_name = name

    # Nếu P0 không phải TL, xoay lưới thô để P0 về TL
    # (dùng phép đổi chỗ rows/cols)
    flip_map = {
        'TL': (False, False),
        'TR': (False, True),   # flip horizontal → P0 về TL
        'BL': (True,  False),  # flip vertical
        'BR': (True,  True),
    }
    flip_v, flip_h = flip_map[p0_name]

    grid = rows_geo  # 3×3 list
    if flip_v:
        grid = list(reversed(grid))
    if flip_h:
        grid = [list(reversed(row)) for row in grid]

    # ── Bước 4: Trả về tọa độ pixel theo thứ tự P0…P8 ──
    ordered_pts = []
    for row in grid:
        for c in row:
            ordered_pts.append(c['center'])

    return np.array(ordered_pts, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: CHỐT GÓC VUÔNG SUB-PIXEL BẰNG HESSIAN
# ─────────────────────────────────────────────────────────────────────────────

def _build_hessian_maps(img_gray):
    """Tính các đạo hàm bậc 2 một lần, dùng lại cho tất cả điểm."""
    I_x  = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    I_y  = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    I_xx = cv2.Sobel(I_x,     cv2.CV_64F, 1, 0, ksize=3)
    I_yy = cv2.Sobel(I_y,     cv2.CV_64F, 0, 1, ksize=3)
    I_xy = cv2.Sobel(I_x,     cv2.CV_64F, 0, 1, ksize=3)
    return I_xx, I_yy, I_xy


def refine_corners_hessian(img_gray, predicted_corners, window_size=5):
    """
    Chốt sub-pixel: với mỗi predicted corner, quét cửa sổ (2w+1)×(2w+1)
    và tìm pixel có det(Hessian) âm cực tiểu nhất (saddle point = góc vuông).
    """
    I_xx, I_yy, I_xy = _build_hessian_maps(img_gray)
    h, w = img_gray.shape
    refined = []

    for pt in predicted_corners:
        u, v = int(round(pt[0])), int(round(pt[1]))
        # Giữ trong ảnh
        if u < window_size or u >= w - window_size or \
           v < window_size or v >= h - window_size:
            refined.append(pt)
            continue

        min_det = float('inf')
        best_pt = pt
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                ny, nx = v + dy, u + dx
                det_H = I_xx[ny, nx] * I_yy[ny, nx] - I_xy[ny, nx] ** 2
                # Saddle point: det < 0 (điểm yên ngựa = góc vuông)
                if det_H < min_det:
                    min_det = det_H
                    best_pt = [nx, ny]

        refined.append(best_pt)

    return np.array(refined, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: HÀM CHÍNH – PIPELINE MỘT ẢNH
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, M_circles, M_squares, compute_homography_fn,
                  out_dir, board_size=(11, 8)):
    """
    Pipeline đầy đủ cho một ảnh:
      1. Phát hiện 9 vòng tròn.
      2. Giải mã hướng → tọa độ 2D chuẩn (circles_2d).
      3. Tính H_draft từ M_circles ↔ circles_2d.
      4. Chiếu M_squares qua H_draft → predicted corners.
      5. Chốt sub-pixel bằng Hessian.
      6. Lưu ảnh debug.

    Trả về refined_corners (N, 2) hoặc None nếu thất bại.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Bước 1 & 2: Phát hiện + giải mã hướng ──
    candidates = extract_9_circles(gray)
    if len(candidates) < 9:
        print(f"  [SKIP] {os.path.basename(img_path)}: chỉ tìm được {len(candidates)} vòng tròn")
        return None

    circles_2d = decode_orientation(candidates)
    if circles_2d is None:
        print(f"  [SKIP] {os.path.basename(img_path)}: giải mã hướng thất bại")
        return None

    # ── Bước 3: H_draft (circles → predicted checkerboard) ──
    H_draft = compute_homography_fn(M_circles, circles_2d)

    # ── Bước 4: Chiếu M_squares qua H_draft ──
    N = M_squares.shape[0]
    M_h = np.hstack([M_squares, np.ones((N, 1))]).T   # (3, N)
    proj = H_draft @ M_h
    proj /= proj[2, :]
    predicted_corners = proj[:2, :].T                  # (N, 2)

    # ── Bước 5: Chốt sub-pixel bằng Hessian ──
    refined_corners = refine_corners_hessian(gray, predicted_corners)

    # ── Bước 6: Lưu ảnh debug ──
    img_draw = img.copy()
    # Vẽ lưới góc vuông
    cv2.drawChessboardCorners(img_draw, board_size,
                              refined_corners.reshape(-1, 1, 2), True)
    # Đánh số P0…P8 lên các vòng tròn
    for idx, pt in enumerate(circles_2d):
        cv2.putText(img_draw, f"P{idx}",
                    (int(pt[0]) + 5, int(pt[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(img_draw, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img_draw)

    return refined_corners

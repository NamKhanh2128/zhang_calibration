import cv2
import numpy as np
import os

def get_spatial_moments(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    xc = M["m10"] / M["m00"]
    yc = M["m01"] / M["m00"]
    return xc, yc

def extract_9_circles(img_gray):
    # Phân ngưỡng ảnh
    _, thresh = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Dùng RETR_EXTERNAL để tránh đếm trùng contour lồng nhau
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:          # loại nhiễu nhỏ
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Circularity lý thuyết max = 1.0; cho thêm 5% sai số ảnh thực
        if not (0.75 < circularity < 1.05):
            continue
        # Kiểm tra aspect ratio của bounding rect gần vuông (tròn)
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.70:
            continue
        pt = get_spatial_moments(c)
        if pt is not None:
            circles.append({'center': pt, 'area': area, 'contour': c})

    if len(circles) < 9:
        return np.array([], dtype=np.float32)

    # Lọc theo kích thước nhất quán: loại bỏ outlier area so với median
    areas = np.array([c['area'] for c in circles], dtype=np.float32)
    median_area = float(np.median(areas))
    circles = [c for c in circles
               if 0.25 * median_area < c['area'] < 4.0 * median_area]

    if len(circles) < 9:
        return np.array([], dtype=np.float32)

    # Lấy 9 hình tròn có diện tích gần median nhất (ổn định hơn lấy top-9 lớn nhất)
    circles = sorted(circles, key=lambda c: abs(c['area'] - median_area))[:9]

    # Sắp xếp theo y trước (từ trên xuống), rồi x (từ trái qua)
    circles.sort(key=lambda c: c['center'][1])
    rows = [circles[0:3], circles[3:6], circles[6:9]]
    for row in rows:
        row.sort(key=lambda c: c['center'][0])

    sorted_pts = []
    for row in rows:
        for c in row:
            sorted_pts.append(c['center'])

    return np.array(sorted_pts, dtype=np.float32)

def refine_corners_hessian(img_gray, predicted_corners, window_size=5):
    # Tính đạo hàm bậc 2
    I_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=3)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=3)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=3)
    
    refined_corners = []
    h, w = img_gray.shape
    
    for pt in predicted_corners:
        u, v = int(round(pt[0])), int(round(pt[1]))
        if u < window_size or u >= w - window_size or v < window_size or v >= h - window_size:
            refined_corners.append(pt)
            continue
            
        # Tìm local minimum của det(H) trong window
        min_det = float('inf')
        best_pt = pt
        
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                y, x = v + dy, u + dx
                ixx = I_xx[y, x]
                iyy = I_yy[y, x]
                ixy = I_xy[y, x]
                det_H = ixx * iyy - ixy * ixy
                
                # Tìm âm cực tiểu (saddle point)
                if det_H < min_det:
                    min_det = det_H
                    best_pt = [x, y]
                    
        refined_corners.append(best_pt)
        
    return np.array(refined_corners, dtype=np.float32)

def process_image(img_path, M_circles, M_squares, compute_homography_fn, out_dir, board_size=(11, 8)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tìm 9 hình tròn
    circles_2d = extract_9_circles(gray)
    if circles_2d is None or len(circles_2d) != 9:
        return None
        
    # 2. Tính Homography nháp từ 9 hình tròn
    H_draft = compute_homography_fn(M_circles, circles_2d)
    
    # 3. Dự đoán vị trí các góc vuông
    # M_squares shape: (N, 2)
    N = M_squares.shape[0]
    ones = np.ones((N, 1))
    M_homo = np.hstack([M_squares, ones]).T # (3, N)
    pred_homo = H_draft @ M_homo # (3, N)
    pred_homo = pred_homo / pred_homo[2, :]
    predicted_corners = pred_homo[:2, :].T # (N, 2)
    
    # 4. Refine góc vuông bằng Hessian
    refined_corners = refine_corners_hessian(gray, predicted_corners)
    
    # Vẽ và lưu ảnh bằng đường nối hoàn chỉnh của OpenCV
    img_draw = img.copy()
    corners_drawn = refined_corners.reshape(-1, 1, 2)
    cv2.drawChessboardCorners(img_draw, board_size, corners_drawn, True)
    
    # Đánh số thứ tự 1-9 lên 9 điểm tròn để xác nhận
    for idx, pt in enumerate(circles_2d):
        cv2.putText(img_draw, str(idx + 1), (int(pt[0]) - 10, int(pt[1]) + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_name, img_draw)
    
    return refined_corners

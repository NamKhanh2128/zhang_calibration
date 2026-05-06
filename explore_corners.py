import cv2
import numpy as np
import math
import glob

img_paths = glob.glob(r"C:\Users\KHANH\Documents\GitHub\zhang_calibration\dataset\input\boardchecker_img\*.jpg")

for img_path in img_paths[:5]:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 100: continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.8:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                circles.append({'area': area, 'center': (cX, cY), 'radius': math.sqrt(area/math.pi)})

    circles.sort(key=lambda x: x['area'], reverse=True)
    nine_circles = circles[:9]
    if len(nine_circles) < 9: continue
    
    # Sort geometrically
    nine_circles.sort(key=lambda c: c['center'][1])
    rows = [nine_circles[0:3], nine_circles[3:6], nine_circles[6:9]]
    for row in rows:
        row.sort(key=lambda c: c['center'][0])
    
    # Compute vector for the 4 corners
    corners = [
        ("TL", rows[0][0]),
        ("TR", rows[0][2]),
        ("BL", rows[2][0]),
        ("BR", rows[2][2])
    ]
    
    print(f"\nImage: {img_path.split(chr(92))[-1]}")
    for name, c in corners:
        cX, cY = c['center']
        R = c['radius']
        
        pad = int(R * 2.5)
        x1, y1 = max(0, cX - pad), max(0, cY - pad)
        x2, y2 = min(img.shape[1], cX + pad), min(img.shape[0], cY + pad)
        
        roi = thresh[y1:y2, x1:x2]
        M_roi = cv2.moments(roi)
        if M_roi["m00"] != 0:
            comX = x1 + M_roi["m10"] / M_roi["m00"]
            comY = y1 + M_roi["m01"] / M_roi["m00"]
            
            dx_norm = (comX - cX) / R
            dy_norm = (comY - cY) / R
            mag = math.hypot(dx_norm, dy_norm)
            print(f"{name}: Mag = {mag:.3f}")

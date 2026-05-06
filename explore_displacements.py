import cv2
import numpy as np
import math

img_path = r"C:\Users\KHANH\Documents\GitHub\zhang_calibration\dataset\input\boardchecker_img\z7796354874034_a8cb97288f26541cafec8330e36efc01.jpg"
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
            circles.append({'area': area, 'circ': circularity, 'center': (cX, cY), 'radius': math.sqrt(area/math.pi)})

circles.sort(key=lambda x: x['area'], reverse=True)
nine_circles = circles[:9]

# Calculate displacement vector for each of the 9 circles
for i, c in enumerate(nine_circles):
    cX, cY = c['center']
    R = c['radius']
    
    # Bounding box around the whole marker
    pad = int(R * 2.5)
    x1, y1 = max(0, cX - pad), max(0, cY - pad)
    x2, y2 = min(img.shape[1], cX + pad), min(img.shape[0], cY + pad)
    
    roi = thresh[y1:y2, x1:x2]
    M_roi = cv2.moments(roi)
    if M_roi["m00"] != 0:
        comX = x1 + M_roi["m10"] / M_roi["m00"]
        comY = y1 + M_roi["m01"] / M_roi["m00"]
        
        # Vector from center of inner circle to center of mass of the whole marker
        dx = comX - cX
        dy = comY - cY
        
        # Normalize by R to make it scale-invariant
        dx_norm = dx / R
        dy_norm = dy / R
        c['displacement'] = (dx_norm, dy_norm)
        c['magnitude'] = math.hypot(dx_norm, dy_norm)
        c['angle'] = math.degrees(math.atan2(dy_norm, dx_norm))
        print(f"Circle at {cX, cY}, R={R:.1f}, Mag={c['magnitude']:.3f}, Angle={c['angle']:.1f}")

# Sort by spatial layout to see the grid
nine_circles.sort(key=lambda c: c['center'][1])
rows = [nine_circles[0:3], nine_circles[3:6], nine_circles[6:9]]
for row in rows:
    row.sort(key=lambda c: c['center'][0])

print("\nGrid Layout Angles (approx):")
for r in range(3):
    row_angles = [f"{c['angle']:6.1f}" for c in rows[r]]
    row_mags = [f"{c['magnitude']:6.3f}" for c in rows[r]]
    print(f"Angles: {row_angles} | Mags: {row_mags}")

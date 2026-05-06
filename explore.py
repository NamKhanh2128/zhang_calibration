import cv2
import numpy as np

img_path = r"C:\Users\KHANH\Documents\GitHub\zhang_calibration\dataset\input\boardchecker_img\z7796354874034_a8cb97288f26541cafec8330e36efc01.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

circles = []
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 50: continue
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0: continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity > 0.6:
        circles.append((i, area, circularity, c))

# Let's check hierarchy of these circles
print("Circles with children or parents in the circle list:")
circle_indices = set([c[0] for c in circles])
for i, area, circ, c in circles:
    h = hierarchy[0][i]
    next_c, prev_c, first_child, parent = h
    if first_child in circle_indices or parent in circle_indices:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(f"Index {i}, Area {area:.1f}, Circ {circ:.2f}, Pos: ({cX}, {cY}), parent: {parent}, first_child: {first_child}")

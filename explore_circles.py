import cv2
import numpy as np

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
            circles.append((area, circularity, cX, cY, c))

circles.sort(key=lambda x: x[0], reverse=True)
print("Top 15 most circular contours by area:")
for c in circles[:15]:
    print(f"Area: {c[0]:.1f}, Circ: {c[1]:.2f}, Center: ({c[2]}, {c[3]})")

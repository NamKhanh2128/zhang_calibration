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

# Calculate ring weight for each of the 9 circles
for i, c in enumerate(nine_circles):
    cX, cY = c['center']
    R = c['radius']
    
    # Create a mask for the ring
    mask = np.zeros_like(img)
    cv2.circle(mask, (cX, cY), int(R * 1.8), 255, -1)
    cv2.circle(mask, (cX, cY), int(R * 1.2), 0, -1)
    
    # Count white pixels in thresh inside the mask
    ring_pixels = cv2.bitwise_and(thresh, thresh, mask=mask)
    weight = np.sum(ring_pixels > 0)
    
    # Normalize weight by the area of the ring to handle perspective scaling
    ring_area = np.pi * ((R*1.8)**2 - (R*1.2)**2)
    c['weight'] = weight / ring_area
    
    print(f"Circle at {cX, cY}, R={R:.1f}, Weight={c['weight']:.3f}")

# Sort by weight to see if we can identify C1, C2, C3 etc.
print("Sorted by weight:")
nine_circles.sort(key=lambda x: x['weight'], reverse=True)
for i, c in enumerate(nine_circles):
    print(f"Rank {i+1}: Center {c['center']}, Weight {c['weight']:.3f}")

import numpy as np
import cv2

# Create a 100x100 black image
img = np.zeros((100, 100, 3), dtype=np.uint8)

# Draw a white circle
cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

# Draw a white square
cv2.rectangle(img, (10, 10), (30, 30), (255, 255, 255), -1)

cv2.imwrite('test_image.png', img)
print("Created test_image.png")

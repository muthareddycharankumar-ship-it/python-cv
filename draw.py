import cv2
import numpy as np

# 1. Load an image
img = cv2.imread("Downloads/1900-x-1200-size-hd-desktop-wallpaper-80.jpg")
if img is None:
    print("Error: Could not load image")
    exit()

cv2.imshow("Original Image", img)

drawn_img = img.copy()

cv2.rectangle(drawn_img, (100, 50), (500, 200), (0, 255, 0),3 )

cv2.circle(drawn_img, (300, 150), 50, (255, 0, 0), -1)  

cv2.imshow("Shapes Drawn", drawn_img)

mask = np.zeros(img.shape[:2], dtype="uint8")

cv2.circle(mask, (250, 250), 100, 255, -1)

cv2.imshow("Binary Mask", mask)

masked_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Masked Image", masked_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

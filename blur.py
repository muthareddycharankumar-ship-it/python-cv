import cv2
import numpy as np

img = cv2.imread("Downloads/1900-x-1200-size-hd-desktop-wallpaper-80.jpg")

if img is None:
    print("Error: Could not read the image.")
    exit()

cv2.imshow("Original Image", img)


gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Gaussian Blur", gaussian_blur)


sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, sharpening_kernel)
cv2.imshow("Sharpened Image", sharpened)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)
cv2.imshow("Sobel Edge Detection", sobel)
canny = cv2.Canny(gray, 100, 200)
cv2.imshow("Canny Edge Detection", canny)

equalized = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()

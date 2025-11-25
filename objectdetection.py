import cv2
import numpy as np

# Load image
image = cv2.imread("Downloads/220217120900_i.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing (blur to remove noise)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold (binary image)
_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and count objects
output = image.copy()
for i, cnt in enumerate(contours):
    # Draw contour
    cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

    # Get bounding box and put number
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(output, f"{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 0, 0), 2)

print("Total objects detected:", len(contours))

# Show results
cv2.imshow("Threshold", thresh)
cv2.imshow("Detected Objects", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

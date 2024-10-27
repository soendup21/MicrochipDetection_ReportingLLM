import cv2
import numpy as np

def detect_harris_corners(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    # Detect corners using the Harris corner detector
    harris_corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    # Dilate the corner spots for better visibility
    harris_corners = cv2.dilate(harris_corners, None)

    # Threshold the corner points
    image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    return image

image_path = 'MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg'
image = cv2.imread(image_path)
corner_image = detect_harris_corners(image)

cv2.imshow('Corners Detected', corner_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

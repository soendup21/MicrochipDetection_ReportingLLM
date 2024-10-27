import cv2
import numpy as np

def detect_shi_tomasi_corners(image, max_corners=4, quality_level=0.01, min_distance=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect corners using Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    corners = np.int0(corners)

    # Mark the corners on the image
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image

image_path = 'MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg'
image = cv2.imread(image_path)
corner_image = detect_shi_tomasi_corners(image)

cv2.imshow('Corners Detected', corner_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

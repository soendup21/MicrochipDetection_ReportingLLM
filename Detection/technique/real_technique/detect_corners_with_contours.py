import cv2
import numpy as np

def detect_corners_with_contours(image, epsilon_factor=0.02):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Draw the corners
    for corner in approx_corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image

image_path = 'MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg'
image = cv2.imread(image_path)
corner_image = detect_corners_with_contours(image)

cv2.imshow('Corners Detected', corner_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

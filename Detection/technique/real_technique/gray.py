import cv2
import numpy as np
def display_image(image, window_name="Image"):
    if image is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to display.")

def load_image(image_path, width=1280, height=720):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No image found at {image_path}")
        return None
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

image_path = 'MicrochipDetection_ReportingLLM/Detection/dataset/test (3).jpg'
image = cv2.imread(image_path)
display_image(image)
gray_image = grayscale(image)
display_image(gray_image)

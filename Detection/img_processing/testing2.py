import cv2
import numpy as np
import time
from get_barcode_n_image_rotated import load_image, bounding_box as get_barcode
from get_tray_reduced_process import bounding_box as get_tray, display_image, crop_polygon

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def main():
    start_time = time.time()

    # Load the input image
    image_path = 'dataset/testing_dataset/tray (7).jpg'
    input_image = load_image(image_path)
    if input_image is None:
        print("Failed to load input image.")
        return

    # Detect barcode and rotate the image
    rotated_image, barcode_polygon, cropped_barcode = get_barcode(input_image, min_area=5000, max_area=100000)
    if barcode_polygon is None:
        print("No barcode detected.")
        return

    # Detect tray on rotated image
    tray_image, tray_polygons = get_tray(rotated_image, min_area=50000, epsilon_factor=0.02)
    if not tray_polygons:
        print("No tray detected.")
        return
    
    end_time = time.time()
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

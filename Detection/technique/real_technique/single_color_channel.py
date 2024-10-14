import cv2
import numpy as np

def display_image(image, window_name="Image"):
    if image is not None:
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

def remove_shadows(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply morphological transformation (Top-Hat) to reduce shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Merge back into the color image for HSV processing
    shadow_removed = cv2.merge([tophat, tophat, tophat])
    return shadow_removed

def bounding_box(image, min_area=3000, epsilon_factor=0.02):
    output_image = image.copy()
    
    # Extract the blue channel
    img_channeled = output_image[:, :, 0]
    display_image(img_channeled,'img_channeled')
    
    # Apply Gaussian Blur to the blue channel to reduce noise
    blurred_image = cv2.GaussianBlur(img_channeled, (7, 7), 0)
    display_image(blurred_image,'blurred_image')
    
    # Since img_channeled is a single channel, skip HSV and process directly
    # Apply binary threshold to isolate blue channel regions
    _, binary_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY)
    display_image(binary_image,'binary_image')

    # Apply Canny edge detection
    edges = cv2.Canny(binary_image, 50, 150)
    display_image(edges, "Canny Edge Detection")

    # Find contours based on the edges detected by Canny
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Approximate the contour to a polygon
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx_polygon = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Check if the polygon has exactly 4 vertices (indicating a rectangle-like shape)
            if len(approx_polygon) == 4:
                # Draw the 4-sided polygon
                cv2.polylines(output_image, [approx_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
                detected_polygons.append(approx_polygon)
    
    if detected_polygons:
        return output_image, detected_polygons
    
    print("No 4-sided gray polygonal objects detected.")
    return output_image, None



# Main program
image_path = 'mask_RCNN//testing_dataset//tray (16).jpg'
img = load_image(image_path)
if img is not None:
    output_image, polygons = bounding_box(img, min_area=3000, epsilon_factor=0.02)
    if polygons:
        print("Detected polygons:", polygons)
    else:
        print("No 4-sided polygons detected.")
    display_image(output_image, "Detected Polygons (Gray Only)")
else:
    print("Failed to load image.")

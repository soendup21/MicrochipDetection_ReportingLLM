import cv2
import numpy as np
import time

# Display an image in a resizable window
def display_image(image, window_name="Image"):
    if image is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No image to display for {window_name}.")

# Load an image from the specified path and return it
def load_image(image_path, width=1280, height=720):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No image found at {image_path}")
        return None
    return cv2.resize(image, (width, height))

# Draw bounding square on the largest detected contour
def bounding_box(image, min_area, max_area):
    output_image = image.copy()
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    display_image(gray_image, "Grayscale Image")
    
    # Blur the image
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 3)
    display_image(blurred_image, "Blurred Image")
    
    # Apply Canny edge detection
    med_val = np.median(blurred_image)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(blurred_image, threshold1=lower, threshold2=upper)
    display_image(edges, "Edge Detection")
    
    # Apply dilation to the edges
    dilated_edges = cv2.dilate(edges, None, iterations=1)
    display_image(dilated_edges, "Dilated Edges")
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    
    if valid_contours:
        epsilon = 0.02 * cv2.arcLength(valid_contours[0], True)
        approx_polygon = cv2.approxPolyDP(valid_contours[0], epsilon, True)
        
        # Draw the polygon on the original image
        cv2.polylines(output_image, [approx_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        display_image(output_image, "Detected Polygon")
        
        # Crop the polygonal area
        cropped_polygon = crop_polygon(image, approx_polygon)
        return output_image, approx_polygon, cropped_polygon
    else:
        print("No contour detected within specified area range.")
    return output_image, None, None

def crop_polygon(image, polygon):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    cropped_image = cv2.bitwise_and(image, mask)
    
    x, y, w, h = cv2.boundingRect(polygon)
    return cropped_image[y:y+h, x:x+w]

# Main
start_time = time.time()
img = load_image('MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg')
if img is not None:
    output_image, approx_polygon, cropped_polygon = bounding_box(img, min_area=10000, max_area=100000)
    if approx_polygon is not None:
        print("Polygon coordinates:", approx_polygon)
        display_image(cropped_polygon, "Cropped Polygon Area")
    else:
        print("No polygon detected.")
else:
    print("Failed to load image.")
end_time = time.time()
print(f"Processing Time: {time.time() - start_time:.2f} seconds")

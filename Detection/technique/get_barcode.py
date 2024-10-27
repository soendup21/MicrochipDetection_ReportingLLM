import cv2
import numpy as np
import time

# Display an image in a resizable window
def display_image(image, window_name="Image"):
    
    # Ensure if there is an image in the path directory
    if image is not None:
        # Create a named window with the ability to resize
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Display the image with the specified window name
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to display.")

        
# Load an image from the specified path and return it.
def load_image(image_path, width=1280, height=720):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: No image found at {image_path}")
        return None
    
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (width, height))
    
    return resized_image

# Draw bounding square on the largest detected contour
def bounding_box(image, min_area, max_area):

    output_image = image.copy()
    display_image(output_image)
    
    # Extract the blue channel Blue=0 Green=1 Red=2 or either make grayscale
    #img_channeled = output_image[:, :, 2]
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    display_image(gray_image)
    
     # Blur the blue channel to reduce noise (1, 1) or (3, 3) or (5, 5)....
     # Can change the last number, it is a sigma which help in blurring more details
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 3)
    display_image(blurred_image)
     
     # Apply Canny edge detection
    med_val = np.median(blurred_image)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(blurred_image, threshold1=lower, threshold2=upper)
    display_image(edges)

    # Apply dilation to connect edges
    dilated_edges = cv2.dilate(edges, None, iterations=1)
    display_image(dilated_edges, "Dilated Edges")
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    
    # Draw a polygon around the largest contour if available
    if valid_contours:
        epsilon = 0.02 * cv2.arcLength(valid_contours[0], True)
        approx_polygon = cv2.approxPolyDP(valid_contours[0], epsilon, True)
        
        # Draw the polygon
        cv2.polylines(output_image, [approx_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Crop the polygonal area
        cropped_polygon = crop_polygon(image, approx_polygon)
        return output_image, approx_polygon, cropped_polygon
    else:
        print("No contour detected within specified area range.")
    return output_image, None, None

def crop_polygon(image, polygon):
    # Create a black mask of the same size as the image
    mask = np.zeros_like(image)

    # Draw the polygon on the mask and fill it with white
    cv2.fillPoly(mask, [polygon], (255, 255, 255))

    # Apply the mask to the image
    cropped_image = cv2.bitwise_and(image, mask)
    
    # Create a bounding box around the polygon to crop the image to its minimal bounding rectangle
    x, y, w, h = cv2.boundingRect(polygon)
    cropped_image = cropped_image[y:y+h, x:x+w]
    
    return cropped_image

# Main  ***if the detection of barcode is not properly, change the (3, 3), 3 of [blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 3)]
start_time = time.time()
img = load_image('MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg')
display_image(img, "Original Image")
if img is not None:
    # Detect corners on the image
    output_image, approx_polygon, cropped_polygon = bounding_box(img, min_area=5000, max_area=100000)
    
    # Display the output image with corners marked if have
    display_image(output_image, "Detected Polygon")
    
    # Check if Polygon were found and display the results
    if approx_polygon is not None:
        print("Polygon coordinates:", approx_polygon)
        display_image(cropped_polygon, "Cropped Polygon Area")
    else:
        print("No polygon detected.")
    
else:
    print("Failed to load image.")

end_time = time.time()
print(f"Processing Time: {time.time() - start_time:.2f} seconds")
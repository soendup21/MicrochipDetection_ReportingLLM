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

# Check if the entire bounding box is in the bottom half of the image
def is_in_bottom_half(polygon, img_height):
    # All points in polygon should be below the midpoint (half-height) of the image
    for point in polygon:
        if point[0][1] < img_height / 2:
            return False
    return True

# Rotate and fit the image within the new frame size
def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # Calculate new bounding dimensions
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust the rotation matrix to consider the translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform the rotation and return the image fitted to the new size
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated_image

# Detect and locate the bounding box
def get_barcode(image, min_area, max_area):
    rotated_image = image.copy()
    height = rotated_image.shape[0]
    
    for _ in range(4):  # Rotate up to 270 degrees (4 rotations of 90 degrees)
        output_image = rotated_image.copy()
        
        # Extract the blue channel Blue=0 Green=1 Red=2 or either make grayscale
        #img_channeled = output_image[:, :, 2]
        gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        
        # Blur the blue channel to reduce noise (1, 1) or (3, 3) or (5, 5)....
        # Can change the last number, it is a sigma which help in blurring more details
        blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 3)
        
        # Apply Canny edge detection
        med_val = np.median(blurred_image)
        lower = int(max(0, 0.7 * med_val))
        upper = int(min(255, 1.3 * med_val))
        edges = cv2.Canny(blurred_image, threshold1=lower, threshold2=upper)
        
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # Draw a polygon around the largest contour if available
        if valid_contours:
            epsilon = 0.02 * cv2.arcLength(valid_contours[0], True)
            approx_polygon = cv2.approxPolyDP(valid_contours[0], epsilon, True)
            
            # Check if the bounding box is entirely within the bottom half of the image
            if is_in_bottom_half(approx_polygon, height):
                cropped_polygon = crop_polygon(rotated_image, approx_polygon)
                return output_image, approx_polygon, cropped_polygon
        
        # Rotate the image by 90 degrees if the bounding box is not in the bottom half
        print("Rotating image to adjust bounding box position.")
        rotated_image = rotate_image(rotated_image, 90)
    
    print("No valid bounding box found or unable to position bounding box in bottom half.")
    return rotated_image, None, None

# Crop the polygonal area from the image
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

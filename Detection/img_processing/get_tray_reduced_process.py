import cv2
import numpy as np
import time

def display_image(image, window_name="Image"):
    if image is not None:
        # Create a resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Resize window to a fixed dimension (e.g., 1280x720)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Display the image
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

def get_tray(image, min_area, epsilon_factor=0.02):
    output_image = image.copy()
    
    # Increase brightness of the image
    brightened_image = increase_brightness(output_image, value=50)
    
    # Apply Gaussian Blur to the brightened image to reduce noise
    blurred_image = cv2.GaussianBlur(brightened_image, (0, 0), 1)
    '''display_image(blurred_image,'blurred_image')'''
    
    # Convert the image to HSV color space for processing
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    
    # Define HSV color range for gray
    # Gray doesn’t have a specific hue so 0 to 180
    # Saturation is low because gray tones have little to no color so 0 to 60
    # Gray can have a wide range of brightness values. Here, a range of 60 to 180 is used, which includes dark to light gray tones.)
    #*** adjust until get rid of metal noise
    lower_gray = np.array([0, 0, 60])
    upper_gray = np.array([180, 60, 180])

    # Create a mask that isolates only gray regions
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    '''display_image(gray_mask,'gray_mask')'''
    detected_polygons = find_polygons(output_image, gray_mask, min_area, epsilon_factor)
    
    # If no tray detected, try with a modified HSV range
    if not detected_polygons:
        print("No tray detected with initial HSV range, trying alternative range.")
        lower_gray = np.array([0, 0, 60])
        upper_gray = np.array([180, 60, 140])
        gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
        '''display_image(gray_mask,'gray_mask')'''
        detected_polygons = find_polygons(output_image, gray_mask, min_area, epsilon_factor)
        
    # If no tray detected, try with a modified HSV range
    if not detected_polygons:
        print("No tray detected with initial HSV range, trying alternative range.")
        lower_gray = np.array([0, 0, 60])
        upper_gray = np.array([180, 20, 180])
        gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
        '''display_image(gray_mask,'gray_mask')'''
        detected_polygons = find_polygons(output_image, gray_mask, min_area, epsilon_factor)
        
    if detected_polygons:
        return output_image, detected_polygons
    else:
        print("No 4-sided gray polygonal objects detected with either HSV range.")
    return output_image, None

def find_polygons(output_image, mask, min_area, epsilon_factor):        #use in bounding_box function
    # Apply Canny edge detection
    med_val = np.median(mask)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(mask, lower, upper)
    '''display_image(edges, 'Canny')'''

    # Apply dilation to connect edges
    dilated_edges = cv2.dilate(edges, None, iterations=1)
    '''display_image(dilated_edges, "Dilated Edges")'''

    # Find contours and filter for quadrilaterals
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx_polygon = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx_polygon) == 4:
                cv2.polylines(output_image, [approx_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
                detected_polygons.append(approx_polygon)
    return detected_polygons

def crop_polygon(image, polygon):                                       #use in bounding_box function
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

def increase_brightness(image, value=50):                               #use in bounding_box function
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Increase the brightness by scaling the V channel
    h, s, v = cv2.split(hsv)    
    v = cv2.add(v, value)   # Adjust value as needed(0 is original brightness, increasing value = brightness increase, decreasing value = brightness decrease)
    v = np.clip(v, 0, 255)  # Ensure values remain within [0, 255]
    hsv_bright = cv2.merge([h, s, v])
    # Convert back to BGR for further processing
    brightened_image = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
    return brightened_image

# Main program
# ** this code only work on a specific environment, so if in different place other than in the sony's will not able to detect.
# ** User's white suit must not include in camera due to shadow of white suit is considered as gray color (ELSE make camera view to see whole tray on table)
# **Also the detection will have problem if there is a metal beside tray

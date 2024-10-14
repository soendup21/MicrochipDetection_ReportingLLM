import cv2
import numpy as np

def display_color_patch(hsv_color, window_name="Color Patch"):
    # Create a blank image
    color_patch = np.zeros((100, 100, 3), np.uint8)
    
    # Convert the HSV color to BGR color space
    bgr_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
    
    # Fill the color patch with the BGR color
    color_patch[:] = bgr_color
    
    # Display the color patch
    cv2.imshow(window_name, color_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"{window_name} - HSV: {hsv_color}, BGR: {bgr_color}")

# Define the HSV colors
lower_gray = [0, 0, 20]     # Dark gray
upper_gray = [0, 0, 100]     # Light gray

# Display the color patches
display_color_patch(lower_gray)
display_color_patch(upper_gray)

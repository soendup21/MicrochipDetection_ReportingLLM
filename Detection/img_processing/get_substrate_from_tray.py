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

def processing_for_substrates(image, min_width=0, min_height=500000, epsilon_factor=0.02):
    output_image = image.copy()

    # Increase brightness of the image
    brightened_image = increase_brightness(output_image, value=50)
    
    # Apply Gaussian Blur to the brightened image to reduce noise
    blurred_image = cv2.GaussianBlur(brightened_image, (5, 5), 0)

    # Convert to HSV and create mask to isolate green, yellow, and dark green regions
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    
    # Primary green and yellow HSV range
    lower_color = np.array([5, 30, 30])
    upper_color = np.array([95, 255, 255])
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    '''display_image(color_mask, "Primary Green and Yellow Mask")'''
    
    # Find bounding boxes for substrates
    substrates = find_bounding_boxes(output_image.copy(), color_mask, min_width, min_height, "Primary Range Bounding Boxes")
    
    # First fallback HSV range if fewer than 6 substrates are detected
    if len(substrates) < 6:
        print("Trying alternative HSV range for green and yellow (1st fallback).")
        lower_color = np.array([10, 40, 40])
        upper_color = np.array([90, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
        '''display_image(color_mask, "Fallback Green and Yellow Mask (1st fallback)")'''
        substrates = find_bounding_boxes(output_image.copy(), color_mask, min_width, min_height, "1st Fallback Range Bounding Boxes")

    # Second fallback HSV range if fewer than 6 substrates are still detected
    if len(substrates) < 6:
        print("Trying another alternative HSV range for green and yellow (2nd fallback).")
        lower_color = np.array([15, 50, 50])
        upper_color = np.array([85, 255, 255])
        color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
        '''display_image(color_mask, "Fallback Green and Yellow Mask (2nd fallback)")'''
        substrates = find_bounding_boxes(output_image.copy(), color_mask, min_width, min_height, "2nd Fallback Range Bounding Boxes")

    # Final check for exactly 6 substrates
    if len(substrates) == 6:
        print("6 substrates detected.")
    else:
        print(f"Detected {len(substrates)} substrates. Adjust parameters or check image quality.")

    display_image(output_image, "Final Detected Substrates with Bounding Boxes")
    return output_image

def find_bounding_boxes(output_image, mask, min_width, min_height, window_name="Bounding Boxes"):
    # Apply dilation to the mask to merge nearby areas
    dilated_mask = cv2.dilate(mask, None, iterations=2)
    display_image(dilated_mask, "Dilated Mask")

    # Find contours directly on the dilated binary mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        # Only select boxes with minimum width, height, and approximately square shape
        if w > min_width and h > min_height and 0.8 < aspect_ratio < 1.2:
            # Draw bounding box on output image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bounding_boxes.append((x, y, w, h))

    # Display bounding boxes for the current HSV range
    '''display_image(output_image, window_name)'''
    return bounding_boxes

def increase_brightness(image, value=50):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    hsv_bright = cv2.merge([h, s, v])
    brightened_image = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
    return brightened_image

# Main function
def main():
    image_path = 'MicrochipDetection_ReportingLLM/Detection/dataset/test_tray3.jpeg'  # Update with the correct path
    image = load_image(image_path)
    
    if image is not None:
        processing_for_substrates(image, min_width=100, min_height=100)

if __name__ == "__main__":
    main()

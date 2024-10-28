import cv2
import numpy as np

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
    display_image(image)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: No image found at {image_path}")
        return None
    
    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (width, height))
    
    return resized_image

# Load the image
img = load_image('MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg')

# blur image
blurred_image = cv2.GaussianBlur(img, (7, 7), 5)

display_image(blurred_image, "window")

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

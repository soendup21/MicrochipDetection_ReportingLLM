import cv2
import numpy as np

# Display an image in a resizable window
def display_image(image, window_name="Image"):
    
    #Ensure if there is image in the path directory
    if image is not None:
        # Display the image with a specified window name
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
def bounding_box(image):
    output_image = image.copy()
    med_val = np.median(output_image)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=lower, threshold2=upper)
    return edges

#main
img = load_image('mask_RCNN//testing_dataset//tray (16).jpg')
if img is not None:
    # Detect corners on the image
    output_image= bounding_box(img)
    
    # Display the output image with corners marked
    display_image(output_image)
else:
    print("Failed to load image.")
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

def canny(image):
    output_image = image.copy()
    med_val = np.median(output_image)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=lower, threshold2=upper)
    return edges

# Load the image (adjust the path if necessary)
image = load_image('MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (3).jpg')
if image is not None:
    # Apply edge detection
    edges = canny(image)
    display_image(edges)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green color and 2-pixel thickness

    # Display the result
    display_image(image, 'Contours')
else:
    print("Could not load image. Check the file path.")

import cv2
import numpy as np

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

def order_points(pts):
    # Initialize a list of coordinates that will be ordered as follows:
    # top-left, top-right, bottom-right, and bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def warp_perspective_to_fit_object(cropped_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    display_image(gray, "Gray Image")  # Display the grayscale image

    # Threshold to create a binary mask where black regions are zero
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    display_image(thresh, "Threshold Binary Image")  # Display the binary mask

    # Find contours of the non-black regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure we have found at least one contour
    if not contours:
        print("No non-black region found.")
        return cropped_image

    # Get the largest contour assuming the object is rectangular
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Check if the contour has 4 points, indicating a rectangle
    if len(approx) == 4:
        # Order points to ensure consistent orientation
        src_pts = order_points(approx.reshape(4, 2).astype("float32"))
        
        # Draw the corner points on the image
        for point in src_pts:
            cv2.circle(cropped_image, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
        display_image(cropped_image, "Detected Corners on Image")  # Display corners on image

        # Define destination points for warping to fit the entire image size
        maxWidth = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])))
        maxHeight = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))
        
        dst_pts = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply warp perspective to fit the detected object into a rectangle
        warped_image = cv2.warpPerspective(cropped_image, M, (maxWidth, maxHeight))
        display_image(warped_image, "Warped Image Fitting the Object to Full Frame")  # Display the warped image
    else:
        print("Detected contour does not have 4 corners. Cannot apply warp perspective.")
        warped_image = cropped_image  # Return the original if not rectangular

    return warped_image

'''cropped_barcode = load_image('technique//real_technique//warp_sample1.jpg')'''
cropped_tray = load_image('MicrochipDetection_ReportingLLM/Detection/dataset/Cropped Polygon Area.jpeg')

# Apply warp perspective to the cropped barcode image to remove black regions
'''warped_barcode = warp_perspective_to_fit_object(cropped_barcode)
display_image(warped_barcode, "Warped Barcode Image Without Black Regions")'''

# Apply warp perspective to the cropped tray image to remove black regions
warped_tray = warp_perspective_to_fit_object(cropped_tray)
display_image(warped_tray, "Warped Tray Image Without Black Regions")
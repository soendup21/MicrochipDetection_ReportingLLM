import time
from get_barcode_n_image_rotated import load_image, get_barcode, display_image
from get_tray_reduced_process import get_tray, crop_polygon
from warp_perspective_for_cropped_img import warp_perspective_to_fit_object

def main():
    start_time = time.time()

    # Step 1: Load input image
    image_path = 'MicrochipDetection_ReportingLLM/Detection/dataset/testing_dataset/tray (4).jpg'
    input_image = load_image(image_path)
    display_image(input_image)
    if input_image is None:
        print("Failed to load input image.")
        return

    # Step 2: Process image to detect and rotate barcode
    rotated_image, barcode_polygon, cropped_barcode = get_barcode(input_image, min_area=5000, max_area=100000)
    if barcode_polygon is None:
        print("No barcode detected.")
        return
    print("Barcode detected and image rotated.")
    
    # Apply warp perspective to the cropped barcode image to remove black regions
    warped_barcode = warp_perspective_to_fit_object(cropped_barcode)
    display_image(warped_barcode, "Warped Barcode Image Without Black Regions")

    # Step 3: Process rotated image to detect tray
    tray_image, tray_polygons = get_tray(rotated_image, min_area=50000, epsilon_factor=0.02)
    display_image(tray_image)
    end_time = time.time()
    if tray_polygons:
        for tray_polygon in tray_polygons:
            cropped_tray = crop_polygon(tray_image, tray_polygon)
            display_image(cropped_tray)
            # Apply warp perspective to the cropped tray image to remove black regions
            warped_tray = warp_perspective_to_fit_object(cropped_tray)
            display_image(warped_tray, "Warped Tray Image Without Black Regions")
    else:
        print("No tray detected.")

    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

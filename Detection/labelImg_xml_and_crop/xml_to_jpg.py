import xml.etree.ElementTree as ET
import cv2
import os

# Dictionary to keep track of count for each label globally
label_count = {}

# Example usage
image_dir = r'labelImg_xml_and_crop\Trays_and_annotate_all'  # Directory where your images are
xml_dir = r'labelImg_xml_and_crop\Trays_and_annotate_all'  # Directory where your XML files are
output_dir = r'labelImg_xml_and_crop\jpg_image'  # Directory where cropped images will be saved

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append((label, xmin, ymin, xmax, ymax))
    
    return objects

def save_cropped_objects(image_file, xml_file, output_dir):
    global label_count  # Make label_count persistent across multiple images

    # Parse the XML file to get bounding boxes and labels
    objects = parse_xml(xml_file)
    
    # Read the image
    image = cv2.imread(image_file)
    
    if image is None:
        print(f"Error: Could not open image {image_file}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the base name of the image file (without extension and any parentheses)
    image_base_name = os.path.basename(image_file).split('(')[0].strip()

    # Remove the word "tray" from the image base name
    image_base_name = image_base_name.replace("tray", "").strip()

    # Construct the output file name, then replace multiple underscores with a single underscore
    image_base_name = image_base_name.replace("__", "_")

    # Loop over each object (bounding box)
    for (label, xmin, ymin, xmax, ymax) in objects:
        # Increment the count for each label across all images
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1

        # Crop the object from the image
        cropped_img = image[ymin:ymax, xmin:xmax]
        
        # Save the cropped image with the custom number and modified base name
        output_path = os.path.join(output_dir, f"{label}_{image_base_name}_{label_count[label]}.jpg")

        # Ensure no double underscores exist in the output file path
        output_path = output_path.replace("__", "_")
        
        # Save the image
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved: {output_path}")

def process_all_images_in_directory(image_dir, xml_dir, output_dir):
    # Get list of all image files and corresponding XML files
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Get the corresponding XML file
            image_file = os.path.join(image_dir, filename)
            xml_file = os.path.join(xml_dir, filename.replace('.jpg', '.xml').replace('.png', '.xml'))

            # Check if the XML file exists before processing
            if os.path.exists(xml_file):
                save_cropped_objects(image_file, xml_file, output_dir)
            else:
                print(f"Warning: No XML file found for {filename}")

process_all_images_in_directory(image_dir, xml_dir, output_dir)

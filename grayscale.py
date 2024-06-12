import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 face detection model
model = YOLO('yolov8n-face.pt')

# Directory for input images
input_image_dir = 'test_rahulsir'

# Directory for output images
output_base_folder = 'test_rahulsir_graycrop'
output_image_dir = os.path.join(output_base_folder, 'test/images')

# Create output directory if it doesn't exist
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

def crop_and_save(image_path, output_image_path, delta=20):
    # Read the image
    image = cv2.imread(image_path, 0)


    if image is not None:
        yolo_results = model.predict(image_path)
        yolo_result = yolo_results[0]
        if len(yolo_result.boxes) > 0:
            # Get the first face bounding box
            x1, y1, x2, y2 = map(int, yolo_result.boxes.xyxy[0].numpy())

            print(x1,y1,y2,x2)
            # Apply delta
            x1 = max(0, int(x1 - delta))
            y1 = max(0, int(y1 - delta))
            x2 = min(image.shape[1], int(x2 + delta))
            y2 = min(image.shape[0], int(y2 + delta))

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Check if the cropped image is valid
            if cropped_image.size == 0:
                print(f"Error: Cropped image is empty for {image_path}")
                return

            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped and saved {os.path.basename(image_path)}.")
        else:
            print(f"No face detected in {os.path.basename(image_path)}. Skipping...")
    else:
        print(f"Failed to read {image_path}. Skipping...")

# Process the 'test' folder
for filename in os.listdir(input_image_dir):
    image_path = os.path.join(input_image_dir, filename)
    output_image_path = os.path.join(output_image_dir, filename)

    crop_and_save(image_path, output_image_path, delta=20)

print("Cropping, grayscale conversion, and saving complete for the 'test' folder.")

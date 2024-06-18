import os
import cv2
from ultralytics import YOLO

# Load the YOLOv8 face detection model
model = YOLO('yolov8n-face.pt')

# Directory for input images
input_image_dir = 'test_images'

# Directory for output images
output_base_folder = 'test_images_crop'
output_image_dir = os.path.join(output_base_folder, 'test/images')
os.makedirs(output_image_dir, exist_ok=True)

def crop_and_save(image_path, output_image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to read {image_path}. Skipping...")
        return

    # Detect faces in the original image
    yolo_results = model.predict(image)
    yolo_result = yolo_results[0]

    if len(yolo_result.boxes) > 0:
        # Get the first face bounding box
        x1, y1, x2, y2 = map(int, yolo_result.boxes.xyxy[0].numpy())

        # Crop the image based on the detected bounding box
        cropped_image = image[y1:y2, x1:x2]
        
        if cropped_image.size == 0:
            print(f"Cropping resulted in an empty image for {os.path.basename(image_path)}. Skipping...")
        else:
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped and saved {os.path.basename(image_path)}.")
    else:
        print(f"No face detected in {os.path.basename(image_path)}. Skipping...")

# Process each image in the directory
for filename in os.listdir(input_image_dir):
    image_path = os.path.join(input_image_dir, filename)
    output_image_path = os.path.join(output_image_dir, filename)
    crop_and_save(image_path, output_image_path)

print("Processing complete.")

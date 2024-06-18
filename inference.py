import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from ultralytics import YOLO

# Load models
model_yolo = YOLO('MODELS/yolov8n-face.pt')
model_segment = load_model('MODELS/beard_segmentation_model_gray_224.h5', compile=False)

# Path configurations
input_image_dir = '/Users/aaditya/Desktop/Beard-Segmentation/TEST_DATASETS/test_images_gray_cropped/images'
output_base_folder = 'results_mobnet'
os.makedirs(output_base_folder, exist_ok=True)

def dynamic_gamma_correction(image, average_target=150):
    tar = image[image > 0]
    average_input = np.mean(tar)
    gamma = np.log(average_target / 255) / np.log(average_input / 255)
    gamma = max(gamma, 0.8)
    corrected_image = np.power(image / 255.0, gamma) * 255.0
    return np.clip(corrected_image, 0, 255).astype(np.uint8)

def process_image(image_path, model_yolo, model_segment):
    original_image = cv2.imread(image_path,0)
    if original_image is None:
        print(f"Failed to read {image_path}. Skipping...")
        return

    resized_image = cv2.resize(original_image, (224, 224))
    yolo_results = model_yolo.predict(resized_image)
    yolo_result = yolo_results[0]

    if len(yolo_result.boxes) > 0:
        x1, y1, x2, y2 = map(int, yolo_result.boxes.xyxy[0].numpy())
        cropped_image = resized_image[y1:y2, x1:x2]
        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = np.expand_dims(cropped_image,axis=-1)
        input_image_normalized = np.expand_dims(cropped_image / 255.0, axis=0)

        gamma_corrected_image = dynamic_gamma_correction(cropped_image)
        gamma_corrected_image = np.expand_dims(gamma_corrected_image / 255.0, axis=0)

        prediction = model_segment.predict(input_image_normalized)[0]
        threshold = 0.5
        binary_mask = (prediction > threshold).astype(np.uint8)

        binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))
        binary_mask = binary_mask * 255

        # Save the results
        # result_path = os.path.join(output_base_folder, os.path.basename(image_path))
        # cv2.imwrite(result_path, binary_mask)  # Save mask as an image file
        # print(f"Processed and saved results for {image_path}.")

        # Display the original image and the mask
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        #plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')

        plt.show()

# Process each image in the directory
for filename in os.listdir(input_image_dir):
    image_path = os.path.join(input_image_dir, filename)
    process_image(image_path, model_yolo, model_segment)

print("Processing complete.")
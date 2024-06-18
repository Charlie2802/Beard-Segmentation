

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime as ort
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
BATCH_SIZE = 16

test_images_dir = r'TEST_DATASETS/test_images_gray_cropped/images'
results_dir = r'RESULTS/results_rewat'

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Function to check if an image is valid
def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, Image.DecompressionBombError, Image.UnidentifiedImageError):
        return False

def load_image_paths(image_dir):
    return sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if is_image_valid(os.path.join(image_dir, fname))])

test_image_paths = load_image_paths(test_images_dir)
print(len(test_image_paths))

# Load ONNX model
sess = ort.InferenceSession('beard.onnx')

def predict_with_onnx(session, input_data):
    input_name = session.get_inputs()[0].name
    input_data = input_data.astype(np.float32)
    result = session.run(None, {input_name: input_data})
    return result[0]

def dynamic_gamma_correction(image, average_target=150):
    tar = image[image > 0]  # Only consider non-zero pixels
    average_input = np.mean(tar)
    
    gamma = np.log(average_target / 255) / np.log(average_input / 255)
    if gamma > 0.8:
        gamma = 1.0

    corrected_image = np.power(image / 255.0, gamma) * 255.0
    corrected_image = np.clip(corrected_image, 0, 255)  
    return corrected_image.astype(np.float32)

for image_path in test_image_paths:
    print(f"Processing image: {image_path}")
    x = cv2.imread(image_path)
    w, h, _ = x.shape
    image = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    input_image = np.expand_dims(img_to_array(image) / 255.0, axis=0)
    input_image_normalized = (input_image[0] * 255).astype(np.uint8)
    gamma_corrected_image = dynamic_gamma_correction(input_image_normalized)
    gamma_corrected_image = np.expand_dims(gamma_corrected_image / 255.0, axis=0)

    prediction = predict_with_onnx(sess, input_image)
    threshold = 0.65
    binary_mask = (prediction > threshold).astype(np.uint8).squeeze()

    binary_mask = cv2.resize(binary_mask, (h, w))
    prediction1 = predict_with_onnx(sess, gamma_corrected_image)[0]
    binary_mask1 = (prediction1 > threshold).astype(np.uint8)

    print(sum(sum(binary_mask1)))

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(input_image.squeeze())
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(np.squeeze(binary_mask))
    plt.title('Predicted Mask (Original)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(gamma_corrected_image.squeeze())
    plt.title('Gamma Corrected Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(np.squeeze(binary_mask1))
    plt.title('Predicted Mask (Gamma Corrected)')
    plt.axis('off')

    # Save the subplot
    base_filename = os.path.basename(image_path)
    subplot_save_path = os.path.join(results_dir, f"result_{base_filename}.png")
    plt.savefig(subplot_save_path)
    plt.close()

    print(f"Saved results to {subplot_save_path}")

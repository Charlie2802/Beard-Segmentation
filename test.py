import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
import cv2

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 1
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

model = load_model('MODELS/beard_segmentation_model_gray_224.h5', compile=False)

def dynamic_gamma_correction(image, average_target=150):
    tar = image[image > 0]  # Only consider non-zero pixels
    average_input = np.mean(tar)
    
    gamma = np.log(average_target / 255) / np.log(average_input / 255)
    if gamma > 0.8:
        gamma = 1.0

    corrected_image = np.power(image / 255.0, gamma) * 255.0
    corrected_image = np.clip(corrected_image, 0, 255)  
    return corrected_image.astype(np.uint8)

for image_path in test_image_paths:
    print(f"Processing image: {image_path}")
    x=cv2.imread(image_path,0)
    w,h=x.shape
    print(w,h)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT),color_mode='grayscale')
    input_image = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image) / 255.0, axis=0)
    input_image_normalized = (input_image[0] * 255).astype(np.uint8)
    gamma_corrected_image = dynamic_gamma_correction(input_image_normalized)
    gamma_corrected_image = np.expand_dims(gamma_corrected_image / 255.0, axis=0)
    prediction = model.predict(input_image)
    threshold = 0.65
    binary_mask = (prediction > threshold).astype(np.uint8)
    #binary_mask=cv2.resize(binary_mask,(w,h))
    prediction1 = model.predict(gamma_corrected_image)[0]
    binary_mask1 = (prediction1 > threshold).astype(np.uint8)

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(input_image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(np.squeeze(binary_mask), cmap='gray')
    plt.title('Predicted Mask (Original)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(gamma_corrected_image.squeeze(), cmap='gray')
    plt.title('Gamma Corrected Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(np.squeeze(binary_mask1), cmap='gray')
    plt.title('Predicted Mask (Gamma Corrected)')
    plt.axis('off')
    plt.show()

    # # Save the subplot
    # base_filename = os.path.basename(image_path)
    # subplot_save_path = os.path.join(results_dir, f"result_{base_filename}.png")
    # plt.savefig(subplot_save_path)
    # plt.close()

    # print(f"Saved results to {subplot_save_path}")

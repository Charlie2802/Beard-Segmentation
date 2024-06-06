import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 1
BATCH_SIZE = 16

test_images_dir = '/Users/aaditya/Desktop/Beard-Segmentation/test_images_gray_cropped/images'

# Function to check if an image is valid
def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, Image.DecompressionBombError, Image.UnidentifiedImageError):
        return False

# Function to load image paths
def load_image_paths(image_dir):
    return sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if is_image_valid(os.path.join(image_dir, fname))])

test_image_paths = load_image_paths(test_images_dir)
print(len(test_image_paths))
# Load the model
model = load_model('beard_segmentation_model_gray_224.h5', compile=False)
results_dir=('results_model_gray_crop_224')
os.makedirs(results_dir)
# Process each image and display results
for image_path in test_image_paths:
    try:
        print(f"Processing image: {image_path}")
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
        input_image = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image) / 255.0, axis=0)

        prediction = model.predict(input_image)[0]

        threshold = 0.65
        binary_mask = (prediction > threshold).astype(np.uint8)

        # Plot the results
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(binary_mask), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.imshow(np.squeeze(binary_mask), alpha=0.5, cmap='jet')
        plt.title('Overlay')
        plt.axis('off')

        print("Displaying plot...")
        # Display the plot
        result_image_path = os.path.join(results_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        plt.savefig(result_image_path, bbox_inches='tight')
        print(f"Saved: {result_image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print("Prediction and display complete for the 'test' folder.")

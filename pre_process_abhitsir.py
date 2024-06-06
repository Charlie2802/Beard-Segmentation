import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to transform the image
def transform_image(image_path):
    img = cv2.imread(image_path, 0)
    img1 = img / 255.0
    transformed_img = img * img1 + (1 - img1) * 1.5 * img
    return transformed_img

# Function to brighten the image
def brighten(image_path):
    gray = cv2.imread(image_path, 0)
    ims = cv2.resize(gray, None, fx=0.15, fy=0.15)
    gg = cv2.blur(ims, (7, 7))
    imb = cv2.resize(gg, (gray.shape[1], gray.shape[0]))
    new = gray * (170 / imb)
    return new
def resize(image):
    resized_image=cv2.resize(image,(224,224))
    
    return resized_image

# Create output directories
output_dir_t1 = "test_images_gray_cropped_t1"
output_dir_t2 = "test_images_gray_cropped_t2"
os.makedirs(output_dir_t1, exist_ok=True)
os.makedirs(output_dir_t2, exist_ok=True)

# Input directory
input_dir = "test_images_gray_cropped\images"

# Process each image
image_files = os.listdir(input_dir)
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    
    # Apply transformations
    transformed_image = transform_image(image_path)
    transformed_image=resize(transformed_image)
    brightened_image = brighten(image_path)
    brightened_image=resize(brightened_image)
    # Save transformed images
    transformed_image_path = os.path.join(output_dir_t1, image_file)
    brightened_image_path = os.path.join(output_dir_t2, image_file)
    cv2.imwrite(transformed_image_path, transformed_image)
    cv2.imwrite(brightened_image_path, brightened_image)
    
    # Optionally, plot the images for verification (not necessary for saving)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title('Transformed')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(brightened_image, cmap='gray')
    plt.title('Brightened')
    plt.axis('off')
    
    plt.show()

print("All images transformed, brightened, and saved successfully!")

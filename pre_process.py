import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def transform_image(image_path):
    img = cv2.imread(image_path,0)
    img1 = img / 255.0
    transformed_img = img * img1 + (1 - img1) * 1.5 * img
    return transformed_img

def brighten(image_path):
    gray = cv2.imread(image_path, 0)
    ims = cv2.resize(gray, None, fx=0.15, fy=0.15)
    gg = cv2.blur(ims, (7, 7))
    imb = cv2.resize(gg, (gray.shape[1], gray.shape[0]))
    new = gray * (170 / imb)
    return new,gray

output_dir = "train_3"
os.makedirs(output_dir, exist_ok=True)
input_dir = "/Users/aaditya/Desktop/Beard_Segementation/Beard-Segmentation/train_2/test/images"
image_files = os.listdir(input_dir)
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    transformed_image = transform_image(image_path)
    brightened_image,gray = brighten(image_path)
    
    # Plot original and processed images
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Processed')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(brightened_image, cmap='gray')
    plt.title('Processed')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(brightened_image, cmap='gray')
    plt.title('Processed')
    plt.axis('off')
    
    plt.show()

print("All images transformed and displayed successfully!")

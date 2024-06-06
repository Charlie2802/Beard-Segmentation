import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

dir = 'test_images_gray_cropped_t1'
test_dir = os.listdir(dir)

for filename in test_dir[:30]:
    image_path = os.path.join(dir, filename)  # Get the full path to the image
    img = cv2.imread(image_path)

    if img is not None:  # Check if the image is successfully loaded
        # Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img=gray_img/255.0
        gamma=0.7
        gamma_img=np.power(gray_img,gamma)
        
        # Adjust pixel values in the darker regions

        #adjusted_img = np.where(gray_img < 30, gray_img + 175, gray_img)
        #print(adjusted_img)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(gamma_img, cmap='gray')
        plt.title('Adjusted Image')
        plt.axis('off')
        plt.show()

    else:
        print(f"Failed to load image: {image_path}")

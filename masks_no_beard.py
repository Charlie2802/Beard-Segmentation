import os
import cv2
import numpy as np

def create_empty_masks(images_folder, masks_folder):
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
    
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_file_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_file_path)
            
            if image is not None:
                height, width, _ = image.shape
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Create the corresponding mask file path with .png extension
                if filename.endswith(".jpg"):
                    mask_file_path = os.path.join(masks_folder, filename.replace(".jpg", ".png"))
                else:
                    mask_file_path = os.path.join(masks_folder, filename.replace(".jpeg", ".png"))
                
                cv2.imwrite(mask_file_path, mask)
            else:
                print(f"Error reading image {filename}. Skipping.")

images_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/No_Beard'
masks_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/No_Beard_Masks'

create_empty_masks(images_folder, masks_folder)

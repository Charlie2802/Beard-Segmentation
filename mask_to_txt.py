import cv2
import numpy as np

mask_path = "/Users/aaditya/Desktop/Beard-Segmentation/train_4_graycrop/train/masks/beard_man (50).jpeg"

# Load the mask image in grayscale mode
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Error loading mask image")
else:
    print(mask/255.0)  # Normalize the pixel values for viewing, not for processing

    # Adjust pixel values and ensure they are within the 0-255 range
 
    mask = np.clip(mask, 0, 255).astype(np.uint8)  # Clip values and convert to uint8

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Draw contours on a copy of the mask image
    mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_with_contours, contours, -1, (0, 255, 0), 1)

    # Display the mask with contours
    cv2.imshow('Mask with Contours', mask_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, display the original

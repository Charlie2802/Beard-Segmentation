import cv2
import os

# Define the directory containing the images
image_dir = "/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED 2/train/images"

# List all files in the directory
files = os.listdir(image_dir)

# Iterate over each file in the directory
for filename in files:
    # Construct the full path to the image file
    file_path = os.path.join(image_dir, filename)
    
    # Print the filename

    
    # Read the image
    image = cv2.imread(file_path)
    # Check if the image was successfully loaded
    if image is not None:
        continue
    else:
        # os.remove(filename)
        print(f"Failed to load {filename}")

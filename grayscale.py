import os
import cv2

# Input directories
input_folders = {
    'train': 'train_2/train/images',
    'test': 'train_2/test/images',
    'val': 'train_2/val/images'
}

# Output base directory
output_base_folder = 'train_3'

# Create the necessary subdirectories
subfolders = ['train/images', 'test/images', 'val/images', 'train/masks', 'test/masks', 'val/masks']
for subfolder in subfolders:
    output_folder = os.path.join(output_base_folder, subfolder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Function to convert images to grayscale and save them
def convert_images_to_grayscale(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Read the image
        image = cv2.imread(file_path)
        
        if image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Form the complete output path
            output_path = os.path.join(output_folder, filename)
            
            # Save the grayscale image
            cv2.imwrite(output_path, gray_image)
            print(f"Converted and saved {filename} to grayscale.")
        else:
            print(f"Failed to read {filename}. Skipping...")
for key, input_folder in input_folders.items():
    output_folder = os.path.join(output_base_folder, key, 'images')
    convert_images_to_grayscale(input_folder, output_folder)

print("Conversion complete.")

import os
def find_mismatched_files(images_folder, masks_folder):
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(masks_folder)

    # Create sets of filenames without extensions
    image_files_no_ext = {os.path.splitext(f)[0] for f in image_files}
    mask_files_no_ext = {os.path.splitext(f)[0] for f in mask_files}

    # Find images without corresponding masks
    images_without_masks = image_files_no_ext - mask_files_no_ext
    # Find masks without corresponding images
    masks_without_images = mask_files_no_ext - image_files_no_ext

    # List images without corresponding masks
    print("Images without corresponding masks:")
    for image_file_no_ext in images_without_masks:
        for ext in [".jpg", ".jpeg"]:
            image_file = image_file_no_ext + ext
            image_path = os.path.join(images_folder, image_file)
            if os.path.exists(image_path):
                print(f"Image: {image_path}")

    # List masks without corresponding images
    print("\nMasks without corresponding images:")
    for mask_file_no_ext in masks_without_images:
        mask_file = mask_file_no_ext + ".png"
        mask_path = os.path.join(masks_folder, mask_file)
        if os.path.exists(mask_path):
            print(f"Mask: {mask_path}")

# Define your directories
images_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/train/images'
masks_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/train/masks'

# Call the function to find and list mismatched files
find_mismatched_files(images_folder, masks_folder)

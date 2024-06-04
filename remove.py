import os

def remove_images_without_masks(images_folder, masks_folder):
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(masks_folder)

    # Create sets of filenames without extensions
    image_files_no_ext = {os.path.splitext(f)[0] for f in image_files if f.endswith((".jpg", ".jpeg"))}
    mask_files_no_ext = {os.path.splitext(f)[0] for f in mask_files if f.endswith(".png")}

    # Find images without corresponding masks
    images_without_masks = image_files_no_ext - mask_files_no_ext

    # Remove images without corresponding masks
    for image_file_no_ext in images_without_masks:
        for ext in [".jpg", ".jpeg"]:
            image_file = image_file_no_ext + ext
            image_path = os.path.join(images_folder, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed image {image_path}")

images_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/No_Beard'
masks_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/No_Beard_Masks'

remove_images_without_masks(images_folder, masks_folder)

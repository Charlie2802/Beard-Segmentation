import os
import cv2

# Directories
base_dir = "/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED"
images_dir = os.path.join(base_dir, "Roboflow_Dataset")
masks_dir = os.path.join(base_dir, "Roboflow_Masks")

# List files
test_images = os.listdir(images_dir)
test_masks = os.listdir(masks_dir)

# Sort the lists
test_images.sort()
test_masks.sort()

corrupted_images = []
missing_masks = []
print(len(test_images),len(test_masks))
# # # Check each image
# for image_file in test_images:
#     image_path = os.path.join(images_dir, image_file)
#     mask_path = os.path.join(masks_dir, image_file)  # Assuming mask files have the same name as image files

#     # Try to read the image
#     if cv2.imread(mask_path) is None:
#         corrupted_images.append(image_file)
# #     #     os.remove(image_path)  # Remove corrupted image
# #     #     if os.path.exists(mask_path):
# #     #         os.remove(mask_path)  # Remove corresponding mask if it exists
# #     # elif not os.path.exists(mask_path):
# #     #     missing_masks.append(image_file)
# #     #     os.remove(image_path)  # Remove image if the corresponding mask does not exist

print("Corrupted images removed:")
print(corrupted_images)
print(f"Total corrupted images removed: {len(corrupted_images)}")

print("Images removed due to missing masks:")
print(missing_masks)
print(f"Total images removed due to missing masks: {len(missing_masks)}")

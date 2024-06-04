import os
import cv2

base_dir = "/Users/aaditya/Downloads/FINAL_BEARD_DATASET"
images_dir = os.path.join(base_dir, "No_Beard")
masks_dir = os.path.join(base_dir, "No_Beard_Masks")

# List files
image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

# Sort the lists
image_files.sort()
mask_files.sort()

list1 = []

# Check each image and its corresponding mask
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    mask_path = os.path.join(masks_dir, image_file)

    if not os.path.exists(mask_path):
        list1.append(image_path)
        continue

    a = cv2.imread(image_path)
    b = cv2.imread(mask_path)

    if a is None or b is None:
        list1.append(image_path)
        continue

    x, y, _ = a.shape
    p, q,_ = b.shape
    
    if x != p or y != q:
        print(x,y,p,q)
        list1.append(image_path)

print(list1)
print(f"Total mismatched or corrupted images: {len(list1)}")

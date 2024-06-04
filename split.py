import os
import shutil
import random

# Paths
dataset_dir = 'train_2'
original_images_dir = os.path.join(dataset_dir, 'images')
original_masks_dir = os.path.join(dataset_dir, 'masks')
# New directories
train_images_dir = os.path.join(dataset_dir, 'train_2/images')
train_masks_dir = os.path.join(dataset_dir, 'train_2/masks')
val_images_dir = os.path.join(dataset_dir, 'val_2/images')
val_masks_dir = os.path.join(dataset_dir, 'val_2/masks')
test_images_dir = os.path.join(dataset_dir, 'test_2/images')
test_masks_dir = os.path.join(dataset_dir, 'test_2/masks')

# Create directories if they don't exist
for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, test_images_dir, test_masks_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# List of files
image_files = os.listdir(original_images_dir)
mask_files = os.listdir(original_masks_dir)

# Sort files to ensure corresponding images and masks match
image_files.sort()
mask_files.sort()

# Split the data
random.seed(42)
combined = list(zip(image_files, mask_files))
random.shuffle(combined)
image_files[:], mask_files[:] = zip(*combined)

total_files = len(image_files)
train_split = int(0.7 * total_files)
val_split = int(0.2 * total_files)

train_images = image_files[:train_split]
train_masks = mask_files[:train_split]
val_images = image_files[train_split:train_split + val_split]
val_masks = mask_files[train_split:train_split + val_split]
test_images = image_files[train_split + val_split:]
test_masks = mask_files[train_split + val_split:]

# Function to move files
def move_files(files, source_dir, dest_dir):
    for file in files:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

# Move the files
move_files(train_images, original_images_dir, train_images_dir)
move_files(train_masks, original_masks_dir, train_masks_dir)
move_files(val_images, original_images_dir, val_images_dir)
move_files(val_masks, original_masks_dir, val_masks_dir)
move_files(test_images, original_images_dir, test_images_dir)
move_files(test_masks, original_masks_dir, test_masks_dir)

print("Data distribution complete.")
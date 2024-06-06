import os
import cv2

dir = 'test_images_gray_cropped_t1'
test_dir = os.listdir(dir)

for filename in test_dir[130:]:
    image_path = os.path.join(dir, filename)  # Get the full path to the image
    img = cv2.imread(image_path)

    if img is not None:  # Check if the image is successfully loaded
        cv2.imshow('win', img)
        cv2.waitKey(0)
    else:
        print(f"Failed to load image: {image_path}")

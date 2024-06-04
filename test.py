import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
BATCH_SIZE = 16


test_images_dir = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\test_new'
test_masks_dir = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\train_2\test\masks'

# Function to check if an image is valid
def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, Image.DecompressionBombError, Image.UnidentifiedImageError):
        return False


def load_image_paths(image_dir):
    return sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if is_image_valid(os.path.join(image_dir, fname))])
test_image_paths = load_image_paths(test_images_dir)
test_mask_paths = load_image_paths(test_masks_dir)

model = load_model('beard_segmentation_model_224.h5', compile=False)
i=0
list1=[]
for image_path, mask_path in zip(test_image_paths[:20], test_mask_paths[:20]):

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    input_image = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image) / 255.0, axis=0)

    mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    i=i+1
    mask_image = np.expand_dims(tf.keras.preprocessing.image.img_to_array(mask) / 255.0, axis=0)

    prediction = model.predict(input_image)[0]
    threshold = 0.65
    binary_mask = (prediction > threshold).astype(np.uint8)
    print(binary_mask.shape,mask_image.shape)
#     tp=0    
#     tn=0
#     fp=0
#     fn=0
#     for i in range(128):
#         for j in range(128):
#             if binary_mask[ i, j, 0] == 1 and mask_image[0, i, j, 0] == 1:
#                 tp += 1
#             elif binary_mask[ i, j, 0] == 0 and mask_image[0, i, j, 0] == 0:
#                 tn += 1
#             elif binary_mask[ i, j, 0] == 1 and mask_image[0, i, j, 0] == 0:
#                 fp += 1
#             elif binary_mask[ i, j, 0] == 0 and mask_image[0, i, j, 0] == 1:
#                 fn += 1
#     if tp+fp+fn==0:
#         list1.append(1)
#     else:
#         list1.append(tp/(tp+fp+fn))
#     #list1.append([tp,tn,fp,fn])

# print(list1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(binary_mask), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(image))
    plt.imshow(np.squeeze(binary_mask), alpha=0.5, cmap='jet')
    plt.title('Overlay')
    plt.axis('off')
    plt.show()

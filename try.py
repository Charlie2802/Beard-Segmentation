import cv2 
import os 
dir=os.listdir('/Users/aaditya/Desktop/Beard_Segementation/Beard-Segmentation/train_4/test/images')
# for i in range(len(dir)):
#     image_path=dir[i]
#     print(image_path)
#     path=os.path.join(dir,image_path)
#     img=cv2.imread(path)
#     print(img.shape)

img=cv2.imread('/Users/aaditya/Desktop/Beard_Segementation/Beard-Segmentation/train_4/train/images/beard_man (48).jpeg')

masks=cv2.imread('/Users/aaditya/Desktop/Beard_Segementation/Beard-Segmentation/train_4/train/masks/beard_man (48).jpeg')
print(img.shape,masks.shape)
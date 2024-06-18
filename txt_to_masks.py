import cv2
import os
import numpy as np

def read_polygons_from_file(file_path):
    polygons = []
    with open(file_path, 'r') as file:
        for line in file:
            current_polygon = []
            stripped_line = line.strip()
            points = stripped_line.split()
            for i in range(1, len(points), 2):
                x, y = points[i:i+2]
                current_polygon.append([int(float(x) * 640), int(float(y) * 640)])
            polygons.append(current_polygon)
    return polygons

def draw_filled_polygons(polygons, color):
    img = np.zeros((640, 640), dtype=np.uint8)
    pts = [np.array(poly, np.int32).reshape((-1, 1, 2)) for poly in polygons]
    cv2.fillPoly(img, pts, color)
    return img

def process_labels_and_create_masks(labels_folder, masks_folder, images_folder):
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
    
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_file_name = filename
            label_file_name = filename.replace(".jpg", ".txt")
            label_file_path = os.path.join(labels_folder, label_file_name)
            
            if os.path.exists(label_file_path):
                polygons = read_polygons_from_file(label_file_path)
                mask = draw_filled_polygons(polygons, 255)
                mask_file_path = os.path.join(masks_folder, filename.replace(".jpg", ".png"))
                cv2.imwrite(mask_file_path, mask)
            else:
                print(f"Label file {label_file_name} does not exist. Skipping mask creation for {filename}.")

labels_folder = '/Users/aaditya/Downloads/Beard_Dataset_2/train/labels'
masks_folder = '/Users/aaditya/Desktop/FINAL_BEARD_DATASET_UPDATED/ROB_MASKS'
images_folder = '/Users/aaditya/Downloads/FINAL_BEARD_DATASET/Roboflow_Dataset'

process_labels_and_create_masks(labels_folder, masks_folder, images_folder)

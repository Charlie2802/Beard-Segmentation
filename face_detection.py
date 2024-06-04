from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from ultralytics import YOLO

# Load image from file
filename = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\test_images\test_images\2.jpg'
image = pyplot.imread(filename)

# Face detection using MTCNN
detector = MTCNN()
faces_mtcnn = detector.detect_faces(image)

# Face detection using YOLO
model = YOLO(r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\yolov8n-face.pt')
results_yolo = model(filename)

# Now you can handle the detected faces from both models as per your requirements
# For example, you can draw bounding boxes around the detected faces and display the image

"""
This script visualizes YOLOv8 object detection results and compares them
with ground-truth annotations.

What it does:
- Randomly selects images from a validation dataset
- Runs a trained YOLOv8 model on those images
- Displays:
    1. The image with predicted bounding boxes
    2. The same image with ground-truth (annotated) bounding boxes
- Prints the number of detected vs annotated objects per image

This is mainly used for qualitative evaluation of model performance.
"""

# ---------------- IMPORTS ---------------- #
import os                       #File and directory handling
import random                   #Random image selection
import cv2                      #Image loading and color conversion
import matplotlib.pyplot as plt #Visualization
from ultralytics import YOLO    #YOLOv8 model interface


# ---------------- CONFIGURATION ---------------- #
#Directory containing validation images
IMAGE_DIR = r'C:/Users/benme/Documents/School/Leerjaar 3/Minor/Project/Code/robotics/robotics/YOLOV8s/images/Val'

#Directory containing YOLO-format label files
LABEL_DIR = r'C:/Users/benme/Documents/School/Leerjaar 3/Minor/Project/Code/robotics/robotics/YOLOV8s/labels/Val'

#Path to the trained YOLOv8s weights
WEIGHTS_PATH = r'C:/Users/benme/Documents/School/Leerjaar 3/Minor/Project/Code/robotics/robotics/YOLOV8s/runs/detect/YOLOV8s-Head-Detection/weights/best.pt'

#Alternative weights YOLOv8n
# WEIGHTS_PATH = r'C:/Users/benme/Documents/School/Leerjaar 3/Minor/Project/Code/robotics/robotics/YOLOV8n/runs/detect/YOLOV8n-Head-Detection4/weights/best.pt'

#Number of random images to visualize
NUM_IMAGES = 10

#YOLO inference thresholds
CONF_THRESHOLD = 0.33   #Minimum confidence for predictions
IOU_THRESHOLD = 0.5     #IoU threshold for Non-Maximum Suppression

#Counters for evaluation output
num_people_predict = 0
num_people_annotate = 0
# ------------------------------------------------ #


# ---------------- MODEL LOADING ---------------- #
#Load the trained YOLOv8 model using the provided weights
model = YOLO(WEIGHTS_PATH)


# ---------------- IMAGE SELECTION ---------------- #
#Get all JPEG images from the image directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpeg')]

#Randomly select a subset of images
random_images = random.sample(image_files, NUM_IMAGES)

#Match each image with its corresponding label file
random_labels = [f.replace('.jpeg', '.txt') for f in random_images]


# ---------------- MAIN LOOP ---------------- #
for img_file, label_file in zip(random_images, random_labels):

    #Construct full paths
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, label_file)

    # -------- IMAGE LOADING -------- #
    #Read image using OpenCV (BGR format)
    img = cv2.imread(img_path)

    #Convert image to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Get image dimensions
    h, w = img.shape[:2]


    # -------- YOLO INFERENCE -------- #
    #Run the model on the image
    results = model(
        img_path,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]

    #Count predicted objects
    num_people_predict = len(results.boxes)


    # -------- VISUALIZATION -------- #
    plt.figure(figsize=(20, 10))

    # ---- LEFT: Predictions ---- #
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Image: {img_file} with Predictions')

    #Draw predicted bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            #Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            #Extract confidence score and class ID
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            #Draw bounding box (red)
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    edgecolor='red',
                    facecolor='none',
                    linewidth=1
                )
            )

            #Add label text
            plt.text(
                x1,
                y1 - 3,
                f'Class: {cls}, Conf: {conf:.2f}',
                color='red',
                fontsize=5
            )


    # ---- RIGHT: Ground Truth ---- #
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Image: {img_file} without Predictions')

    num_people_annotate = 0

    #Draw ground-truth bounding boxes (if label file exists)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                #YOLO label format: class x_center y_center width height
                if len(parts) == 5:
                    num_people_annotate += 1

                    cls, x_center, y_center, width, height = map(float, parts)

                    #Convert normalized coordinates to pixel values
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    #Draw bounding box (green)
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            edgecolor='green',
                            facecolor='none',
                            linewidth=1
                        )
                    )

                    #Add class label
                    plt.text(
                        x1,
                        y1 - 3,
                        f'GT Class: {int(cls)}',
                        color='green',
                        fontsize=5
                    )


    # -------- CONSOLE OUTPUT -------- #
    print(
        f'Image: {img_file} - '
        f'Predicted People: {num_people_predict}, '
        f'Annotated People: {num_people_annotate}'
    )

    plt.show()
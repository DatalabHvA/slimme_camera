"""
Vision_A.py

PURPOSE
-------
This script runs on a Raspberry Pi and is responsible for:
1. Capturing images using the Pi Camera
2. Detecting people (heads) using a YOLOv8 model
3. Applying temporal filtering to stabilize detections
4. Detecting whether the separation wall between Room A and Room B is open or closed
5. Printing results for consumption by another process (interface.py)

IMPORTANT
---------
- This code has NOT been fully tested.
- All thresholds, regions, and paths are experimental and must be calibrated.
- This script is executed via subprocess from interface.py.
- The ONLY output should be a single printed line in CSV format:
  
    CAMERA_ID,WALL_STATE,PEOPLE_COUNT
"""

# ------------------ IMPORTS ------------------ #
from picamera2 import Picamera2        #Raspberry Pi camera interface
from ultralytics import YOLO           #YOLOv8 object detection
import time                            #Delays between frames
import os                              #File handling
import numpy as np                     #Numerical operations
from datetime import datetime          #Timestamped filenames
import cv2                             #Image processing (OpenCV)


# ------------------ CONFIGURATION ------------------ #
#Path to the trained YOLOv8 model weights
MODEL_PATH = "YOLOV8s/runs/train/YOLOV8s-Head-Detection/weights/best.pt"

#YOLO inference thresholds
CONF_THRESHOLD = 0.33    #Minimum confidence for a detection
IOU_THRESHOLD = 0.5      #IoU threshold for Non-Maximum Suppression

#Temporal filtering parameters
TEMPORAL_MIN_HITS = 2    #Detection must appear in at least N frames

#Region to mask (exclude) from detection
#Used to block the other classroom from detection
#Format: (x1, y1, x2, y2)
MASK_REGION = (0, 300, 640, 480)

#Static identifiers (used by interface.py)
CAMERA_ID = "ROOM_A"     # Which room this camera belongs to

#Default wall state (overwritten by detection logic)
WALL_STATE = "CLOSED"    #"OPEN" or "CLOSED"

#Small probe regions used to detect wall state
#These should lie ON the wall area in the image
WALL_CHECK_REGIONS = [
    (100, 320, 140, 360),
    (260, 320, 300, 360),
    (420, 320, 460, 360),
    (580, 320, 620, 360),
]

#Grayscale brightness threshold
#Darker values imply the wall is closed
WALL_DARK_THRESHOLD = 45
# -------------------------------------------------- #


# ------------------ IMAGE CAPTURE ------------------ #
def capture_images(picam2, num_images=3, delay=2) -> list[str]:
    """
    Capture multiple images with a delay between them.
    Used for temporal filtering to reduce false detections.

    :param picam2: Initialized Picamera2 object
    :param num_images: Number of images to capture
    :param delay: Delay (seconds) between captures
    :return: List of image file paths
    """
    image_paths = []

    for i in range(num_images):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S_%f")
        path = f"/tmp/frame_{timestamp}.jpg"

        picam2.capture_file(path)
        image_paths.append(path)

        if i < num_images - 1:
            time.sleep(delay)

    return image_paths


# ------------------ WALL DETECTION ------------------ #
def is_wall_closed(image_path: str) -> bool:
    """
    Determines whether the separation wall is closed by
    checking brightness in predefined wall regions.

    Assumption:
    - Closed wall is darker than open wall.

    :param image_path: Path to image
    :return: True if wall is closed, False otherwise
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    intensities = []

    for (x1, y1, x2, y2) in WALL_CHECK_REGIONS:
        roi = gray[y1:y2, x1:x2]
        intensities.append(roi.mean())

    avg_intensity = sum(intensities) / len(intensities)

    return avg_intensity < WALL_DARK_THRESHOLD


def get_wall_state(image_paths: list[str]) -> str:
    """
    Determines wall state using majority voting across frames.

    :param image_paths: List of captured image paths
    :return: "CLOSED" or "OPEN"
    """
    closed_votes = 0

    for path in image_paths:
        if is_wall_closed(path):
            closed_votes += 1

    #Wall is considered closed if detected in at least 2 frames
    return "CLOSED" if closed_votes >= 2 else "OPEN"


# ------------------ IMAGE MASKING ------------------ #
def apply_mask(image_path: str, mask_region: tuple[int, int, int, int]) -> None:
    """
    Masks (blacks out) a rectangular region in the image.
    This prevents detection in the other classroom.

    NOTE:
    - This modifies the image file in-place.

    :param image_path: Path to image
    :param mask_region: (x1, y1, x2, y2)
    """
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = mask_region

    img[y1:y2, x1:x2] = 0
    cv2.imwrite(image_path, img)


# ------------------ TEMPORAL FILTERING ------------------ #
def IoU(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    :param boxA: [x1, y1, x2, y2]
    :param boxB: [x1, y1, x2, y2]
    :return: IoU value (0–1)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)


def temporal_filter(detections, iou_threshold, min_hits):
    """
    Filters detections across multiple frames.

    Idea:
    - A detection is only kept if it appears in multiple frames
    - Helps reduce false positives and flickering detections

    :param detections: List of frame detections
    :param iou_threshold: IoU threshold for matching boxes
    :param min_hits: Minimum number of frames required
    :return: Filtered list of bounding boxes
    """
    all_boxes = [box for frame in detections for box in frame]
    final_boxes = []

    for box in all_boxes:
        hits = 0
        for frame in detections:
            for other_box in frame:
                if IoU(box, other_box) >= iou_threshold:
                    hits += 1
                    break

        if hits >= min_hits:
            final_boxes.append(box)

    # Remove duplicate boxes
    unique_boxes = []
    for box in final_boxes:
        if not any(IoU(box, ub) >= iou_threshold for ub in unique_boxes):
            unique_boxes.append(box)

    return unique_boxes


# ------------------ PERSON COUNTING ------------------ #
def count_people(image_paths: list[str]) -> tuple[int, str]:
    """
    Full detection pipeline:
    1. Detect wall state
    2. Mask irrelevant region
    3. Run YOLO detection
    4. Apply temporal filtering
    5. Count people

    :param image_paths: List of image paths
    :return: (people_count, wall_state)
    """
    model = YOLO(MODEL_PATH)
    frame_detections = []

    #Determine wall state BEFORE masking
    wall_state = get_wall_state(image_paths)

    for path in image_paths:
        apply_mask(path, MASK_REGION)

        results = model(
            path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])

        frame_detections.append(boxes)

    filtered_boxes = temporal_filter(
        frame_detections,
        IOU_THRESHOLD,
        TEMPORAL_MIN_HITS
    )

    return len(filtered_boxes), wall_state


# ------------------ MAIN ENTRY POINT ------------------ #
def main():
    """
    Main execution function.

    IMPORTANT:
    - This script is called via subprocess.
    - Output format MUST remain unchanged.
    """

    #Initialize camera
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    #Capture images for temporal filtering
    image_paths = capture_images(picam2, num_images=3, delay=2)

    #Stop camera
    picam2.stop()

    #Run detection pipeline
    people_count, wall_state = count_people(image_paths)

    #Clean up temporary images
    for path in image_paths:
        os.remove(path)

    #Output for interface.py (CSV format)
    print(f"{CAMERA_ID},{wall_state},{people_count}")


if __name__ == "__main__":
    main()
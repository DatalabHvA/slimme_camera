"""
vision_shared.py

PURPOSE
-------
Shared logic used by both Vision_A.py and Vision_B.py.
Both scripts are identical except for their configuration constants.
This module avoids code duplication by centralising all functions here.
"""

# ------------------ IMPORTS ------------------ #
from picamera2 import Picamera2        # Raspberry Pi camera interface
from ultralytics import YOLO           # YOLOv8 object detection
import time                            # Delays between frames
import os                              # File handling
import cv2                             # Image processing (OpenCV)
from datetime import datetime          # Timestamped filenames


# ------------------ IMAGE CAPTURE ------------------ #
def capture_images(picam2, num_images=3, delay=2) -> list[str]:
    """
    Capture multiple still images with a delay in between.
    Used for temporal filtering to reduce false detections.

    :param picam2: Initialized Picamera2 object
    :param num_images: Number of images to capture
    :param delay: Delay (seconds) between captures
    :return: List of file paths to captured images
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
def is_wall_closed(image_path: str, wall_check_regions: list, wall_dark_threshold: float) -> bool:
    """
    Determines whether the separation wall is closed by
    checking brightness in predefined wall regions.

    Assumption: closed wall is darker than open wall.

    :param image_path: Path to image
    :param wall_check_regions: List of (x1, y1, x2, y2) regions to check
    :param wall_dark_threshold: Brightness threshold (lower = darker = closed)
    :return: True if wall is closed, False otherwise
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    intensities = [gray[y1:y2, x1:x2].mean() for (x1, y1, x2, y2) in wall_check_regions]
    avg_intensity = sum(intensities) / len(intensities)

    return avg_intensity < wall_dark_threshold


def get_wall_state(image_paths: list[str], wall_check_regions: list, wall_dark_threshold: float) -> str:
    """
    Determines wall state using majority voting across frames.

    :param image_paths: List of captured image paths
    :param wall_check_regions: Regions to check for wall brightness
    :param wall_dark_threshold: Brightness threshold
    :return: "CLOSED" or "OPEN"
    """
    closed_votes = sum(
        1 for path in image_paths
        if is_wall_closed(path, wall_check_regions, wall_dark_threshold)
    )

    return "CLOSED" if closed_votes >= 2 else "OPEN"


# ------------------ IMAGE MASKING ------------------ #
def apply_mask(image_path: str, mask_region: tuple[int, int, int, int]) -> None:
    """
    Blacks out a rectangular region in the image in-place.
    Prevents YOLO from detecting people in the neighboring classroom.

    :param image_path: Path to image
    :param mask_region: (x1, y1, x2, y2)
    """
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = mask_region
    img[y1:y2, x1:x2] = 0
    cv2.imwrite(image_path, img)


# ------------------ TEMPORAL FILTERING ------------------ #
def iou(box_a, box_b) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    :param box_a: [x1, y1, x2, y2]
    :param box_b: [x1, y1, x2, y2]
    :return: IoU value (0–1)
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    return inter_area / float(area_a + area_b - inter_area)


def temporal_filter(detections: list, iou_threshold: float, min_hits: int) -> list:
    """
    Filters detections across multiple frames.
    A detection is only kept if it appears in at least min_hits frames.

    :param detections: List of per-frame detection lists
    :param iou_threshold: IoU threshold for matching boxes across frames
    :param min_hits: Minimum number of frames a detection must appear in
    :return: Filtered list of unique bounding boxes
    """
    all_boxes = [box for frame in detections for box in frame]
    final_boxes = []

    for box in all_boxes:
        hits = sum(
            1 for frame in detections
            if any(iou(box, other) >= iou_threshold for other in frame)
        )
        if hits >= min_hits:
            final_boxes.append(box)

    # Remove duplicates
    unique_boxes = []
    for box in final_boxes:
        if not any(iou(box, ub) >= iou_threshold for ub in unique_boxes):
            unique_boxes.append(box)

    return unique_boxes


# ------------------ PERSON COUNTING ------------------ #
def count_people(image_paths: list[str], model_path: str, conf_threshold: float,
                 iou_threshold: float, mask_region: tuple, wall_check_regions: list,
                 wall_dark_threshold: float, temporal_min_hits: int) -> tuple[int, str]:
    """
    Full detection pipeline:
    1. Detect wall state (before masking)
    2. Mask irrelevant region
    3. Run YOLO detection
    4. Apply temporal filtering
    5. Count people

    :return: (people_count, wall_state)
    """
    model = YOLO(model_path)
    frame_detections = []

    wall_state = get_wall_state(image_paths, wall_check_regions, wall_dark_threshold)

    for path in image_paths:
        apply_mask(path, mask_region)

        results = model(path, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

        boxes = [[*box.xyxy[0].cpu().numpy()] for box in results.boxes]
        frame_detections.append(boxes)

    filtered_boxes = temporal_filter(frame_detections, iou_threshold, temporal_min_hits)

    return len(filtered_boxes), wall_state


# ------------------ CAMERA HELPER ------------------ #
def run_detection(camera_id: str, model_path: str, conf_threshold: float,
                  iou_threshold: float, mask_region: tuple, wall_check_regions: list,
                  wall_dark_threshold: float, temporal_min_hits: int) -> None:
    """
    Main entry point called by vision_A.py and vision_B.py.
    Initializes camera, captures images, runs detection, prints CSV result.

    Output format: CAMERA_ID,WALL_STATE,PEOPLE_COUNT
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    image_paths = capture_images(picam2, num_images=3, delay=2)
    picam2.stop()

    people_count, wall_state = count_people(
        image_paths, model_path, conf_threshold, iou_threshold,
        mask_region, wall_check_regions, wall_dark_threshold, temporal_min_hits
    )

    for path in image_paths:
        os.remove(path)

    print(f"{camera_id},{wall_state},{people_count}")

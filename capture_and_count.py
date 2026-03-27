from picamera2 import Picamera2
from ultralytics import YOLO
import os
import sys
import csv
import subprocess
from datetime import datetime

MODEL_PATH = "YOLOV8s/runs/detect/YOLOV8s-Head-Detection/weights/best.pt"
CONF_THRESHOLD = 0.33
IOU_THRESHOLD = 0.5
CSV_FILE = "people_count.csv"

timer = datetime.now().hour
if timer < 8 or timer >= 20:
    sys.exit()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Foto maken
now = datetime.now()
formatted = now.strftime("%d%m%Y_%H%M%S")
image_path = os.path.join("Pictures", formatted + ".jpeg")

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()
picam2.capture_file(image_path)
picam2.stop()

# Mensen tellen
model = YOLO(MODEL_PATH)
results = model(image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
people_count = len(results.boxes)

# Opslaan in CSV
write_header = not os.path.exists(CSV_FILE)
with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["bestandsnaam", "aantal_mensen"])
    writer.writerow([formatted + ".jpeg", people_count])

# Push naar GitHub
subprocess.run(["git", "add", "Pictures/", CSV_FILE], check=True)
subprocess.run(["git", "commit", "-m", f"Foto en telling {formatted}"], check=True)
subprocess.run(["git", "push", "github", "main"], check=True)

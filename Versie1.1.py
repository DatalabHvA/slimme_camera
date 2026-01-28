#!/usr/bin/env python3
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
picam2 = Picamera2()

start_time = time.time()

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
model.eval()

# Transform for input
transform = T.Compose([
    T.ToTensor()
])

# --- Load a new image ---
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
image = picam2.capture_image().convert("RGB")
picam2.stop()
img_tensor = transform(image)

# --- Load a saved image ---
# folder = 'Pictures'
# name = "/01122025_110002" # Name of picture = DDMMYYYY_HHMMSS that it was made
# image_path = os.path.join(folder + name + ".jpeg") # <<-- change this to your picture path
# image = Image.open(image_path).convert("RGB")
# img_tensor = transform(image)

#Run inference
with torch.no_grad():
    predictions = model([img_tensor])

#Extract predictions
labels = predictions[0]["labels"].numpy()
scores = predictions[0]["scores"].numpy()
boxes = predictions[0]["boxes"].numpy()

#Count people
people = [(box, score) for box, label, score in zip(boxes, labels, scores) if label == 1 and score > 0.31]

print(f"Number of people detected: {len(people)}")

for i, (box, score) in enumerate(people, 1):
    print(f"Person {i}: {score*100:.2f}% certainty")

end_time = time.time()
print("Duur van R-CNN:", end_time - start_time)

"""
#--- OPTIONAL: Draw boxes with certainty ---
#Convert PIL image back to OpenCV
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

for box, score in people:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_cv, f"{score*100:.1f}%", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Get screen size (optional, or set your desired max width/height)
screen_width = 800
screen_height = 600

# Get image size
height, width = image_cv.shape[:2]

# Compute scale factor to fit image inside screen dimensions
scale_width = screen_width / width
scale_height = screen_height / height
scale = min(scale_width, scale_height)  # Keep aspect ratio

# Resize the image
new_width = int(width * scale)
new_height = int(height * scale)
resized_image = cv2.resize(image_cv, (new_width, new_height))

# Show the resized image
cv2.imshow("Detections", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
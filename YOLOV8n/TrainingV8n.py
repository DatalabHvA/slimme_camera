from ultralytics import YOLO

def main():
    """
    Train a YOLOv8n model on the head detection dataset
    with large images and memory-friendly augmentations for small object detection.
    """
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=200,         # Small dataset; enough epochs to learn without overfitting
        imgsz=1536,         # Keep large image size for small object detection
        batch=2,            # Reduce actual batch size to lower RAM usage
        device=0,           # Use GPU 0
        workers=1,          # Avoid DataLoader duplicating big images in RAM
        name="YOLOV8n-Head-Detection",

        # Augmentation parameters
        degrees=10.0,       # small rotations
        translate=0.05,     # slight camera jitter
        scale=0.5,          # slight scale variation
        fliplr=0.5,         # horizontal flip
        mosaic=0.25,         # moderate mosaic augmentation
        close_mosaic=20,    # close mosaic for small objects


        # Colour augmentation
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        # Regularization
        erasing=0.0,

        # Training control
        patience=40         # early stopping patience
    )

if __name__ == "__main__":
    main()
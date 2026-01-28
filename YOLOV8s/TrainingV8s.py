from ultralytics import YOLO

def main():
    """
    Train a YOLOv8s model on the head detection dataset
    with large images and memory-friendly augmentations for small object detection.
    """
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset.yaml",
        epochs=150,         # Small dataset; enough epochs to learn without overfitting
        imgsz=1664,         # Keep large image size for small object detection
        batch=4,            # Reduce actual batch size to lower RAM usage
        device=0,           # Use GPU 0
        workers=2,          # Avoid DataLoader duplicating big images in RAM
        name="YOLOV8s-Head-Detection",

        # Augmentation parameters
        degrees=10.0,       # small rotations
        translate=0.05,     # slight camera jitter
        scale=0.5,          # slight scale variation
        fliplr=0.5,         # horizontal flip
        mosaic=0.35,        # moderate mosaic augmentation
        close_mosaic=20,    # close mosaic for small objects


        # Colour augmentation
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        # Regularization
        erasing=0.0,

        # Training control
        patience=30         # early stopping patience
    )

if __name__ == "__main__":
    main()
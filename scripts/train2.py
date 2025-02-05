from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8 model
    model = YOLO("C:/Users/sayan/Documents/Kriti@3/runs/weed_detection/weights/best.pt")

    # Train the model on labeled data
    model.train(
        data="C:/Users/sayan/Documents/Kriti@3/dataset/data2.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        workers=0,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.1,
        weight_decay=5e-4,
        device="cuda",
        exist_ok=True,
        project="runs3",
        name="weed_detection",
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color augmentation
        flipud=0.5, fliplr=0.5,  # Flip images
        mosaic=1.0,  # Create synthetic training images
        mixup=0.2,  # Blending images
 
    )

    # Save the trained model
    model.export(format="onnx")

if __name__ == "__main__":
    train_model()

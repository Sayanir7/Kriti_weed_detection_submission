from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8 model
    model = YOLO("yolo11n.pt")


    # Train the model on labeled data
    model.train(
        data="C:/Users/sayan/Documents/Kriti@3/dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.1,
        weight_decay=5e-4,
        device="cuda",
        exist_ok=True,
        project="runs",
        name="weed_detection",
    )

    # Save the trained model
    model.export(format="onnx")

if __name__ == "__main__":
    train_model()

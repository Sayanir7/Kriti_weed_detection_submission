from ultralytics import YOLO

def evaluate_model():
    # Load a pretrained YOLOv8 model
    model = YOLO("C:/Users/sayan/Documents/Kriti@2/runs2/weed_detection/weights/best.pt")

    results = model.val(data="C:/Users/sayan/Documents/Kriti@2/dataset/data2.yaml", imgsz=640)
    print(results.box.map)
   
   
   

if __name__ == "__main__":
    evaluate_model()


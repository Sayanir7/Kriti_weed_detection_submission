from ultralytics import YOLO
import os
import glob

# Load the trained model
model = YOLO("C:/Users/sayan/Documents/Kriti@3/runs/weed_detection/weights/best.pt")

CONF_THRESHOLD = 0.80

# Paths
image_dir = "C:/Users/sayan/Documents/Kriti@3/dataset/images/unlabelled/"
pseudo_label_dir = "C:/Users/sayan/Documents/Kriti@3/dataset/labels/unlabelled/"
os.makedirs(pseudo_label_dir, exist_ok=True)

# Get all image filenames
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))  # Change extension if needed

# Run inference on all images
results = model.predict(source=image_dir, conf=CONF_THRESHOLD, save_txt=True)

# Move generated labels to match the image file names
for image_path, result in zip(image_files, results):
    image_name = os.path.basename(image_path).replace(".jpg", ".txt")  # Get filename without path
    result.save_txt(os.path.join(pseudo_label_dir, image_name))  # Save with the same name

print("✅ Pseudo-labels generated successfully!")

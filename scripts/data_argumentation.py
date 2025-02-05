import os
import cv2
import albumentations as A
import numpy as np
from glob import glob
from shutil import copyfile

# Define Paths (Modify these if needed)
IMAGE_DIR = "C:/Users/sayan/Documents/Kriti@3/dataset/images/unlabelled/"  # Folder with original images
LABEL_DIR = "C:/Users/sayan/Documents/Kriti@3/dataset/labels/unlabelled/"  # Folder with corresponding YOLO labels
OUTPUT_IMAGE_DIR = "C:/Users/sayan/Documents/Kriti@3/dataset/images/unlabelled_aug_images/"  # Where augmented images will be saved
OUTPUT_LABEL_DIR = "C:/Users/sayan/Documents/Kriti@3/dataset/labels/unlabelled_aug_labels/"  # Where corresponding labels will be saved

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def clip_bboxes(bboxes, **kwargs):
    return np.clip(np.array(bboxes), 0, 1).tolist()


# Define the augmentation pipeline
transform = A.Compose([
    A.Lambda(bboxes=clip_bboxes),  # ✅ Apply bounding box clipping here
    A.HorizontalFlip(p=0.5),   # 50% chance of horizontal flip
    A.VerticalFlip(p=0.2),     # 20% chance of vertical flip
    A.RandomBrightnessContrast(p=0.3),  # Adjust brightness & contrast
    A.GaussianBlur(blur_limit=3, p=0.2),  # Apply blur
    A.MotionBlur(blur_limit=5, p=0.2),  # Simulate motion blur
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # Adjust gamma (brightness)
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # Color jitter
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # Improve local contrast
    A.Perspective(scale=(0.05, 0.1), p=0.2),  # Perspective transform (simulates different camera angles)
], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"]  # ✅ Keep this format unchanged
))

# Get all image files
image_files = glob(os.path.join(IMAGE_DIR, "*.jpg"))  # Modify for PNG/JPEG if needed

# Process each image
for img_path in image_files:
    # Read Image
    image = cv2.imread(img_path)
    h, w, _ = image.shape  # Get image dimensions

    # Get corresponding label file
    base_name = os.path.basename(img_path).replace(".jpg", "")
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")

    # Load YOLO Labels (if exists)
    bboxes = []
    class_labels = []
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])  # Class ID
                x, y, bw, bh = map(float, parts[1:])  # YOLO bbox (normalized)
                bboxes.append([x, y, bw, bh])
                class_labels.append(class_id)

    # Apply Augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    
    # Save Augmented Image
    aug_img_name = f"{base_name}_aug.jpg"
    aug_img_path = os.path.join(OUTPUT_IMAGE_DIR, aug_img_name)
    cv2.imwrite(aug_img_path, aug_image)

    # Save Augmented Labels (YOLO format)
    if aug_bboxes:
        aug_label_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}_aug.txt")
        with open(aug_label_path, "w") as f:
            for cls, bbox in zip(class_labels, aug_bboxes):
                x, y, bw, bh = bbox  # Normalized YOLO format
                f.write(f"{cls} {x} {y} {bw} {bh}\n")

print("Augmentation complete! Check 'aug_images' and 'aug_labels' folders.".encode("utf-8", "ignore").decode())

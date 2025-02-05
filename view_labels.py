import cv2
import os
import glob

# Define paths
image_dir = "C:/Users/sayan/Documents/Kriti@2/dataset/images/val/"  # Path to images
label_dir = "C:/Users/sayan/Documents/Kriti@2/dataset/labels/val/"  # Path to pseudo-labels

# Class names (Modify according to your dataset)
class_names = ["weed", "sesame"]

# Get all image files
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

for image_path in image_files:
    # Get corresponding label file
    label_path = os.path.join(label_dir, os.path.basename(image_path).replace(".jpg", ".txt"))

    # Read the image'
    image = cv2.imread(image_path)

    # Read label file
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:5])  # Read first 5 values

            # Check if confidence score exists (6th value)
            conf = float(data[5]) if len(data) == 6 else 1.0  # Default to 1.0 if missing

            # Convert YOLO format to pixel coordinates
            img_h, img_w, _ = image.shape
            x_center, y_center, width, height = (
                x_center * img_w, y_center * img_h, width * img_w, height * img_h
            )
            x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Put label text
            label = f"{class_names[class_id]}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show image
    cv2.imshow("Pseudo-Label Visualization", image)
        # Wait for key press, exit on 'q'
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break  # Exit loop if 'q' is pressed

cv2.destroyAllWindows()

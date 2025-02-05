
# 🌱 Semi-Supervised Weed Detection

## 📌 Overview

This project explores **semi-supervised learning** for **weed detection** in agricultural images. Given a **small labeled dataset** and a **larger unlabeled dataset**, we implemented **pseudo-labeling** and **consistency regularization** techniques to enhance detection accuracy while reducing reliance on annotated data.

## 📂 Dataset

- **Labeled Data:** 200 images with sesame crops and weed annotations.
- **Unlabeled Data:** 1000 similar images without labels.
- **Test Data:** 50 images with ground-truth annotations.

## 🏗 Approach

We implemented the following **semi-supervised learning techniques**:

 **Pseudo-Labeling:**  
   - Train a base model using labeled data.  
   - Use this model to generate pseudo-labels for unlabeled images.  
   - Retrain using both labeled and pseudo-labeled data.  


## Models
There are three models 
- Baseline model: runs/weed_detection/weights/best.pt
- Model after trained with pseudo-labels: runs2/weed_detection/weights/best.pt
- Final model: runs3/weed_detection/weights/best.pt
## 🔧Training Details

- **Architecture:** YOLO11n.pt Object Detection Model  
- **Data Augmentations:**
  - Random Cropping, Flipping
  - CutMix & MixUp
  - Color Jittering & Gaussian Noise
  -  A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.2),

## 📊 Results

The final model was evaluated using the metric:  
**0.5 * (F1-Score) + 0.5 * (mAP@[.5:.95])**

| Model | Score |
|--------|-------|
| Baseline (Supervised Only) | 0.582|
| Pseudo-Labeling | 0.725 |
| Final Model| 0.795 |



## ⚡ Challenges & Solutions

- **Noisy pseudo-labels:** Applied a confidence threshold (0.80) to filter unreliable labels.
- **Overfitting on small labeled data:** Used strong augmentations and dropout.



## 🚀 How to Run

###  Clone the repository
```bash
git clone https://github.com/Sayanir7/Kriti_weed_detection_submission.git
cd Kriti_weed_detection_submission
```

### Evaluate the model
```bash
cd scripts
python evaluate.py
```



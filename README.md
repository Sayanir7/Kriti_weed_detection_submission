# Kriti_weed_detection_submission

# 🌱 Semi-Supervised Weed Detection

## 📌 Overview

This project explores **semi-supervised learning** for **weed detection** in agricultural images. Given a **small labeled dataset** and a **larger unlabeled dataset**, we implemented **pseudo-labeling** and **consistency regularization** techniques to enhance detection accuracy while reducing reliance on annotated data.

## 📂 Dataset

- **Labeled Data:** 200 images with sesame crops and weed annotations.
- **Unlabeled Data:** 1000 similar images without labels.
- **Test Data:** 100 images with ground-truth annotations.

## 🏗 Approach

We implemented the following **semi-supervised learning techniques**:

1. **Pseudo-Labeling:**  
   - Train a base model using labeled data.  
   - Use this model to generate pseudo-labels for unlabeled images.  
   - Retrain using both labeled and pseudo-labeled data.  

2. **Mean Teacher (Consistency Regularization):**  
   - Train a student model while enforcing consistency with a teacher model.  
   - Apply small perturbations (e.g., random cropping, flipping) to improve robustness.  

## 🔧 Model & Training Details

- **Architecture:** YOLOv5 Object Detection Model  
- **Loss Functions:** Cross-entropy and localization loss  
- **Data Augmentations:**
  - Random Cropping, Flipping
  - CutMix & MixUp
  - Color Jittering & Gaussian Noise  

## 📊 Results

The final model was evaluated using the metric:  
**0.5 * (F1-Score) + 0.5 * (mAP@[.5:.95])**

| Model | Score |
|--------|-------|
| Baseline (Supervised Only) | 52.3 |
| Pseudo-Labeling | 67.1 |
| Mean Teacher | 73.5 |

The **Mean Teacher approach** yielded the best performance, demonstrating the benefits of semi-supervised learning.

## ⚡ Challenges & Solutions

- **Noisy pseudo-labels:** Applied a confidence threshold (0.85) to filter unreliable labels.
- **Overfitting on small labeled data:** Used strong augmentations and dropout.
- **High computational cost:** Optimized with mixed-precision training.

## 📁 Repository Structure


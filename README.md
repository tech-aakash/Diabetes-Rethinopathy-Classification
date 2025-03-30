# Diabetic Retinopathy Classification Using Deep Learning

This repository contains the implementation of a **Diabetic Retinopathy Classification** project using three state-of-the-art deep learning models: **Swin Transformer**, **Vision Transformer (ViT)**, and **YOLOv11m**. The goal of this research is to detect and classify diabetic retinopathy from fundus images into five distinct classes (Class 0 to Class 4), indicating the severity of the disease.

---

## 📌 **Project Overview**

Diabetic retinopathy is a leading cause of blindness worldwide. Early detection and accurate classification are critical for effective treatment and management. This project applies deep learning models to analyze fundus images and predict the severity of diabetic retinopathy.

- **Swin Transformer**: Achieved the highest accuracy and robust classification, especially for severe cases.
- **Vision Transformer (ViT)**: Demonstrated balanced performance with good specificity across different classes.
- **YOLOv11m**: Faster detection but exhibited challenges in classifying early-stage diabetic retinopathy.

---

## 📊 **Performance Metrics**

### 🔎 **Confusion Matrix - Swin Transformer**
- Provides an in-depth class-wise performance analysis of Swin Transformer.
  
![Swin Transformer Confusion Matrix](images/swin%20transformer%20confusion%20matrix.png)

---

### 🔎 **Confusion Matrix - Vision Transformer (ViT)**

- ViT demonstrated better classification for early-stage DR.

![ViT Confusion Matrix](images/vit%20confusion%20matrix.png)

---

### 🔎 **Confusion Matrix - YOLOv11m**

- YOLOv11m struggled with early-stage classification, especially Class 1.

![YOLOv11m Confusion Matrix](images/yolo%20confusion_matrix_normalized.png)

---

## 📈 **Sensitivity (Recall) Comparison**

- Sensitivity measures the model's ability to correctly identify positive cases. Higher sensitivity is essential for detecting severe cases.

![Sensitivity (Recall) Comparison](images/Sensitivity%20(Recall)%20Comparison.png)

---

## 📈 **Specificity Comparison**

- Specificity measures how well the models identify negative cases, ensuring a low false positive rate.

![Specificity Comparison](images/Specificity%20Comparison.png)

---

## ⚠️ **Caution**
> Ensure that you update the file paths in the training and inference scripts before running the code. The dataset paths and model checkpoint directories may differ based on your system configuration.

---

## 💡 **Acknowledgments**

- The dataset used for this study is publicly available on [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).
- Please cite the dataset as:
  

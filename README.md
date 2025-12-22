# 🦠 Gallbladder Cancer Detection from Ultrasound Images Using Deep Learning

This repository contains code and models for detecting gallbladder cancer (GBC) from ultrasound images using multiple deep learning architectures and ensemble methods. The project aims to improve early and accurate detection of GBC, supporting clinical decision-making.

---

## 📌 Overview

Gallbladder cancer is a highly aggressive disease, often diagnosed late. This project implements a multi-model deep learning ensemble approach using ultrasound images to classify gallbladder conditions. The system uses pre-trained CNN architectures such as ResNet, EfficientNet, MobileNet, DenseNet, and ShuffleNet. Ensemble techniques (normal averaging) are applied to improve accuracy.

Key contributions:

- Individual models with transfer learning
- Normal ensemble learning
- Evaluation using accuracy, F1-score, ROC-AUC, and confusion matrices
- Visualization of training/validation performance and ROC curves

---

## 📁 Dataset Description

Dataset used: [Gallbladder Cancer Ultrasound Dataset](https://www.kaggle.com/datasets/aneerbansaha/gallbladder-cancer/data)

- Total images: 2,294
- Classes: Normal, Benign, Malignant, Gallstone, Abnormal.
- Train/Validation/Test Split:
  - Training: 1,605 images
  - Validation: 346 images
  - Test: 343 images (Made from splitting training data manually and randomly)

---

## 🧹 Data Preprocessing
- Resize images to a consistent size
- Normalize pixel values
- Convert grayscale to RGB (3 channels)
- Data augmentation:
  - Horizontal and vertical flips
  - Random rotation
  - Zoom
  - Color Jitter (Random brightness/contrast adjustments)

---

## 🧠 Methodology

### Training Details
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate Scheduler: StepLR
- Transfer learning with pre-trained weights
- Batch Size: 32
- Epochs: 30
- Early stopping: patience of 5 epochs

### Models
- Pre-Trained Deep Learning Models
- **Normal Ensemble:** Equal averaging of model probabilities

---

## 📈 Evaluation Metrics

- Classification Report
- Training/Validation loss and accuracy plots
- Confusion matrices
- ROC curve and AUC Score

---

## 📊 Results

- Total 16 models were used in this study.
- Individual model accuracies ranged from **60.64% to 86.30%**
- Highest accuracy obtained by Ensemble: **87.17%**

### Model Performance Summary (Test Accuracies)

| Model               | Test Accuracy |
|--------------------|---------------|
| Ensemble            | 87.17%        |
| EfficientNetB0      | 86.30%        |
| MobileNetV2         | 86.01%        |
| DenseNet201         | 85.71%        |
| EfficientNetB1      | 85.42%        |
| ResNet18            | 85.13%        |
| DenseNet121         | 85.13%        |
| EfficientNetB4      | 84.84%        |
| ResNet50V2          | 84.84%        |
| ResNet50V1          | 84.26%        |
| ShuffleNetV2        | 83.67%        |
| EfficientNetB2      | 83.38%        |
| EfficientNetB3      | 83.38%        |
| GBCNet              | 74.05%        |
| SqueezeNet          | 67.35%        |
| RadFormer           | 60.64%        |


---

## 🚀 Usage
### 1. Clone the repository:
   ```
   git clone https://github.com/Slightsmile/gallbladder-cancer-detection-dl.git
   ```
### 2. Install dependencies
### 3. Run Python file in Compiler

---

## 📦 Requirements - Dependencies

- Python 3.9+
- PyTorch 2.x
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy

---

## ✅ Conclusion

This project highlights the potential of machine learning for early gallbladder cancer diagnosis based on ultrasonogram images. The comparative study provides insights into the strengths and limitations of different models in handling real-world healthcare data.

---

> Developed as a part of the Research and Innovation Project at Daffodil International University.

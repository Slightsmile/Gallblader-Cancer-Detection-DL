# 🦠 Gallbladder Cancer Detection from Ultrasound Images Using Deep Learning

This repository contains code and models for detecting gallbladder cancer (GBC) from ultrasound images using multiple deep learning architectures and ensemble methods. The project aims to improve early and accurate detection of GBC, supporting clinical decision-making.

---

## 📌 Overview

Gallbladder cancer is a highly aggressive disease, often diagnosed late. This project implements a multi-model deep learning ensemble approach using ultrasound images to classify gallbladder conditions. The system uses pre-trained CNN architectures such as ResNet, EfficientNet, MobileNet, DenseNet, and ShuffleNet. Ensemble techniques, including **normal averaging** and **weighted averaging**, are applied to improve accuracy.

Key contributions:

- Individual models with transfer learning  
- Normal and weighted ensemble learning  
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
- **Weighted Ensemble:** Probabilities weighted according to top 6 individual model accuracies  

---

## 📈 Evaluation Metrics

- Classification Report
- Training/Validation loss and accuracy plots  
- Confusion matrices
- ROC curve and AUC Score

---

## 📊 Results

- Total 17 models were used in this study.
- Individual model accuracies ranged from **58.6% to 88.63%**  
- Highest accuracy obtained by Weighted ensemble accuracy: **89.21%**

### Model Performance Summary (Test Accuracies)

| Model               | Test Accuracy |
|--------------------|---------------|
| Weighted Ensemble   | 89.21%        |
| Normal Ensemble     | 88.92%        |
| EfficientNetB2      | 88.63%        |
| EfficientNetB1      | 87.76%        |
| EfficientNetB3      | 87.76%        |
| EfficientNetB0      | 87.46%        |
| EfficientNetB4      | 86.59%        |
| MobileNetV2         | 85.71%        |
| ShuffleNetV2        | 85.13%        |
| DenseNet201         | 85.13%        |
| ResNet50V2          | 85.13%        |
| ResNet18            | 84.26%        |
| DenseNet121         | 84.26%        |
| ResNet50V1          | 83.38%        |
| GBCNet              | 74.05%        |
| SqueezeNet          | 70.26%        |
| RadFormer           | 58.60%        |


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

This project highlights the potential of machine learning for early gallblader cancer diagnosis based on ultrasonogram images. The comparative study provides insights into the strengths and limitations of different models in handling real-world healthcare data.

---

> Developed as a part of the Research and Innovation Project at Daffodil International University.


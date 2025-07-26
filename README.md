# 🧠 Liver Disease Prediction using Machine Learning

This repository contains a deep learning-based model to predict various **liver diseases** from image data using **PyTorch**. The dataset is annotated in **YOLOv5 format**, and the task is modeled as **multi-class classification** with four disease categories:

- `ballooning`
- `fibrosis`
- `inflammation`
- `steatosis`

---

## 📁 Dataset

- Images are stored in **YOLOv5 directory structure**.
- Each image has corresponding `.txt` annotations for bounding boxes and class labels.
- Format supports multiple image resolutions and augmentation-ready pipelines.

### 🎯 Classes


names: ['ballooning', 'fibrosis', 'inflammation', 'steatosis']

### ⚙️ Tech Stack
Python 3.10+

PyTorch

Torchvision

OpenCV

NumPy

YAML


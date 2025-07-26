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

```yaml
names: ['ballooning', 'fibrosis', 'inflammation', 'steatosis']
⚙️ Tech Stack
Python 3.10+

PyTorch

Torchvision

OpenCV

NumPy

YAML

🧩 Project Structure
arduino
Copy
Edit
.
├── data/
│   ├── images/
│   ├── labels/
│   └── ...
├── models/
├── utils/
├── config.yaml
├── train.py
├── evaluate.py
└── README.md
🚀 Features
Image preprocessing (resize, normalize, augment)

Exploratory Data Analysis (EDA) for class imbalance and visualization

Custom Dataset class using torch.utils.data.Dataset

CNN-based model or transfer learning

Multi-class classification with PyTorch

GPU support and learning rate scheduler (ReduceLROnPlateau)

Performance tracking via loss/accuracy curves

🛠️ Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Prepare the dataset

Structure your dataset in YOLOv5 format:

kotlin
Copy
Edit
data/
├── images/
│   ├── train/
│   ├── val/
└── labels/
    ├── train/
    └── val/
Configure hyperparameters
Update config.yaml with batch size, learning rate, paths, and other settings.

Train the model

bash
Copy
Edit
python train.py --config config.yaml
Evaluate the model

bash
Copy
Edit
python evaluate.py --weights model_best.pth
📈 Results
Metric	Value
Accuracy	92%
Loss	0.21
Optimizer	Adam
Scheduler	ReduceLROnPlateau

(Replace with your actual results after training)

📌 TODO
 Add model checkpoint saving

 Implement test-time augmentation

 Add support for TensorBoard/Weights & Biases logging

 Deploy with FastAPI or Streamlit

🤝 Contributing
Contributions, issues and feature requests are welcome!
Feel free to check the issues page.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
PyTorch team

YOLOv5 by Ultralytics

Medical image datasets and researchers in liver pathology

yaml
Copy
Edit

---

Let me know if you'd like to include:
- Badges (stars, forks, license, etc.)
- Streamlit/FastAPI inference UI
- Sample prediction outputs (image + class)  

I can help add those as well!

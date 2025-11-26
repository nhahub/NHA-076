# Land Type Classification using EuroSAT  
**Sentinel-2 Satellite Imagery | 10-Class Land Use / Land Cover Classification**
🛰️ 98.44% accurate Land Use/Land Cover classification on EuroSAT dataset using Sentinel-2 imagery (PyTorch + ResNet)

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mohamedelnahry/land-type-classification-eurosat)  
[![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-blue?style=for-the-badge)](https://www.kaggle.com/datasets/mohamedelnahry/land-type-classification-using-eurosatsentinel-2)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blueviolet?style=for-the-badge&logo=python)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)  

## Project Overview
This project implements a high-accuracy land use/land cover (LULC) classification model using the **EuroSAT** dataset, which consists of **27,000 labeled Sentinel-2 satellite images** (64×64 pixels, 13 spectral bands) covering **10 different land types** across Europe.

The goal: accurately classify satellite imagery into one of the following classes:
- AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial  
- Pasture · PermanentCrop · Residential · River · SeaLake

**Achieved Validation Accuracy: 98.44%** (one of the highest reported on this dataset with custom CNN / transfer learning).

## Key Results
| Metric                  | Value                                    |
|-------------------------|------------------------------------------|
| **Validation Accuracy** | **98.44%**                               |
| **Validation Loss**     | **0.047**                                |
| **Best Classes**        | Forest, SeaLake, Residential (>99.5%)    |
| **Most Confused**       | Highway vs PermanentCrop (spectral overlap) |



## Dataset
- **Source**: EuroSAT (Sentinel-2 L2A)  
- **Size**: ~27,000 images (2,700 per class)  
- **Resolution**: 64×64 pixels  
- **Bands**: 13 spectral bands (we use RGB + selected NIR/SWIR for better separation)  
- **Split**: 70% train · 15% validation · 15% test

## Model & Training
- Framework: **PyTorch** (also includes TensorFlow/Keras version in `/tensorflow_version`)
- Architecture: Custom CNN + Transfer Learning options (ResNet18/34/50 pre-trained variants available)
- Optimizer: Adam (lr = 0.001 → 0.0001 with scheduler)
- Loss: CrossEntropyLoss + Label Smoothing (optional)
- Augmentations: RandomHorizontalFlip, RandomVerticalFlip, Rotate, ColorJitter, Normalize
- Training Time: ~12–18 minutes on Kaggle GPU / Colab (50 epochs)

## Project Structure
EuroSAT-Land-Classification/
```text
EuroSAT-Land-Classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                          # (NOT uploaded – download from Kaggle)
│   └── EuroSAT/                   # place dataset here after download
│
├── notebooks/                     # quick experiments & EDA
│   └── 01_Land_Classification_EuroSAT.ipynb
│
├── src/                           # clean, reusable code
│   ├── __init__.py
│   ├── config.py                  # all hyperparameters & paths
│   ├── dataset.py                 # EuroSAT Dataset class + transforms
│   ├── model.py                   # CustomCNN + ResNet variants
│   ├── train.py                   # training + validation loop
│   ├── evaluate.py                # confusion matrix, metrics, plots
│   ├── inference.py               # load model & predict on new images
│   └── utils.py                   # helper functions
│
├── experiments/                   # one folder per run/experiment
│   └── resnet18_98.44/
│       ├── model_best.pth         # best checkpoint
│       ├── training_log.csv
│       └── config.yaml
│
├── outputs/                       # automatically generated results
│   ├── confusion_matrix.png
│   ├── metrics_summary.png
│   ├── accuracy_loss_curve.png
│   ├── classification_report.txt
│   └── sample_predictions.jpg
│
└── images/                        # lightweight images for README only
    ├── confusion_matrix.png
    ├── metrics_table.png
    └── thumbnail.jpg
```
## How to Run
1. **Download the dataset** from Kaggle:  
   https://www.kaggle.com/datasets/mohamedelnahry/land-type-classification-using-eurosatsentinel-2
2. **Place it** in `/data/EuroSAT/` or update the path in the notebook.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python src/train.py

### 🚀 Quick Start
```bash
git clone https://github.com/yourname/EuroSAT-Land-Classification.git
cd EuroSAT-Land-Classification
pip install -r requirements.txt
python src/train.py

   

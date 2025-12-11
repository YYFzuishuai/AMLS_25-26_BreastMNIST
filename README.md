# AMLS 25/26 – BreastMNIST Classification

This repository contains the full implementation of the AMLS 25/26 final assignment.
The aim of this project is to evaluate traditional machine learning methods and deep learning techniques for binary tumour classification on the BreastMNIST dataset, comparing performance under different feature extraction and model capacity configurations.  
Model A focuses on traditional ML classifiers with different feature extraction strategies, while Model B applies deep learning (CNN-based) and model capacity and augmentation experiments.

---

## 📁 Project Structure

```text
AMLS_25-26_BreastMNIST/
├── Code/                         # All model implementations
│   ├── Model_A/
│   │   ├── svm_model.py          # Baseline: flatten + SVM
│   │   ├── pca_svm.py            # PCA + SVM (dimensionality reduction)
│   │   └── hog_svm.py            # HOG feature extraction + SVM
│   │
│   └── Model_B/
│       └── cnn_model.py          # CNN baseline + capacity + augmentation
│
├── Datasets/                     # Left empty in submission
│   └── README.md                 # Explains that data is auto-downloaded
│
├── main.py                       # Main script to run all experiments
├── requirements.txt              # Dependencies (locked versions)
├── README.md                     # Documentation (this file)
└── .gitignore                    # Git ignore rules
```

The project is organized into modular components, separating classical machine-learning models (Model A: SVM + PCA + HOG) and deep-learning methods (Model B: CNN with capacity tuning and data augmentation).

The Datasets directory is intentionally left empty in the submitted package as required by the coursework specification. The BreastMNIST dataset is automatically downloaded during execution through the MedMNIST API.

---

## 🧠 Model Summary

| Model | Technique |
|-------|----------|
| Model A - Baseline | SVM on flattened pixels |
| Model A - PCA | PCA → SVM classification |
| Model A - HOG | Histogram of Oriented Gradients + SVM |
| Model A - Capacity | Hyperparameters (C, gamma) |
| Model B - Baseline CNN | Simple ConvNet |
| Model B - Capacity | More epochs |
| Model B - Data Aug | Rotation, Horizontal Flip |

---

## 🚀 How to Run

### 1️⃣ Install requirements
```bash
pip install -r requirements.txt
```

### 2️⃣ Run all experiments
```bash
python main.py
```

---

## 📦 Required Packages (minimum)
```bash
torch==2.9.1
torchvision==0.24.1
medmnist==3.0.2
numpy==2.3.5
scikit-learn==1.7.2
scikit-image==0.25.2
pillow==12.0.0
```

---

## 📝 Dataset
This project uses BreastMNIST, part of the MedMNIST benchmark.

The dataset will automatically download at runtime.
Source: https://medmnist.com/

---

## 👤 Author
This project was developed as part of the AMLS 25/26 coursework.
(Anonymous submission — University policy prohibits naming.)

---

## Version History
- v1.0 — Initial release  
  - Implemented Model A (SVM-based) and Model B (CNN-based)

- v1.1 — Fixes & improvements  
  - Fixed missing `Datasets` directory required for submission  
  - Corrected Project Structure diagram in README.md

- v1.1.5 — Improvements    
  - Corrected Project Structure diagram in README.md again

- v1.2 — Final update    
  - Removed unnecessary packages (e.g., pandas, matplotlib) from both requirements.txt and README to ensure a clean and minimal dependency set
  - Removed the unused Results directory from the project structure description in README to avoid confusion, as experiment outputs are generated during runtime and not required for submission
  - Added comprehensive English documentation and inline comments across all existing code files for improved readability and clarity

---
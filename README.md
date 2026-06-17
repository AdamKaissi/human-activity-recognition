# Human Activity Recognition
### CMP3011 Introduction to Computer Vision
**Adam El Kaissi (2101431) | Awais Ahmed (2281583)**
**Instructor: Prof MD Imran Hosen | Submission: 17 June 2026**

---

## Overview

This project implements a **Human Activity Recognition (HAR)** system using two approaches:

1. **VideoMAE (Primary)** — A pretrained video understanding model applied via transfer learning. No training required. This is what the report results are based on.
2. **CNN + LSTM (Extended)** — A full training pipeline built from scratch using ResNet-50 + LSTM on HMDB51, included to demonstrate understanding of the full deep learning pipeline.

---

## Quick Start — VideoMAE (Recommended)

No dataset or training needed. Just open the notebook in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdamKaissi/human-activity-recognition/blob/main/HAR_VideoMAE_Colab.ipynb)

1. Click the badge above to open in Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells — the notebook handles everything automatically

**Total runtime: ~5 minutes**

---

## CNN + LSTM Training Pipeline

### Project Structure
har_project/

├── data/

│   └── hmdb51/               ← dataset goes here after download

├── models/

│   ├── cnn_lstm.py           ← CNN + LSTM model architecture

│   └── transforms.py         ← video augmentation transforms

├── utils/

│   ├── dataset.py            ← HMDB51 dataset loader

│   ├── trainer.py            ← training & validation loop

│   └── evaluate.py           ← metrics, confusion matrix, F1-score

├── train.py                  ← main training entry point

├── evaluate.py               ← standalone evaluation script

├── demo.py                   ← real-time webcam demo (optional)

├── config.py                 ← all hyperparameters in one place

├── requirements.txt

└── notebooks/

└── explore.ipynb         ← dataset exploration notebook
### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download HMDB51

```bash
wget -P data/ http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget -P data/ http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
sudo apt-get install unrar -y
unrar x data/hmdb51_org.rar data/hmdb51/
unrar x data/test_train_splits.rar data/hmdb51/
```

### 3. Train the model

```bash
python train.py
```

All outputs (checkpoints, plots, logs) are saved to `outputs/`.

### 4. Evaluate

```bash
python evaluate.py --checkpoint outputs/best_model.pth
```

### 5. Real-time demo (optional, requires webcam)

```bash
python demo.py --checkpoint outputs/best_model.pth
```

---

## Architecture

| Component       | Detail                                       |
|----------------|----------------------------------------------|
| Backbone        | ResNet-50 (ImageNet pretrained, frozen stem) |
| Temporal module | 2-layer LSTM (hidden size 512)               |
| Classifier      | FC → Dropout → FC → Softmax                 |
| Input           | 16 frames per clip, 224×224                  |
| Classes         | 51 (HMDB51)                                  |

---

## Results (VideoMAE on Test Videos)

| Video             | True Label     | Prediction        | Confidence | Correct?  |
|------------------|----------------|-------------------|------------|-----------|
| push_ups.mp4     | push up        | push up           | 66.5%      | ✓         |
| pull_ups.mp4     | pull ups       | pull ups          | 82.8%      | ✓         |
| juggling.mp4     | juggling balls | juggling balls    | 88.6%      | ✓         |
| deadlifting.mp4  | deadlifting    | clean and jerk    | 58.2%      | ✗ (Top-4) |
| hula_hooping.mp4 | hula hooping   | jumpstyle dancing | 8.1%       | ✗ (Top-2) |
| squats.mp4       | squat          | stretching arm    | 41.9%      | ✗         |

**Top-1 Accuracy: 50.0% (3/6) | Top-5 Accuracy: 66.7% (4/6)**

---

## Expected Results (CNN + LSTM Training)

| Metric      | Expected Range |
|------------|----------------|
| Top-1 Acc   | 48–58%         |
| Top-5 Acc   | 78–85%         |
| F1 (macro)  | 0.45–0.55      |

Training takes ~2–4 hours on a single GPU (T4/V100). Use `--debug` flag for a quick test run.

---

## Libraries & Frameworks

- **PyTorch** — model inference and training
- **Hugging Face Transformers** — VideoMAE pretrained model
- **OpenCV** — video decoding and frame extraction
- **torchvision** — ResNet-50 backbone and data augmentation
- **scikit-learn** — evaluation metrics
- **Matplotlib / Seaborn** — visualisation

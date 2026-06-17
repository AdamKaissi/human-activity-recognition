# Human Activity Recognition — HMDB51

Deep learning pipeline for Human Activity Recognition using a CNN + LSTM architecture trained on the HMDB51 dataset.

---

## Project Structure

```
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
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download HMDB51

```bash
# Download dataset splits and videos
wget -P data/ http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
wget -P data/ http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar

# Extract (requires unrar)
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

| Component        | Detail                                      |
|-----------------|---------------------------------------------|
| Backbone         | ResNet-50 (ImageNet pretrained, frozen stem) |
| Temporal module  | 2-layer LSTM (hidden size 512)              |
| Classifier       | FC → Dropout → FC → Softmax                |
| Input            | 16 frames per clip, 224×224                 |
| Classes          | 51 (HMDB51)                                 |

---

## Expected Results

| Metric       | Expected Range  |
|-------------|-----------------|
| Top-1 Acc    | 48–58%          |
| Top-5 Acc    | 78–85%          |
| F1 (macro)   | 0.45–0.55       |

Training takes ~2–4 hours on a single GPU (T4/V100). On CPU it will be slow — use a small subset with `--debug` flag for testing.

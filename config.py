"""
config.py — Central configuration for the HAR project.
Edit this file to change hyperparameters, paths, and model settings.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join("data", "hmdb51")
SPLITS_DIR      = os.path.join(DATA_DIR, "testTrainMulti_7030_splits")
OUTPUT_DIR      = "outputs"
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES     = 51          # HMDB51 has 51 action categories
SPLIT_ID        = 1           # HMDB51 has 3 train/test splits (1, 2, or 3)

# ─── Video Sampling ───────────────────────────────────────────────────────────
NUM_FRAMES      = 16          # frames sampled per video clip
FRAME_HEIGHT    = 224
FRAME_WIDTH     = 224

# ─── Model ────────────────────────────────────────────────────────────────────
BACKBONE        = "resnet50"  # "resnet50" or "mobilenet_v2"
FREEZE_BACKBONE = True        # freeze CNN weights during initial training
LSTM_HIDDEN     = 512
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.5
FC_DROPOUT      = 0.5
FEATURE_DIM     = 2048        # ResNet-50 output; use 1280 for MobileNetV2

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS          = 30
BATCH_SIZE      = 8           # reduce to 4 if you run out of VRAM
LEARNING_RATE   = 1e-3
LR_STEP_SIZE    = 10          # decay LR every N epochs
LR_GAMMA        = 0.1         # LR decay factor
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 4           # set to 0 on Windows

# ─── Fine-tuning (phase 2) ────────────────────────────────────────────────────
FINETUNE_EPOCH  = 15          # unfreeze backbone after this many epochs
FINETUNE_LR     = 1e-4

# ─── Evaluation ───────────────────────────────────────────────────────────────
TOPK            = (1, 5)      # compute Top-1 and Top-5 accuracy

# ─── Debug / Quick Run ────────────────────────────────────────────────────────
DEBUG           = False       # set True to run on tiny subset quickly
DEBUG_SAMPLES   = 50          # number of samples per split in debug mode

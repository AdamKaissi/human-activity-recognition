"""
train.py — Main training script for Human Activity Recognition.

Usage:
    python train.py                    # normal training
    python train.py --debug            # quick sanity check on tiny subset
    python train.py --backbone mobilenet_v2
    python train.py --epochs 50 --batch_size 16
"""

import argparse
import os
import torch

import config
from models.cnn_lstm import CNNLSTM
from utils.dataset import build_dataloaders
from utils.trainer import train
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="HAR Training — CNN + LSTM on HMDB51")
    parser.add_argument("--debug",     action="store_true", help="Run on small subset for testing")
    parser.add_argument("--backbone",  type=str, default=config.BACKBONE,
                        choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--epochs",    type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size",type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num_frames",type=int, default=config.NUM_FRAMES)
    parser.add_argument("--split_id",  type=int, default=config.SPLIT_ID, choices=[1, 2, 3])
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Override config from CLI ──
    config.DEBUG       = args.debug
    config.BACKBONE    = args.backbone
    config.EPOCHS      = args.epochs
    config.BATCH_SIZE  = args.batch_size
    config.LEARNING_RATE = args.lr
    config.NUM_FRAMES  = args.num_frames
    config.SPLIT_ID    = args.split_id

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Human Activity Recognition — HMDB51")
    print(f"  Device:    {device}")
    print(f"  Backbone:  {config.BACKBONE}")
    print(f"  Frames:    {config.NUM_FRAMES}")
    print(f"  Epochs:    {config.EPOCHS}")
    print(f"  Batch:     {config.BATCH_SIZE}")
    print(f"  Debug:     {config.DEBUG}")
    print(f"{'='*60}\n")

    # ── Data ──
    print("Loading dataset...")
    train_loader, test_loader, class_names = build_dataloaders(
        data_dir=config.DATA_DIR,
        splits_dir=config.SPLITS_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        split_id=config.SPLIT_ID,
        debug=config.DEBUG,
    )

    # ── Model ──
    model = CNNLSTM(
        num_classes=config.NUM_CLASSES,
        backbone_name=config.BACKBONE,
        freeze_backbone=config.FREEZE_BACKBONE,
    ).to(device)

    stats = model.count_parameters()
    print(f"Model parameters — Total: {stats['total']:,}  |  Trainable: {stats['trainable']:,}\n")

    # ── Train ──
    best_val_top1 = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        output_dir=config.OUTPUT_DIR,
        epochs=config.EPOCHS,
    )

    # ── Evaluate best model ──
    print("\nLoading best checkpoint for final evaluation...")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=config.OUTPUT_DIR,
    )

    print("\nFinal Results:")
    print(f"  Top-1 Accuracy  : {metrics['top1']:.2f}%")
    print(f"  Top-5 Accuracy  : {metrics['top5']:.2f}%")
    print(f"  F1 (macro)      : {metrics['f1']:.2f}%")
    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

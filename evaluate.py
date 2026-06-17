"""
evaluate.py — Standalone evaluation script.

Load a trained checkpoint and run full evaluation on the test set.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split_id 2
"""

import argparse
import os
import torch

import config
from models.cnn_lstm import CNNLSTM
from utils.dataset import build_dataloaders
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="HAR Evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split_id",  type=int, default=config.SPLIT_ID, choices=[1, 2, 3])
    parser.add_argument("--batch_size",type=int, default=config.BATCH_SIZE)
    parser.add_argument("--output_dir",type=str, default=config.OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg  = checkpoint.get("config", {})

    # Override config from saved checkpoint if available
    num_classes = saved_cfg.get("num_classes", config.NUM_CLASSES)
    backbone    = saved_cfg.get("backbone",    config.BACKBONE)
    num_frames  = saved_cfg.get("num_frames",  config.NUM_FRAMES)

    print(f"Checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"| Val Top-1: {checkpoint.get('val_top1', '?'):.1f}%")

    # Data
    _, test_loader, class_names = build_dataloaders(
        data_dir=config.DATA_DIR,
        splits_dir=config.SPLITS_DIR,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        split_id=args.split_id,
    )

    # Model
    model = CNNLSTM(num_classes=num_classes, backbone_name=backbone).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully.\n")

    # Evaluate
    metrics = evaluate(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=args.output_dir,
    )

    print(f"\nTop-1 Accuracy : {metrics['top1']:.2f}%")
    print(f"Top-5 Accuracy : {metrics['top5']:.2f}%")
    print(f"F1-Score (macro): {metrics['f1']:.2f}%")


if __name__ == "__main__":
    main()

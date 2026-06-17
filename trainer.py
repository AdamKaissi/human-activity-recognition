"""
utils/trainer.py — Training and validation loop.

Features:
  - Phase 1: Train only LSTM + classifier (backbone frozen)
  - Phase 2: Unfreeze backbone and fine-tune end-to-end
  - Top-K accuracy tracking
  - Best-model checkpointing
  - CSV + console logging
"""

import os
import time
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import config
from models.cnn_lstm import CNNLSTM


# ─── Accuracy helpers ─────────────────────────────────────────────────────────

def topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """Compute Top-K accuracy for a batch."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                      # (maxk, B)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.item() / batch_size * 100.0)
        return results  # [top1_acc%, top5_acc%]


# ─── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(
    model:      CNNLSTM,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> dict[str, float]:

    model.train() if is_train else model.eval()
    total_loss = 0.0
    top1_sum   = 0.0
    top5_sum   = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            frames  = batch["frames"].to(device, non_blocking=True)   # (B, T, C, H, W)
            labels  = batch["label"].to(device, non_blocking=True)    # (B,)

            logits = model(frames)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            top1, top5 = topk_accuracy(logits, labels, topk=config.TOPK)
            total_loss += loss.item()
            top1_sum   += top1
            top5_sum   += top5
            n_batches  += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "top1": top1_sum  / max(n_batches, 1),
        "top5": top5_sum  / max(n_batches, 1),
    }


# ─── Full training loop ───────────────────────────────────────────────────────

def train(
    model:        CNNLSTM,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    output_dir:   str = config.OUTPUT_DIR,
    epochs:       int = config.EPOCHS,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only optimise trainable params (backbone may be frozen in phase 1)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    log_path = os.path.join(output_dir, "training_log.csv")
    log_fields = ["epoch", "train_loss", "train_top1", "train_top5",
                  "val_loss", "val_top1", "val_top5", "lr", "elapsed_s"]

    best_val_top1 = 0.0
    phase2_started = False

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Phase 2: unfreeze backbone ──
            if epoch == config.FINETUNE_EPOCH and not phase2_started:
                model.unfreeze_backbone()
                for pg in optimizer.param_groups:
                    pg["lr"] = config.FINETUNE_LR
                phase2_started = True
                print(f"\n{'='*60}")
                print(f"  PHASE 2 — fine-tuning entire network at lr={config.FINETUNE_LR}")
                print(f"{'='*60}\n")

            train_stats = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            val_stats   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
            scheduler.step()

            elapsed = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]

            row = {
                "epoch":      epoch,
                "train_loss": round(train_stats["loss"], 4),
                "train_top1": round(train_stats["top1"], 2),
                "train_top5": round(train_stats["top5"], 2),
                "val_loss":   round(val_stats["loss"],   4),
                "val_top1":   round(val_stats["top1"],   2),
                "val_top5":   round(val_stats["top5"],   2),
                "lr":         current_lr,
                "elapsed_s":  round(elapsed, 1),
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"| Train Loss: {train_stats['loss']:.4f}  Top1: {train_stats['top1']:.1f}%  Top5: {train_stats['top5']:.1f}% "
                f"| Val Loss: {val_stats['loss']:.4f}  Top1: {val_stats['top1']:.1f}%  Top5: {val_stats['top5']:.1f}% "
                f"| LR: {current_lr:.2e}  Time: {elapsed:.0f}s"
            )

            # Save best checkpoint
            if val_stats["top1"] > best_val_top1:
                best_val_top1 = val_stats["top1"]
                ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_top1":   best_val_top1,
                    "config": {
                        "num_classes": config.NUM_CLASSES,
                        "backbone":    config.BACKBONE,
                        "num_frames":  config.NUM_FRAMES,
                    }
                }, ckpt_path)
                print(f"  ✓ Best model saved (Val Top-1: {best_val_top1:.1f}%)")

            # Save periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
                )

    print(f"\nTraining complete. Best Val Top-1: {best_val_top1:.1f}%")
    print(f"Log saved to: {log_path}")
    return best_val_top1

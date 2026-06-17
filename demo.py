"""
demo.py — Real-time Human Activity Recognition demo using webcam.

Captures frames in a rolling window, runs the CNN-LSTM model,
and overlays the predicted activity on the live feed.

Usage:
    python demo.py --checkpoint outputs/checkpoints/best_model.pth
    python demo.py --checkpoint outputs/checkpoints/best_model.pth --camera 0
"""

import argparse
import collections
import os
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

import config
from models.cnn_lstm import CNNLSTM
from utils.dataset import get_class_labels


# ─── Config ───────────────────────────────────────────────────────────────────

WINDOW_SIZE   = config.NUM_FRAMES   # frames in the rolling buffer
INFERENCE_FPS = 4                   # run model inference N times per second
FRAME_SIZE    = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

# Per-frame transform (no augmentation — test mode)
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def draw_overlay(frame, label: str, confidence: float, fps: float):
    """Draw prediction overlay on an OpenCV frame (BGR)."""
    h, w = frame.shape[:2]

    # Background banner
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)

    # Prediction text
    cv2.putText(frame, f"Action: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    return frame


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg   = checkpoint.get("config", {})
    num_classes = saved_cfg.get("num_classes", config.NUM_CLASSES)
    backbone    = saved_cfg.get("backbone",    config.BACKBONE)

    model = CNNLSTM(num_classes=num_classes, backbone_name=backbone).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, num_classes


# ─── Main demo loop ───────────────────────────────────────────────────────────

def run_demo(checkpoint_path: str, camera_idx: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model, num_classes = load_model(checkpoint_path, device)

    # Load class names
    label_map  = get_class_labels(config.SPLITS_DIR)
    class_names = [cls for cls, _ in sorted(label_map.items(), key=lambda x: x[1])]

    # Open webcam
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_idx}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nWebcam feed started. Press 'q' to quit.\n")

    frame_buffer = collections.deque(maxlen=WINDOW_SIZE)
    last_pred    = "Initialising..."
    last_conf    = 0.0
    last_infer   = 0.0
    infer_interval = 1.0 / INFERENCE_FPS

    fps_timer  = time.time()
    fps_frames = 0
    display_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        fps_frames += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            display_fps = fps_frames / elapsed
            fps_frames  = 0
            fps_timer   = time.time()

        # Add frame to rolling buffer (convert BGR→RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TRANSFORM(rgb)
        frame_buffer.append(tensor)

        # Run inference at controlled rate
        now = time.time()
        if len(frame_buffer) == WINDOW_SIZE and (now - last_infer) >= infer_interval:
            last_infer = now
            clip = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)  # (1, T, C, H, W)

            with torch.no_grad():
                logits = model(clip)
                probs  = torch.softmax(logits, dim=1).squeeze(0)
                conf, pred_idx = probs.max(0)

            last_pred = class_names[pred_idx.item()].replace("_", " ").title()
            last_conf = conf.item()

        # Overlay and display
        display_frame = draw_overlay(frame.copy(), last_pred, last_conf, display_fps)
        cv2.imshow("HAR — Real-Time Demo (press Q to quit)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time HAR Demo")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--camera",     type=int, default=0,
                        help="Webcam device index (default: 0)")
    args = parser.parse_args()
    run_demo(args.checkpoint, args.camera)

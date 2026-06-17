"""
models/cnn_lstm.py — CNN + LSTM model for Human Activity Recognition.

Architecture:
    1. Pre-trained ResNet-50 (or MobileNetV2) extracts per-frame spatial features.
    2. A 2-layer LSTM captures temporal dynamics across the frame sequence.
    3. A fully-connected classifier maps LSTM output to activity classes.

Input:  (B, T, C, H, W)  — batch of video clips
Output: (B, num_classes)  — raw logits
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# ─── Backbone factory ─────────────────────────────────────────────────────────

def _build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """
    Return (feature_extractor, feature_dim) for the chosen backbone.
    The classification head is removed; the extractor outputs a 1-D feature vector.
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = models.resnet50(weights=weights)
        except Exception:
            # Fallback: load architecture without pretrained weights (e.g. no internet)
            backbone = models.resnet50(weights=None)
        feature_dim = backbone.fc.in_features      # 2048
        backbone.fc = nn.Identity()                # strip classifier
        return backbone, feature_dim

    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            backbone = models.mobilenet_v2(weights=weights)
        except Exception:
            backbone = models.mobilenet_v2(weights=None)
        feature_dim = backbone.classifier[1].in_features   # 1280
        backbone.classifier = nn.Identity()
        return backbone, feature_dim

    else:
        raise ValueError(f"Unknown backbone '{name}'. Choose 'resnet50' or 'mobilenet_v2'.")


# ─── CNN-LSTM ─────────────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """
    CNN + LSTM model for video-based activity recognition.

    The CNN processes each frame independently (shared weights).
    Frame features are fed as a sequence into the LSTM.
    The final hidden state is classified.
    """

    def __init__(
        self,
        num_classes:      int  = config.NUM_CLASSES,
        backbone_name:    str  = config.BACKBONE,
        freeze_backbone:  bool = config.FREEZE_BACKBONE,
        lstm_hidden:      int  = config.LSTM_HIDDEN,
        lstm_layers:      int  = config.LSTM_LAYERS,
        lstm_dropout:     float = config.LSTM_DROPOUT,
        fc_dropout:       float = config.FC_DROPOUT,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone (shared across frames) ──
        self.backbone, feature_dim = _build_backbone(backbone_name, pretrained=True)

        if freeze_backbone:
            self._freeze_backbone()

        # ── Temporal module ──
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Classifier head ──
        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Weight init for the classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Call after initial training to fine-tune the full network."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone unfrozen — fine-tuning all parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)  video clip tensor
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Reshape to (B*T, C, H, W) for CNN
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)          # (B*T, feature_dim)

        # Reshape to (B, T, feature_dim) for LSTM
        features = features.view(B, T, -1)

        # LSTM — take only the final time step output
        lstm_out, _ = self.lstm(features)    # (B, T, hidden)
        last_hidden  = lstm_out[:, -1, :]   # (B, hidden)

        logits = self.classifier(last_hidden)  # (B, num_classes)
        return logits

    def count_parameters(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CNNLSTM(num_classes=51)
    stats = model.count_parameters()
    print(f"Total params:     {stats['total']:,}")
    print(f"Trainable params: {stats['trainable']:,}")

    # Dummy forward pass: batch=2, 16 frames, 3 channels, 224×224
    dummy = torch.randn(2, 16, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 51)

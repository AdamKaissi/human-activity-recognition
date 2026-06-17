"""
utils/dataset.py — HMDB51 Dataset loader.

Reads the official HMDB51 train/test split files, locates video files,
samples fixed-length frame sequences, and applies transforms.
"""

import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# ─── Label map ────────────────────────────────────────────────────────────────

def get_class_labels(splits_dir: str) -> dict[str, int]:
    """
    Build {class_name: index} from the split directory structure.
    HMDB51 split files are named like: brush_hair_test_split1.txt
    The class name is inferred from the unique prefixes.
    """
    split_files = sorted(glob.glob(os.path.join(splits_dir, "*_split1.txt")))
    classes = sorted({
        os.path.basename(f).rsplit("_test_split", 1)[0]
        for f in split_files
    })
    return {cls: idx for idx, cls in enumerate(classes)}


# ─── Frame Sampling ───────────────────────────────────────────────────────────

def sample_frames(video_path: str, num_frames: int) -> np.ndarray | None:
    """
    Uniformly sample `num_frames` frames from a video file.
    Returns an array of shape (num_frames, H, W, 3) in RGB, or None on error.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    # Uniformly spaced indices clamped to valid range
    indices = np.linspace(0, max(total - 1, 0), num=num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Duplicate last valid frame if read fails
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return np.stack(frames)  # (T, H, W, 3)


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return per-frame transforms for train or val/test split.
    ImageNet normalisation is applied because we use pretrained weights.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.FRAME_HEIGHT + 32, config.FRAME_WIDTH + 32)),
            transforms.RandomCrop((config.FRAME_HEIGHT, config.FRAME_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.FRAME_HEIGHT, config.FRAME_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HMDB51Dataset(Dataset):
    """
    PyTorch Dataset for HMDB51.

    Each sample is a dict:
        frames  : Tensor (T, C, H, W) — normalised frame sequence
        label   : int                 — class index
        class_  : str                 — class name string
        path    : str                 — path to source video
    """

    def __init__(
        self,
        data_dir: str,
        splits_dir: str,
        split: str = "train",
        split_id: int = 1,
        num_frames: int = config.NUM_FRAMES,
        debug: bool = False,
        debug_samples: int = config.DEBUG_SAMPLES,
    ):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.data_dir   = data_dir
        self.split      = split
        self.num_frames = num_frames
        self.transform  = get_transforms(split)
        self.label_map  = get_class_labels(splits_dir)

        self.samples: list[tuple[str, int]] = []  # (video_path, label)
        self._load_split(splits_dir, split, split_id)

        if debug:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.samples), size=min(debug_samples, len(self.samples)), replace=False)
            self.samples = [self.samples[i] for i in idx]

        print(f"[HMDB51] {split} split — {len(self.samples)} samples, {len(self.label_map)} classes")

    def _load_split(self, splits_dir: str, split: str, split_id: int):
        """Parse the official HMDB51 split txt files."""
        split_code = 1 if split == "train" else 2   # 1=train, 2=test, 0=unused

        split_files = sorted(glob.glob(
            os.path.join(splits_dir, f"*_test_split{split_id}.txt")
        ))

        for split_file in split_files:
            class_name = os.path.basename(split_file).rsplit("_test_split", 1)[0]
            if class_name not in self.label_map:
                continue
            label = self.label_map[class_name]
            class_video_dir = os.path.join(self.data_dir, class_name)

            with open(split_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    filename, code = parts[0], int(parts[1])
                    if code != split_code:
                        continue
                    video_path = os.path.join(class_video_dir, filename)
                    if os.path.exists(video_path):
                        self.samples.append((video_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video_path, label = self.samples[idx]
        class_name = os.path.basename(os.path.dirname(video_path))

        frames_np = sample_frames(video_path, self.num_frames)

        # Apply per-frame transform then stack into (T, C, H, W)
        frame_tensors = []
        for frame in frames_np:
            frame_tensors.append(self.transform(frame))
        frames = torch.stack(frame_tensors)  # (T, C, H, W)

        return {
            "frames": frames,
            "label":  torch.tensor(label, dtype=torch.long),
            "class_": class_name,
            "path":   video_path,
        }

    def get_class_names(self) -> list[str]:
        return [cls for cls, _ in sorted(self.label_map.items(), key=lambda x: x[1])]


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str = config.DATA_DIR,
    splits_dir: str = config.SPLITS_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    split_id: int = config.SPLIT_ID,
    debug: bool = config.DEBUG,
) -> tuple[DataLoader, DataLoader, list[str]]:

    train_ds = HMDB51Dataset(data_dir, splits_dir, split="train", split_id=split_id, debug=debug)
    test_ds  = HMDB51Dataset(data_dir, splits_dir, split="test",  split_id=split_id, debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_ds.get_class_names()

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

IMAGE_SIZE = (192, 256)
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def image_path(root: Path, img_id: str, laboratory: str) -> Path:
    extension = ".jpg" if laboratory == "LabA" else ".tif"
    return root / "imgs" / f"{img_id}{extension}"


class BrainOrganoidDataset(Dataset):
    def __init__(self, root: Path, records: pd.DataFrame, augment: bool = False) -> None:
        self.root = Path(root)
        self.records = records.reset_index(drop=True).copy()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.records.iloc[index]
        path = image_path(self.root, str(row.img_id), str(row.Imaging))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.root / "labels" / f"{row.img_id}.npy").astype(np.uint8)
        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if torch.rand(()) < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if torch.rand(()) < 0.2:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            contrast = float(torch.empty(1).uniform_(0.85, 1.15))
            brightness = float(torch.empty(1).uniform_(-0.05, 0.05))
            image = np.clip(image.astype(np.float32) / 255 * contrast + brightness, 0, 1)
        else:
            image = image.astype(np.float32) / 255
        image = (image - IMAGE_MEAN) / IMAGE_STD
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0).astype(np.float32))
        target = torch.from_numpy(mask[None].astype(np.float32))
        return tensor, target, index


def build_lraspp_model() -> nn.Module:
    model = lraspp_mobilenet_v3_large(
        weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
    )
    model.classifier.low_classifier = nn.Conv2d(40, 1, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, 1, kernel_size=1)
    return model


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive_fraction = target.mean().detach().clamp(0.02, 0.5)
    positive_weight = ((1 - positive_fraction) / positive_fraction).clamp(max=6.0)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
    )
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice_loss


def batch_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid() >= 0.5
    truth = target >= 0.5
    intersection = (prediction & truth).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    return (2 * intersection + 1) / (denominator + 1)


def largest_central_otsu(image: np.ndarray) -> np.ndarray:
    """Classical, training-free baseline for one central brightfield organoid."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(gray, (9, 9), 0)
    _, binary = cv2.threshold(smooth, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool)
    center = np.asarray([IMAGE_SIZE[1] / 2, IMAGE_SIZE[0] / 2])
    diagonal = float(np.linalg.norm(center))
    scores = []
    for component in range(1, count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        distance = float(np.linalg.norm(centroids[component] - center)) / diagonal
        scores.append(area / (1 + 3 * distance))
    chosen = int(np.argmax(scores)) + 1
    return labels == chosen


def acquisition_fingerprint(image: np.ndarray) -> np.ndarray:
    """Low-dimensional microscope fingerprint for explicit domain-shift gating."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB image")
    normalized = image.astype(np.float32) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    height, width = gray.shape
    channel_mean = normalized.mean(axis=(0, 1))
    channel_std = normalized.std(axis=(0, 1))
    quantiles = np.quantile(gray, [0.05, 0.25, 0.5, 0.75, 0.95])
    sharpness = np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())
    histogram = np.histogram(gray, bins=32, range=(0, 1), density=False)[0].astype(float)
    histogram /= max(histogram.sum(), 1)
    entropy = -(histogram[histogram > 0] * np.log(histogram[histogram > 0])).sum()
    return np.concatenate(
        [
            np.asarray([height / 1000, width / 1000], dtype=float),
            channel_mean,
            channel_std,
            quantiles,
            np.asarray([sharpness, entropy], dtype=float),
        ]
    )

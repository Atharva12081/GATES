from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_ooc_index(image_root: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths = [path for path in image_root.rglob("*") if path.suffix.lower() in extensions]
    by_stem: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_stem:
            raise ValueError(f"duplicate image stem: {path.stem}")
        by_stem[path.stem] = path
    frame = metadata.copy()
    frame["imageID"] = frame.imageID.astype(str)
    frame["path"] = frame.imageID.map(lambda item: str(by_stem.get(item, "")))
    missing = frame.path.eq("")
    if missing.any():
        raise FileNotFoundError(
            f"missing {int(missing.sum())} images, including {frame[missing].imageID.iloc[0]}"
        )
    if len(paths) != len(frame):
        raise ValueError(f"found {len(paths)} images for {len(frame)} metadata rows")
    frame["bad"] = (frame["Decision 1/2 (good/bad)"] == 2).astype(int)
    frame["date"] = frame.imageID.str.split("_").str[0]
    return frame


def classical_image_features(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = image.astype(np.float32) / 255
    histogram = np.histogram(normalized, bins=32, range=(0, 1), density=True)[0]
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_histogram = np.histogram(lbp, bins=10, range=(0, 10), density=True)[0]
    gradients = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    laplacian = cv2.Laplacian(normalized, cv2.CV_32F)
    quantiles = np.quantile(normalized, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    summary = np.asarray(
        [
            normalized.mean(),
            normalized.std(),
            laplacian.var(),
            np.abs(laplacian).mean(),
        ]
    )
    return np.concatenate([summary, quantiles, histogram, lbp_histogram, gradients]).astype(
        np.float32
    )


class OOCImageDataset(Dataset):
    def __init__(self, records: pd.DataFrame, *, augment: bool = False) -> None:
        self.records = records.reset_index(drop=True)
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        else:
            self.transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        with Image.open(self.records.iloc[index].path) as source:
            image = source.convert("RGB")
        return self.transform(image), int(self.records.iloc[index].bad), index


def pretrained_embeddings(
    records: pd.DataFrame,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(OOCImageDataset(records), batch_size=batch_size, num_workers=0)
    output = np.empty((len(records), 576), dtype=np.float32)
    with torch.inference_mode():
        for images, _, indices in loader:
            embedding = model(images.to(device)).cpu().numpy()
            output[indices.numpy()] = embedding
    return output


def predict_mobilenet(
    records: pd.DataFrame,
    state_dict: dict[str, torch.Tensor],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    loader = DataLoader(OOCImageDataset(records), batch_size=batch_size, num_workers=0)
    output = np.empty(len(records), dtype=np.float32)
    with torch.inference_mode():
        for images, _, indices in loader:
            output[indices.numpy()] = model(images.to(device)).squeeze(1).sigmoid().cpu().numpy()
    return output


def train_mobilenet(
    train_records: pd.DataFrame,
    development_records: pd.DataFrame,
    *,
    epochs: int = 10,
    batch_size: int = 24,
    seed: int = 20261024,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    """Fine-tune MobileNetV3 and select epochs only on development dates."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    model.to(device)
    train_loader = DataLoader(
        OOCImageDataset(train_records, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    development_loader = DataLoader(
        OOCImageDataset(development_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    positives = float(train_records.bad.sum())
    negatives = float(len(train_records) - positives)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    best_average_precision = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device)).squeeze(1)
            loss = criterion(logits, targets.float().to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        probabilities = []
        targets = []
        with torch.inference_mode():
            for images, batch_targets, _ in development_loader:
                probabilities.extend(
                    model(images.to(device)).squeeze(1).sigmoid().cpu().numpy().tolist()
                )
                targets.extend(batch_targets.numpy().tolist())
        from sklearn.metrics import average_precision_score

        average_precision = float(average_precision_score(targets, probabilities))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "development_average_precision": average_precision,
        }
        history.append(row)
        print(row, flush=True)
        if average_precision > best_average_precision:
            best_average_precision = average_precision
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("MobileNet fine-tuning produced no model")
    return best_state, history

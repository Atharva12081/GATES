from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from gates.act.brain_vision import (
    BrainOrganoidDataset,
    acquisition_fingerprint,
    batch_dice,
    build_lraspp_model,
    image_path,
    largest_central_otsu,
    seed_everything,
    segmentation_loss,
)
from gates.act.vision import (
    brain_organoid_split,
    calibrate_selective_threshold,
    grouped_bootstrap_interval,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/brain_organoid_raw/extracted/data"
PUBLISHED_PREDICTIONS = ROOT / "data/brain_organoid_raw/df_pred.csv"
MODEL_PATH = ROOT / "artifacts/models/gates_act_brain_lraspp.pt"
EVIDENCE_PATH = ROOT / "artifacts/act/canonical/brain_vision_evidence.json"
PREDICTION_PATH = ROOT / "artifacts/act/canonical/brain_vision_predictions.csv"
SEED = 20261024


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    train_records: pd.DataFrame,
    calibration_records: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[torch.nn.Module, list[dict[str, float]], torch.device]:
    seed_everything(SEED)
    device = select_device()
    train_loader = DataLoader(
        BrainOrganoidDataset(DATA_ROOT, train_records, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    calibration_loader = DataLoader(
        BrainOrganoidDataset(DATA_ROOT, calibration_records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_lraspp_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    best_dice = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)["out"]
            loss = segmentation_loss(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        dices = []
        with torch.inference_mode():
            for images, targets, _ in calibration_loader:
                logits = model(images.to(device))["out"].cpu()
                dices.extend(batch_dice(logits, targets).numpy().tolist())
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "calibration_dice": float(np.mean(dices)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["calibration_dice"] > best_dice:
            best_dice = row["calibration_dice"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a model")
    model.load_state_dict(best_state)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "history": history, "seed": SEED}, MODEL_PATH)
    return model, history, device


def load_model() -> tuple[torch.nn.Module, list[dict[str, float]], torch.device]:
    device = select_device()
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = build_lraspp_model()
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload["history"], device


def evaluate_model(
    model: torch.nn.Module,
    records: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    dataset = BrainOrganoidDataset(DATA_ROOT, records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.inference_mode():
        for images, targets, indices in loader:
            images = images.to(device)
            probability = model(images)["out"].sigmoid()
            flip_probability = torch.flip(
                model(torch.flip(images, dims=[3]))["out"].sigmoid(),
                dims=[3],
            )
            ensemble = (probability + flip_probability) / 2
            predictions = ensemble >= 0.5
            truths = targets.to(device) >= 0.5
            for item, record_index in enumerate(indices.numpy().tolist()):
                prediction = predictions[item, 0]
                truth = truths[item, 0]
                intersection = int((prediction & truth).sum().cpu())
                denominator = int(prediction.sum().cpu() + truth.sum().cpu())
                dice = (2 * intersection + 1) / (denominator + 1)
                predicted_area = float(prediction.float().mean().cpu())
                truth_area = float(truth.float().mean().cpu())
                area_error = abs(predicted_area - truth_area) / max(truth_area, 1e-8)
                tta_area = abs(
                    float((probability[item, 0] >= 0.5).float().mean().cpu())
                    - float((flip_probability[item, 0] >= 0.5).float().mean().cpu())
                ) / max(predicted_area, 1e-8)
                confidence = torch.abs(2 * ensemble[item, 0] - 1)
                uncertainty = float(torch.quantile(1 - confidence, 0.95).cpu()) + tta_area
                source = records.iloc[record_index]
                rows.append(
                    {
                        "img_id": source.img_id,
                        "org_id": int(source.org_id),
                        "clone": source.Clone,
                        "day": int(source.Day),
                        "laboratory": source.Imaging,
                        "split": source.split,
                        "model": "lraspp_lab_a_only",
                        "dice": dice,
                        "area_relative_error": area_error,
                        "uncertainty": uncertainty,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_otsu(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in records.itertuples(index=False):
        path = image_path(DATA_ROOT, str(row.img_id), str(row.Imaging))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prediction = largest_central_otsu(image)
        truth = np.load(DATA_ROOT / "labels" / f"{row.img_id}.npy").astype(np.uint8)
        truth = cv2.resize(
            truth,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        intersection = np.logical_and(prediction, truth).sum()
        denominator = prediction.sum() + truth.sum()
        predicted_area = float(prediction.mean())
        truth_area = float(truth.mean())
        rows.append(
            {
                "img_id": row.img_id,
                "org_id": int(row.org_id),
                "clone": row.Clone,
                "day": int(row.Day),
                "laboratory": row.Imaging,
                "split": row.split,
                "model": "central_otsu",
                "dice": float((2 * intersection + 1) / (denominator + 1)),
                "area_relative_error": abs(predicted_area - truth_area) / max(truth_area, 1e-8),
                "uncertainty": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(frame: pd.DataFrame) -> dict[str, object]:
    dice_ci = grouped_bootstrap_interval(frame.dice, frame.org_id)
    error_ci = grouped_bootstrap_interval(frame.area_relative_error, frame.org_id)
    return {
        "images": len(frame),
        "organoids": int(frame.org_id.nunique()),
        "mean_dice": float(frame.dice.mean()),
        "mean_dice_group_bootstrap_95ci": list(dice_ci),
        "median_dice": float(frame.dice.median()),
        "mean_area_relative_error": float(frame.area_relative_error.mean()),
        "mean_area_relative_error_group_bootstrap_95ci": list(error_ci),
        "median_area_relative_error": float(frame.area_relative_error.median()),
        "area_error_over_10_percent": float((frame.area_relative_error > 0.10).mean()),
    }


def published_summary(metadata: pd.DataFrame) -> dict[str, object]:
    published = pd.read_csv(PUBLISHED_PREDICTIONS)
    published["split"] = brain_organoid_split(published)
    protected = published[published.split.eq("test")].copy()
    result: dict[str, object] = {}
    for laboratory in ["LabA", "LabB"]:
        lab = protected[protected.Imaging.eq(laboratory)]
        result[laboratory] = {}
        for model in ["CellProfiler", "OrganoSeg", "MOrgAna_MLP_W", "SegFormer"]:
            error = (
                lab[f"pred_{model}_org_size_mum2"] - lab.org_size_mikrometer2
            ).abs() / lab.org_size_mikrometer2
            result[laboratory][model] = {
                "mean_dice": float(lab[f"pred_{model}_dice"].mean()),
                "mean_area_relative_error": float(error.mean()),
                "area_error_over_10_percent": float((error > 0.10).mean()),
                "training_scope": (
                    "training-free"
                    if model in {"CellProfiler", "OrganoSeg"}
                    else "published specimen-disjoint CV using both laboratories"
                ),
            }
    expected = set(metadata[metadata.split.eq("test")].img_id)
    if expected != set(protected.img_id):
        raise RuntimeError("published predictions do not align with protected test images")
    return result


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided paired randomization test over protected specimen means."""

    differences = np.asarray(differences, dtype=float)
    if not len(differences) or len(differences) > 20:
        raise ValueError("exact sign-flip test requires 1 to 20 pairs")
    masks = np.arange(2 ** len(differences), dtype=np.uint64)[:, None]
    bit = np.arange(len(differences), dtype=np.uint64)[None, :]
    signs = 1 - 2 * ((masks >> bit) & 1).astype(np.int8)
    randomized = (signs * differences).mean(axis=1)
    observed = abs(float(differences.mean()))
    return float((np.abs(randomized) >= observed - 1e-15).mean())


def paired_segformer_comparison(
    protected_predictions: pd.DataFrame,
) -> dict[str, object]:
    published = pd.read_csv(PUBLISHED_PREDICTIONS)
    rows: dict[str, object] = {}
    for laboratory in ["LabA", "LabB"]:
        ours = protected_predictions[
            protected_predictions.laboratory.eq(laboratory)
        ][["img_id", "org_id", "dice", "area_relative_error"]]
        reference = published[published.Imaging.eq(laboratory)].copy()
        reference["segformer_area_relative_error"] = (
            reference.pred_SegFormer_org_size_mum2 - reference.org_size_mikrometer2
        ).abs() / reference.org_size_mikrometer2
        paired = ours.merge(
            reference[
                ["img_id", "pred_SegFormer_dice", "segformer_area_relative_error"]
            ],
            on="img_id",
            validate="one_to_one",
        )
        specimen = paired.groupby("org_id").mean(numeric_only=True)
        dice_delta = specimen.dice - specimen.pred_SegFormer_dice
        error_delta = (
            specimen.area_relative_error - specimen.segformer_area_relative_error
        )
        rows[laboratory] = {
            "organoid_pairs": len(specimen),
            "lraspp_minus_segformer_mean_dice": float(dice_delta.mean()),
            "dice_two_sided_exact_sign_flip_p": exact_sign_flip_pvalue(
                dice_delta.to_numpy()
            ),
            "lraspp_dice_wins": int((dice_delta > 0).sum()),
            "lraspp_minus_segformer_mean_area_relative_error": float(
                error_delta.mean()
            ),
            "area_error_two_sided_exact_sign_flip_p": exact_sign_flip_pvalue(
                error_delta.to_numpy()
            ),
            "lraspp_area_error_wins": int((error_delta < 0).sum()),
        }
    return rows


def fingerprint_matrix(records: pd.DataFrame) -> np.ndarray:
    features = []
    for row in records.itertuples(index=False):
        path = image_path(DATA_ROOT, str(row.img_id), str(row.Imaging))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        features.append(acquisition_fingerprint(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    return np.vstack(features)


def evaluate_group_shift(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    protected: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray]:
    scaler = StandardScaler().fit(fingerprint_matrix(train))
    train_features = scaler.transform(fingerprint_matrix(train))
    calibration_features = scaler.transform(fingerprint_matrix(calibration))
    protected_features = scaler.transform(fingerprint_matrix(protected))
    detector = LedoitWolf().fit(train_features)
    calibration_score = detector.mahalanobis(calibration_features)
    threshold = float(np.quantile(calibration_score, 0.95, method="higher"))
    protected_score = detector.mahalanobis(protected_features)
    shifted = protected_score > threshold
    result: dict[str, object] = {
        "method": "Lab-A acquisition fingerprint plus Ledoit-Wolf Mahalanobis distance",
        "calibration_quantile": 0.95,
        "threshold": threshold,
        "secondary_analysis": (
            "specified after the initial external segmentation failure; not treated as an "
            "untouched primary claim"
        ),
        "protected": {},
    }
    for laboratory in ["LabA", "LabB"]:
        mask = protected.Imaging.to_numpy() == laboratory
        result["protected"][laboratory] = {
            "images": int(mask.sum()),
            "flagged_fraction": float(shifted[mask].mean()),
            "median_score": float(np.median(protected_score[mask])),
        }
    return result, shifted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-model", action="store_true")
    args = parser.parse_args()

    metadata = pd.read_csv(DATA_ROOT / "dataset_overview.csv")
    metadata["split"] = brain_organoid_split(metadata)
    train = metadata[metadata.split.eq("train") & metadata.Imaging.eq("LabA")].copy()
    calibration = metadata[metadata.split.eq("calibration") & metadata.Imaging.eq("LabA")].copy()
    protected = metadata[metadata.split.eq("test")].copy()
    if args.reuse_model:
        model, history, device = load_model()
    else:
        model, history, device = train_model(
            train,
            calibration,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    calibration_predictions = evaluate_model(model, calibration, device, args.batch_size)
    gate = calibrate_selective_threshold(
        calibration_predictions.uncertainty,
        calibration_predictions.area_relative_error > 0.10,
        maximum_risk=0.10,
    )
    protected_predictions = evaluate_model(model, protected, device, args.batch_size)
    protected_predictions["accepted"] = gate.authorized & (
        protected_predictions.uncertainty <= gate.threshold
    )
    otsu_predictions = evaluate_otsu(protected)
    group_shift, shifted = evaluate_group_shift(train, calibration, protected)
    protected_predictions["shifted"] = shifted
    protected_predictions["released"] = protected_predictions.accepted & ~shifted
    all_predictions = pd.concat(
        [calibration_predictions, protected_predictions, otsu_predictions],
        ignore_index=True,
    )
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(PREDICTION_PATH, index=False)

    protected_metrics: dict[str, object] = {}
    for model_name, model_frame in all_predictions[all_predictions.split.eq("test")].groupby(
        "model"
    ):
        protected_metrics[model_name] = {
            laboratory: metric_summary(lab_frame)
            for laboratory, lab_frame in model_frame.groupby("laboratory")
        }
    selective = {}
    for laboratory, lab in protected_predictions.groupby("laboratory"):
        accepted = lab[lab.accepted]
        released = lab[lab.released]
        selective[laboratory] = {
            "coverage": float(lab.accepted.mean()),
            "accepted_images": len(accepted),
            "mean_area_relative_error": (
                float(accepted.area_relative_error.mean()) if len(accepted) else None
            ),
            "area_error_over_10_percent": (
                float((accepted.area_relative_error > 0.10).mean()) if len(accepted) else None
            ),
            "shift_gated_coverage": float(lab.released.mean()),
            "shift_gated_images": len(released),
            "shift_gated_area_error_over_10_percent": (
                float((released.area_relative_error > 0.10).mean()) if len(released) else None
            ),
        }
    evidence = {
        "schema_version": 1,
        "dataset": {
            "doi": "10.5281/zenodo.10301912",
            "license": "CC-BY-4.0",
            "images": len(metadata),
            "organoids": int(metadata.org_id.nunique()),
            "clones": int(metadata.Clone.nunique()),
            "laboratories": sorted(metadata.Imaging.unique()),
        },
        "protocol": {
            "seed": SEED,
            "training_laboratory": "LabA",
            "protected_external_laboratory": "LabB",
            "split_organoids": metadata.drop_duplicates("org_id").split.value_counts().to_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "test_used_for_selection": False,
        },
        "training_history": history,
        "calibration_gate": {
            "threshold": gate.threshold if np.isfinite(gate.threshold) else None,
            "coverage": gate.coverage,
            "failures": gate.failures,
            "accepted": gate.accepted,
            "one_sided_95_percent_upper_risk": gate.upper_risk,
            "authorized": gate.authorized,
            "failure_definition": "absolute area error > 10%",
        },
        "protected_test": protected_metrics,
        "selective_measurement": selective,
        "group_shift_gate": group_shift,
        "published_locked_subset": published_summary(metadata),
        "paired_against_published_segformer": paired_segformer_comparison(
            protected_predictions
        ),
        "interpretation_boundary": (
            "The new LR-ASPP model is trained on Lab A only. Published learned baselines used both "
            "laboratories in specimen-disjoint cross-validation and are contextual, not a matched "
            "unseen-laboratory comparison."
        ),
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence["protected_test"], indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from gates.act.ooc_vision import build_ooc_index
from gates.act.vision import brain_organoid_split, ooc_chronological_split

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "artifacts/act/canonical"
OOC_RAW = ROOT / "data/ooc_image_raw"
BRAIN_RAW = ROOT / "data/brain_organoid_raw"


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def audit_ooc() -> dict[str, object]:
    metadata_path = OOC_RAW / "OOC_datasheet.xlsx"
    archive_path = OOC_RAW / "OOC_image_dataset.zip"
    metadata = pd.read_excel(metadata_path)
    frame = build_ooc_index(OOC_RAW / "extracted", metadata)
    frame["split"] = ooc_chronological_split(frame)
    dates = {name: set(part.date.unique()) for name, part in frame.groupby("split")}
    if dates["train"] & dates["calibration"] or dates["train"] & dates["test"]:
        raise AssertionError("OOC date leakage")
    if dates["calibration"] & dates["test"]:
        raise AssertionError("OOC calibration/test date leakage")
    prediction = pd.read_csv(CANONICAL / "ooc_vision_predictions.csv")
    expected_ids = set(frame.loc[frame.split.eq("test"), "imageID"].astype(str))
    if expected_ids != set(prediction.imageID.astype(str)):
        raise AssertionError("OOC prediction IDs do not equal protected test IDs")
    evidence = json.loads((CANONICAL / "ooc_vision_evidence.json").read_text())
    target = prediction.bad.to_numpy(int)
    probability = prediction.probability_bad.to_numpy(float)
    assert_close(roc_auc_score(target, probability), evidence["test"]["auroc"], "OOC AUROC")
    assert_close(
        average_precision_score(target, probability),
        evidence["test"]["average_precision"],
        "OOC average precision",
    )
    unsafe = prediction.unsafe_pass.astype(bool)
    if int(unsafe.sum()) != evidence["test"]["unsafe_passes"]:
        raise AssertionError("OOC unsafe-pass count mismatch")
    development = json.loads((CANONICAL / "ooc_vision_development.json").read_text())
    return {
        "archive_md5_verified": checksum(archive_path, "md5") == "8f7e058996203d48eb03b2d86c0a2e4d",
        "metadata_sha256": checksum(metadata_path),
        "images": len(frame),
        "date_groups": int(frame.date.nunique()),
        "split_date_groups": {key: len(value) for key, value in dates.items()},
        "protected_images": len(prediction),
        "selected_model": evidence["selected_model"],
        "release_authorized": development["release_gate"]["authorized"],
        "automatic_pass_coverage": evidence["test"]["automatic_pass_coverage"],
        "unsafe_passes": int(unsafe.sum()),
        "recomputed_auroc": float(roc_auc_score(target, probability)),
        "recomputed_average_precision": float(average_precision_score(target, probability)),
    }


def audit_brain() -> dict[str, object]:
    metadata_path = BRAIN_RAW / "extracted/data/dataset_overview.csv"
    metadata = pd.read_csv(metadata_path)
    metadata["split"] = brain_organoid_split(metadata)
    specimen_splits = metadata[["org_id", "split"]].drop_duplicates()
    if specimen_splits.groupby("org_id").split.nunique().max() != 1:
        raise AssertionError("brain-organoid specimen leakage")
    prediction = pd.read_csv(CANONICAL / "brain_vision_predictions.csv")
    evidence = json.loads((CANONICAL / "brain_vision_evidence.json").read_text())
    recomputed = {}
    protected = prediction[prediction.split.eq("test")]
    for model, model_frame in protected.groupby("model"):
        recomputed[model] = {}
        for laboratory, lab in model_frame.groupby("laboratory"):
            expected = evidence["protected_test"][model][laboratory]
            dice = float(lab.dice.mean())
            area_error = float(lab.area_relative_error.mean())
            assert_close(dice, expected["mean_dice"], f"{model}/{laboratory} Dice")
            assert_close(
                area_error,
                expected["mean_area_relative_error"],
                f"{model}/{laboratory} area error",
            )
            recomputed[model][laboratory] = {
                "images": len(lab),
                "organoids": int(lab.org_id.nunique()),
                "mean_dice": dice,
                "mean_area_relative_error": area_error,
            }
    return {
        "metadata_sha256": checksum(metadata_path),
        "published_predictions_sha256": checksum(BRAIN_RAW / "df_pred.csv"),
        "images": len(metadata),
        "organoids": int(metadata.org_id.nunique()),
        "split_organoids": specimen_splits.split.value_counts().to_dict(),
        "protected_recomputed": recomputed,
        "external_shift_flagged_fraction": evidence["group_shift_gate"]["protected"]["LabB"][
            "flagged_fraction"
        ],
        "external_shift_gated_coverage": evidence["selective_measurement"]["LabB"][
            "shift_gated_coverage"
        ],
    }


def main() -> None:
    result = {
        "schema": "gates-act-vision-audit-v1",
        "status": "PASS",
        "ooc": audit_ooc(),
        "brain_organoid": audit_brain(),
        "claim_boundary": (
            "Imaging experiments are independent validation modules and are not paired "
            "with the MEA acquisition benchmark."
        ),
    }
    output = CANONICAL / "vision_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

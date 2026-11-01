from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gates.act.ooc_vision import (
    build_ooc_index,
    classical_image_features,
    predict_mobilenet,
    pretrained_embeddings,
    train_mobilenet,
)
from gates.act.vision import (
    calibrate_selective_threshold,
    grouped_bootstrap_interval,
    ooc_chronological_split,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/ooc_image_raw"
EXTRACTED_ROOT = RAW_ROOT / "extracted"
MODEL_PATH = ROOT / "artifacts/models/gates_act_ooc_qc.joblib"
DEVELOPMENT_PATH = ROOT / "artifacts/act/canonical/ooc_vision_development.json"
EVIDENCE_PATH = ROOT / "artifacts/act/canonical/ooc_vision_evidence.json"
PREDICTION_PATH = ROOT / "artifacts/act/canonical/ooc_vision_predictions.csv"


def load_index() -> pd.DataFrame:
    metadata = pd.read_excel(RAW_ROOT / "OOC_datasheet.xlsx")
    frame = build_ooc_index(EXTRACTED_ROOT, metadata)
    frame["split"] = ooc_chronological_split(frame)
    calibration_dates = sorted(frame[frame.split.eq("calibration")].date.unique())
    frame["calibration_role"] = ""
    frame.loc[
        frame.split.eq("calibration") & frame.date.isin(calibration_dates[:5]),
        "calibration_role",
    ] = "development"
    frame.loc[
        frame.split.eq("calibration") & frame.date.isin(calibration_dates[5:]),
        "calibration_role",
    ] = "release"
    return frame


def cached_features(
    frame: pd.DataFrame, kind: str, cache_prefix: str = "development"
) -> np.ndarray:
    cache = RAW_ROOT / f"{cache_prefix}_{kind}_features.npz"
    if cache.exists():
        payload = np.load(cache)
        if np.array_equal(payload["image_id"].astype(str), frame.imageID.to_numpy(str)):
            return payload["features"]
    if kind == "classical":
        rows = []
        for index, path in enumerate(frame.path):
            if index % 100 == 0:
                print(f"classical features {index}/{len(frame)}", flush=True)
            rows.append(classical_image_features(path))
        features = np.vstack(rows)
    elif kind == "pretrained":
        features = pretrained_embeddings(frame)
    else:
        raise ValueError(kind)
    np.savez_compressed(cache, image_id=frame.imageID.to_numpy(str), features=features)
    return features


def candidate_models() -> dict[str, object]:
    return {
        "classical_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=7),
        ),
        "classical_random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=7,
        ),
        "pretrained_linear_probe": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=7),
        ),
    }


def fit_platt(probability: np.ndarray, target: np.ndarray) -> LogisticRegression:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=2000).fit(logit, target)


def apply_platt(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(target, prediction)),
        "bad_recall_at_0_5": float(recall_score(target, prediction)),
        "brier": float(brier_score_loss(target, probability)),
    }


def develop() -> None:
    frame = load_index()
    if frame.split.eq("test").sum() != 1373:
        raise RuntimeError("protected test cardinality changed")

    feature_sets = {
        "classical": cached_features(frame[~frame.split.eq("test")].copy(), "classical"),
        "pretrained": cached_features(frame[~frame.split.eq("test")].copy(), "pretrained"),
    }
    development_frame = frame[~frame.split.eq("test")].reset_index(drop=True)
    train_local = development_frame.split.eq("train").to_numpy()
    dev_local = development_frame.calibration_role.eq("development").to_numpy()
    release_local = development_frame.calibration_role.eq("release").to_numpy()
    models = candidate_models()
    rows = []
    fitted = {}
    for name, model in models.items():
        kind = "pretrained" if name == "pretrained_linear_probe" else "classical"
        features = feature_sets[kind]
        model.fit(features[train_local], development_frame.bad.to_numpy()[train_local])
        probability = model.predict_proba(features[dev_local])[:, 1]
        metrics = probability_metrics(development_frame.bad.to_numpy()[dev_local], probability)
        rows.append({"model": name, **metrics})
        fitted[name] = (model, kind)
    mobilenet_state, mobilenet_history = train_mobilenet(
        development_frame[train_local],
        development_frame[dev_local],
    )
    mobilenet_dev_probability = predict_mobilenet(development_frame[dev_local], mobilenet_state)
    rows.append(
        {
            "model": "finetuned_mobilenet_v3",
            **probability_metrics(
                development_frame.bad.to_numpy()[dev_local],
                mobilenet_dev_probability,
            ),
        }
    )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["average_precision", "bad_recall_at_0_5"], ascending=False
    )
    winner_name = str(leaderboard.iloc[0].model)
    if winner_name == "finetuned_mobilenet_v3":
        winner = None
        kind = "finetuned"
    else:
        winner, kind = fitted[winner_name]
    calibrated_candidates = {}
    for candidate_name, (candidate, candidate_kind) in fitted.items():
        candidate_features = feature_sets[candidate_kind]
        candidate_dev_probability = candidate.predict_proba(candidate_features[dev_local])[:, 1]
        calibrated_candidates[candidate_name] = {
            "model": candidate,
            "calibrator": fit_platt(
                candidate_dev_probability,
                development_frame.bad.to_numpy()[dev_local],
            ),
            "feature_kind": candidate_kind,
        }
    mobilenet_calibrator = fit_platt(
        mobilenet_dev_probability,
        development_frame.bad.to_numpy()[dev_local],
    )
    calibrated_candidates["finetuned_mobilenet_v3"] = {
        "state_dict": mobilenet_state,
        "calibrator": mobilenet_calibrator,
        "feature_kind": "finetuned",
    }
    if kind == "finetuned":
        dev_probability = mobilenet_dev_probability
        release_raw_probability = predict_mobilenet(
            development_frame[release_local], mobilenet_state
        )
    else:
        features = feature_sets[kind]
        dev_probability = winner.predict_proba(features[dev_local])[:, 1]
        release_raw_probability = winner.predict_proba(features[release_local])[:, 1]
    calibrator = fit_platt(dev_probability, development_frame.bad.to_numpy()[dev_local])
    release_probability = apply_platt(
        calibrator,
        release_raw_probability,
    )
    release_target = development_frame.bad.to_numpy()[release_local]
    gate = calibrate_selective_threshold(
        release_probability,
        release_target.astype(bool),
        maximum_risk=0.05,
    )
    locked_threshold = gate.threshold if gate.authorized else 0.0
    release_bad_recall = float(
        (release_probability[release_target == 1] >= locked_threshold).mean()
    )
    if release_bad_recall < 0.95:
        raise RuntimeError("release threshold did not preserve 95% bad-image recall")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": winner,
            "mobilenet_state": mobilenet_state if kind == "finetuned" else None,
            "calibrator": calibrator,
            "feature_kind": kind,
            "model_name": winner_name,
            "threshold": locked_threshold,
            "authorized": gate.authorized,
            "training_image_ids": development_frame.imageID[train_local].tolist(),
            "development_image_ids": development_frame.imageID[dev_local].tolist(),
            "release_image_ids": development_frame.imageID[release_local].tolist(),
            "candidate_models": calibrated_candidates,
        },
        MODEL_PATH,
    )
    evidence = {
        "stage": "development_locked_before_test_pixels",
        "candidate_leaderboard": leaderboard.to_dict(orient="records"),
        "selected_model": winner_name,
        "feature_kind": kind,
        "mobilenet_training_history": mobilenet_history,
        "split_counts": frame.groupby("split").size().to_dict(),
        "split_date_counts": frame.groupby("split").date.nunique().to_dict(),
        "release_gate": {
            "threshold": locked_threshold,
            "authorized": gate.authorized,
            "accepted": gate.accepted,
            "failures": gate.failures,
            "coverage": gate.coverage,
            "one_sided_95_percent_upper_risk": gate.upper_risk,
            "bad_recall": release_bad_recall,
        },
        "protected_test_pixels_accessed": False,
    }
    DEVELOPMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVELOPMENT_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2))


def evaluate_test() -> None:
    frame = load_index()
    payload = joblib.load(MODEL_PATH)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    feature_sets = {
        kind: cached_features(test, kind, cache_prefix="test")
        for kind in sorted(
            {
                item["feature_kind"]
                for item in payload["candidate_models"].values()
                if item["feature_kind"] != "finetuned"
            }
        )
    }
    if payload["feature_kind"] == "finetuned":
        raw_probability = predict_mobilenet(test, payload["mobilenet_state"])
    else:
        features = feature_sets[payload["feature_kind"]]
        raw_probability = payload["model"].predict_proba(features)[:, 1]
    probability = apply_platt(payload["calibrator"], raw_probability)
    target = test.bad.to_numpy()
    review = probability >= payload["threshold"]
    unsafe_pass = (~review) & target.astype(bool)
    predictions = test[["imageID", "date", "cell type", "bad", "split"]].copy()
    predictions["probability_bad"] = probability
    predictions["decision"] = np.where(review, "RETAKE_OR_REVIEW", "PASS")
    predictions["unsafe_pass"] = unsafe_pass
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_PATH, index=False)

    date_recall = predictions.groupby("date").apply(
        lambda part: 1 - part.unsafe_pass.sum() / max(part.bad.sum(), 1),
        include_groups=False,
    )
    recall_ci = grouped_bootstrap_interval(
        1 - unsafe_pass[target == 1].astype(float),
        test.loc[target == 1, "date"],
    )
    baseline_results = {}
    for name, candidate in payload["candidate_models"].items():
        if name == payload["model_name"]:
            candidate_raw_probability = raw_probability
        elif candidate["feature_kind"] == "finetuned":
            candidate_raw_probability = predict_mobilenet(test, candidate["state_dict"])
        else:
            candidate_features = feature_sets[candidate["feature_kind"]]
            candidate_raw_probability = candidate["model"].predict_proba(candidate_features)[:, 1]
        candidate_probability = apply_platt(
            candidate["calibrator"],
            candidate_raw_probability,
        )
        candidate_prediction = candidate_probability >= 0.5
        candidate_recall = candidate_prediction[target == 1].astype(float)
        baseline_results[name] = {
            **probability_metrics(target, candidate_probability),
            "bad_recall_date_group_bootstrap_95ci": list(
                grouped_bootstrap_interval(
                    candidate_recall,
                    test.loc[target == 1, "date"],
                )
            ),
        }
    evidence = {
        "dataset": {
            "doi": "10.5281/zenodo.10203721",
            "license_record": "CC-BY-4.0",
            "images": len(frame),
            "acquisition_date_groups": int(frame.date.nunique()),
            "cell_types": int(frame["cell type"].nunique()),
        },
        "selected_model": payload["model_name"],
        "locked_candidate_baselines_on_same_protected_test": baseline_results,
        "test": {
            **probability_metrics(target, probability),
            "images": len(test),
            "date_groups": int(test.date.nunique()),
            "bad_images": int(target.sum()),
            "bad_recall": float(1 - unsafe_pass.sum() / target.sum()),
            "bad_recall_date_group_bootstrap_95ci": list(recall_ci),
            "automatic_pass_coverage": float((~review).mean()),
            "unsafe_passes": int(unsafe_pass.sum()),
            "unsafe_risk_among_passes": float(unsafe_pass.sum() / max((~review).sum(), 1)),
            "date_level_recall_minimum": float(date_recall.min()),
            "date_level_recall_median": float(date_recall.median()),
        },
        "contextual_published_image_split": {
            "accuracy": 0.81,
            "precision": 0.79,
            "recall": 0.78,
            "comparison_warning": "published split is not documented as acquisition-date-disjoint",
        },
        "claim_boundary": (
            "Protected evaluation is chronological by acquisition date. Date is a conservative "
            "batch proxy because chip identifiers are unavailable."
        ),
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "test"])
    args = parser.parse_args()
    if args.stage == "develop":
        develop()
    else:
        evaluate_test()


if __name__ == "__main__":
    main()

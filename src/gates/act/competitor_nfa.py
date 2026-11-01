from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.model_v3 import RiskSensitiveSurface


@dataclass(frozen=True)
class NFATask:
    chemical: str
    fold: int
    levels: np.ndarray
    mean: np.ndarray
    valid: np.ndarray


def load_tasks(path: Path) -> list[NFATask]:
    data = np.load(path, allow_pickle=False)
    offsets = data["offsets"]
    tasks = []
    for index, chemical in enumerate(data["chem"]):
        start, stop = offsets[index : index + 2]
        concentration = data["logc"][start:stop]
        response = data["y"][start:stop].astype(np.float32)
        mask = data["m"][start:stop]
        levels = np.unique(concentration)
        level_index = np.searchsorted(levels, concentration)
        numerator = np.zeros((len(levels), 4, 17), dtype=float)
        denominator = np.zeros_like(numerator)
        np.add.at(numerator, level_index, response * mask)
        np.add.at(denominator, level_index, mask)
        valid = denominator > 0
        mean = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
        tasks.append(NFATask(str(chemical), int(data["fold"][index]), levels, mean, valid))
    return tasks


def contexts(task: NFATask, k: int = 3) -> list[np.ndarray]:
    """Port of NeuroChip Twin's MIT-licensed deterministic R2 design protocol."""
    seed = int(hashlib.sha256(f"{task.chemical}|{k}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for _ in range(20):
        values = tuple(sorted(rng.choice(task.levels, size=k, replace=False).tolist()))
        if values not in seen:
            seen.add(values)
            result.append(np.asarray(values, dtype=np.float32))
        if len(result) == 5:
            break
    return result


def _coordinates() -> np.ndarray:
    return np.asarray(
        [(dose / 6, output / 67) for dose, output in itertools.product(range(7), range(68))],
        dtype=float,
    )


def _training_matrix(tasks: list[NFATask]) -> np.ndarray:
    matrix = np.stack(
        [np.where(task.valid, task.mean, np.nan).reshape(-1) for task in tasks], axis=0
    )
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return np.where(np.isfinite(matrix), matrix, medians)


def _interpolation(task: NFATask, measured: np.ndarray) -> np.ndarray:
    output = np.zeros_like(task.mean)
    for cell in range(68):
        values = task.mean.reshape(7, 68)[:, cell]
        valid = task.valid.reshape(7, 68)[:, cell] & measured
        if valid.any():
            output.reshape(7, 68)[:, cell] = np.interp(
                task.levels, task.levels[valid], values[valid]
            )
    output[measured] = task.mean[measured]
    return output


def _paired(a: pd.Series, b: pd.Series, seed: int = 0) -> dict[str, object]:
    common = sorted(set(a.index) & set(b.index))
    difference = np.asarray([a[name] - b[name] for name in common])
    rng = np.random.default_rng(seed)
    draws = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
    return {
        "n_chemicals": len(common),
        "mean_difference": float(difference.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "fraction_improved": float(np.mean(difference < 0)),
    }


def run_nfa_head_to_head(task_path: Path, competitor_csv: Path, output: Path) -> dict[str, object]:
    tasks = [task for task in load_tasks(task_path) if len(task.levels) == 7]
    rows = []
    for fold in (1, 2, 3, 4):
        training = [task for task in tasks if task.fold != fold]
        surface = RiskSensitiveSurface(_coordinates(), _training_matrix(training))
        for task in (item for item in tasks if item.fold == fold):
            errors = {"gates_rank": [], "interpolation_replay": []}
            for context in contexts(task):
                measured = np.isin(task.levels, context)
                observed_mask = task.valid & measured[:, None, None]
                observed = np.flatnonzero(observed_mask.reshape(-1))
                truth = task.mean.reshape(-1)
                prediction = surface.fit_predict(
                    observed.tolist(), truth[observed], effect_threshold=0.40
                ).mean.reshape(7, 4, 17)
                interpolation = _interpolation(task, measured)
                hidden = task.valid & ~measured[:, None, None]
                errors["gates_rank"].append(
                    float(np.mean(np.abs(prediction[hidden] - task.mean[hidden])))
                )
                errors["interpolation_replay"].append(
                    float(np.mean(np.abs(interpolation[hidden] - task.mean[hidden])))
                )
            rows.append(
                {
                    "chemical": task.chemical,
                    "fold": fold,
                    **{name: float(np.mean(value)) for name, value in errors.items()},
                }
            )
    frame = pd.DataFrame(rows)
    public = pd.read_csv(competitor_csv)
    public = public.loc[
        public["fold"].isin([1, 2, 3, 4])
        & public["k"].eq(3)
        & public["method"].isin(["anchorboost", "loglinear_interp"])
    ].pivot(index=["chemical", "fold"], columns="method", values="curve_mae")
    frame = frame.merge(public.reset_index(), on=["chemical", "fold"], validate="one_to_one")
    indexed = frame.set_index("chemical")
    fold_differences = {
        str(fold): float((part["gates_rank"] - part["anchorboost"]).mean())
        for fold, part in frame.groupby("fold")
    }
    result = {
        "status": "diagnostic",
        "subset": "exactly seven concentration levels; public folds 1-4; k=3",
        "chemicals": len(frame),
        "means": {
            column: float(frame[column].mean())
            for column in (
                "gates_rank",
                "interpolation_replay",
                "anchorboost",
                "loglinear_interp",
            )
        },
        "interpolation_replay_max_abs_gap": float(
            np.max(np.abs(frame["interpolation_replay"] - frame["loglinear_interp"]))
        ),
        "gates_vs_anchorboost": _paired(indexed["gates_rank"], indexed["anchorboost"]),
        "fold_differences": fold_differences,
        "fold_wins": int(sum(value < 0 for value in fold_differences.values())),
    }
    comparison = result["gates_vs_anchorboost"]
    result["gate"] = (
        "GO"
        if comparison["mean_difference"] < 0
        and comparison["ci95"][1] < 0
        and result["fold_wins"] >= 4
        else "NO-GO"
    )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "h1b_nfa_per_chemical.csv", index=False)
    (output / "h1b_nfa_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result

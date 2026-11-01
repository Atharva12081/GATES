from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.data import assay_grid, coordinate_matrix, load_pi_assay, response_vector
from gates.act.model import AdaptiveResponseSurface, select_initial_design

STRATEGIES = ("random", "maximin", "max_variance", "act")


def _split_groups(replicates: list[int], test: int) -> tuple[list[int], list[int]]:
    remaining = [replicate for replicate in replicates if replicate != test]
    offset = replicates.index(test)
    rotated = remaining[offset % len(remaining) :] + remaining[: offset % len(remaining)]
    return rotated[:3], rotated[3:]


def _trace(
    surface: AdaptiveResponseSurface,
    target: np.ndarray,
    strategy: str,
    max_budget: int,
    seed: int,
) -> list[dict[str, float | int]]:
    observed = select_initial_design(surface.coordinates)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    while True:
        prediction = surface.fit_predict(observed, target[observed])
        hidden = np.array([i for i in range(len(target)) if i not in observed])
        error = prediction.mean[hidden] - target[hidden]
        rows.append(
            {
                "budget": len(observed),
                "nrmse": float(np.sqrt(np.mean(error**2)) / surface.response_scale),
                "mae": float(np.mean(np.abs(error))),
                "mean_uncertainty": prediction.uncertainty_score,
                "misfit_score": prediction.misfit_score,
                "next_index": -1,
            }
        )
        if len(observed) >= max_budget:
            break
        next_index = surface.select_next(prediction, observed, strategy, rng)
        rows[-1]["next_index"] = next_index
        observed.append(next_index)
    return rows


def _calibration_rule(
    traces: list[list[dict[str, float | int]]], target_nrmse: float
) -> tuple[float, float, int]:
    by_budget = zip(*traces, strict=True)
    for group_rows in by_budget:
        if all(float(row["nrmse"]) <= target_nrmse for row in group_rows):
            uncertainty = max(float(row["mean_uncertainty"]) for row in group_rows)
            misfit = max(float(row["misfit_score"]) for row in group_rows) * 1.10
            return uncertainty, misfit, int(group_rows[0]["budget"])
    return -1.0, -1.0, int(traces[0][-1]["budget"])


def _paired_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def run_act_benchmark(
    raw_dir: Path,
    output_dir: Path,
    max_budget: int = 30,
    target_nrmse: float = 0.12,
    seed: int = 20261023,
) -> dict[str, object]:
    frame = load_pi_assay(raw_dir)
    grid = assay_grid(frame)
    coordinates = coordinate_matrix(grid)
    replicates = sorted(frame["replicate"].unique().tolist())
    responses = {replicate: response_vector(frame, replicate, grid) for replicate in replicates}
    trace_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    for test in replicates:
        training, calibration = _split_groups(replicates, test)
        training_matrix = np.vstack([responses[group] for group in training])
        for strategy in STRATEGIES:
            surface = AdaptiveResponseSurface(coordinates, training_matrix, seed + test)
            calibration_traces = [
                _trace(surface, responses[group], strategy, max_budget, seed + 100 * group)
                for group in calibration
            ]
            uncertainty_threshold, ood_threshold, calibration_budget = _calibration_rule(
                calibration_traces, target_nrmse
            )
            test_trace = _trace(surface, responses[test], strategy, max_budget, seed + 1000 * test)
            decision = "CONTINUE"
            terminal = test_trace[-1]
            if strategy == "act" and uncertainty_threshold >= 0:
                for row in test_trace:
                    if float(row["misfit_score"]) > ood_threshold:
                        decision = "ABSTAIN"
                        terminal = test_trace[-1]
                        break
                    if float(row["mean_uncertainty"]) <= uncertainty_threshold:
                        decision = "STOP"
                        terminal = row
                        break
            decision_budget = int(terminal["budget"])
            measurements_consumed = len(grid) if decision == "ABSTAIN" else decision_budget
            terminal_nrmse = 0.0 if decision == "ABSTAIN" else float(terminal["nrmse"])
            for row in test_trace:
                trace_rows.append(
                    {
                        "test_replicate": test,
                        "strategy": strategy,
                        "train_replicates": ";".join(map(str, training)),
                        "calibration_replicates": ";".join(map(str, calibration)),
                        **row,
                    }
                )
            decision_rows.append(
                {
                    "test_replicate": test,
                    "strategy": strategy,
                    "decision": decision,
                    "decision_budget": decision_budget,
                    "measurements_consumed": measurements_consumed,
                    "terminal_nrmse": terminal_nrmse,
                    "calibration_budget": calibration_budget,
                    "uncertainty_threshold": uncertainty_threshold,
                    "ood_threshold": ood_threshold,
                    "saved_fraction": 1.0 - measurements_consumed / len(grid),
                }
            )

    traces = pd.DataFrame(trace_rows)
    decisions = pd.DataFrame(decision_rows)
    budget_auc = (
        traces.groupby(["test_replicate", "strategy"], as_index=False)["nrmse"]
        .mean()
        .rename(columns={"nrmse": "mean_nrmse_over_budget"})
    )
    act = budget_auc.loc[budget_auc["strategy"] == "act"].set_index("test_replicate")
    comparisons = []
    for baseline in ("random", "maximin", "max_variance"):
        other = budget_auc.loc[budget_auc["strategy"] == baseline].set_index("test_replicate")
        differences = (other["mean_nrmse_over_budget"] - act["mean_nrmse_over_budget"]).to_numpy()
        rng = np.random.default_rng(seed)
        boot = np.array(
            [rng.choice(differences, len(differences), replace=True).mean() for _ in range(10_000)]
        )
        comparisons.append(
            {
                "baseline": baseline,
                "act_improvement_mean_nrmse": float(differences.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "paired_sign_flip_p": _paired_sign_flip_pvalue(differences),
                "groups": len(differences),
            }
        )
    comparisons_frame = pd.DataFrame(comparisons)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(output_dir / "acquisition_traces.csv", index=False)
    decisions.to_csv(output_dir / "gates_act_decisions.csv", index=False)
    comparisons_frame.to_csv(output_dir / "paired_comparisons.csv", index=False)
    grid.assign(grid_index=np.arange(len(grid))).to_csv(output_dir / "assay_grid.csv", index=False)
    act_decisions = decisions.loc[decisions["strategy"] == "act"]
    stopped = act_decisions.loc[act_decisions["decision"] == "STOP"]
    summary = {
        "dataset": "OrganoID gemcitabine PI fluorescence",
        "groups": len(replicates),
        "grid_cells_per_group": len(grid),
        "max_budget": max_budget,
        "target_nrmse": target_nrmse,
        "validation": "leave-one-replicate-out; three training and two calibration groups",
        "act_mean_nrmse_over_budget": float(
            budget_auc.loc[budget_auc["strategy"] == "act", "mean_nrmse_over_budget"].mean()
        ),
        "decisions": act_decisions["decision"].value_counts().to_dict(),
        "stopped_tolerance_violations": int((stopped["terminal_nrmse"] > target_nrmse).sum()),
        "retrospective_grid_measurement_reduction": float(
            1.0 - act_decisions["measurements_consumed"].sum() / (len(replicates) * len(grid))
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

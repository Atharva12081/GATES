from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.benchmark_v3 import (
    POLICIES,
    V3Config,
    _apply_conformal,
    _load_sources,
    _paired_sign_flip,
    _trace,
    _training_matrix,
)
from gates.act.brown_mea import load_brown_mea
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.neural_mea import neural_coordinates
from gates.act.stopping_v3 import _risk
from gates.act.tipping_mea import load_tipping_mea, tipping_surface_arrays


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(20_000)]
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run_tipping_confirmation(
    legacy_workbook: Path,
    phase1_workbook: Path,
    brown_csv: Path,
    tipping_csvs: list[Path],
    model_lock_path: Path,
    stopping_lock_path: Path,
    output_dir: Path,
):
    model_lock = json.loads(model_lock_path.read_text())
    stop_lock = json.loads(stopping_lock_path.read_text())
    config = V3Config(**model_lock["selected_config"])
    seed = int(model_lock["seed"])
    legacy, phase1, meta, source_responses = _load_sources(legacy_workbook, phase1_workbook)
    brown = load_brown_mea(brown_csv)
    excluded = set(legacy.surfaces["compound_group"])
    excluded.update(phase1.surfaces["compound_group"])
    excluded.update(brown.surfaces["compound_group"])
    tipping = load_tipping_mea(tipping_csvs, excluded)
    test_meta, test_responses = tipping_surface_arrays(tipping)
    training = _training_matrix(meta, source_responses)
    coordinates = neural_coordinates()
    rows = []
    decisions = []
    for record in test_meta.itertuples(index=False):
        truth = test_responses[record.run_id]
        surface = RiskSensitiveSurface(
            coordinates,
            training,
            config.covariance_blend,
            config.noise_fraction,
            config.analog_weight,
            config.neighbor_count,
        )
        for policy in POLICIES:
            rule = stop_lock["rules"][policy]
            path = []
            for row in _trace(surface, truth, policy, record.run_id, seed, config):
                band = _apply_conformal(
                    row, truth, float(rule["conformal_q"]), config.effect_threshold
                )
                risk_score = _risk(
                    row, stop_lock["risk_weights"], float(band["mean_interval_width"])
                )
                serial = (
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"mean", "std", "hidden_indices", "required_q"}
                    }
                    | band
                    | {"risk_score": risk_score}
                )
                path.append(serial)
                rows.append(
                    {
                        "run_id": record.run_id,
                        "source": record.source,
                        "compound": record.compound,
                        "compound_group": record.compound_group,
                        "policy": policy,
                        **serial,
                    }
                )
            path_frame = pd.DataFrame(path)
            terminal = path_frame.iloc[-1].copy()
            decision = "COMPLETE"
            for _, row in path_frame.loc[path_frame["round"].isin([2, 3])].iterrows():
                certificate = (
                    row["observed_effect_divs"] >= config.minimum_effect_divs
                    or row["max_effect_upper"] < config.effect_threshold
                )
                if row["round"] == 2 and row["shift_score"] > rule["ood_threshold"]:
                    decision = "ABSTAIN"
                    break
                if (
                    rule["risk_threshold"] >= 0
                    and certificate
                    and row["risk_score"] <= rule["risk_threshold"]
                ):
                    decision = "STOP"
                    terminal = row
                    break
            if decision in {"COMPLETE", "ABSTAIN"}:
                terminal = path_frame.iloc[-1].copy()
                terminal["round"] = 4
                terminal["div"] = 12
                terminal["selected_cells"] = 32
                terminal["nrmse"] = 0.0
                terminal["effect_recall"] = 1.0
                terminal["missed_effect_cells"] = 0
                terminal["compound_effect_missed"] = 0
            decisions.append(
                {
                    "run_id": record.run_id,
                    "source": record.source,
                    "compound": record.compound,
                    "compound_group": record.compound_group,
                    "policy": policy,
                    "decision": decision,
                    "decision_round": int(terminal["round"]),
                    "decision_div": int(terminal["div"]),
                    "selected_cells_used": int(terminal["selected_cells"]),
                    "physical_wells_allocated": 24,
                    "instrument_well_recordings": 24 * int(terminal["round"]),
                    "recording_sessions": int(terminal["round"]),
                    "terminal_nrmse": float(terminal["nrmse"]),
                    "effect_recall": float(terminal["effect_recall"]),
                    "missed_effect_cells": int(terminal["missed_effect_cells"]),
                    "compound_effect_missed": int(terminal["compound_effect_missed"]),
                    "stop_safe": bool(
                        terminal["nrmse"] <= config.target_nrmse
                        and terminal["effect_recall"] >= config.target_effect_recall
                    ),
                }
            )
    traces = pd.DataFrame(rows)
    decision_frame = pd.DataFrame(decisions)
    group = (
        traces.loc[traces["round"].isin([2, 3])]
        .groupby(["compound_group", "policy"], as_index=False)
        .agg(effect_recall=("effect_recall", "mean"), nrmse=("nrmse", "mean"))
    )
    rs = group.loc[group.policy.eq("risk_sensitive")].set_index("compound_group")
    rng = np.random.default_rng(seed)
    comparison_rows = []
    for baseline in POLICIES[1:]:
        other = group.loc[group.policy.eq(baseline)].set_index("compound_group")
        common = rs.index.intersection(other.index)
        recall = (rs.loc[common, "effect_recall"] - other.loc[common, "effect_recall"]).to_numpy()
        nrmse = (other.loc[common, "nrmse"] - rs.loc[common, "nrmse"]).to_numpy()
        recall_ci = _bootstrap_interval(recall, rng)
        nrmse_ci = _bootstrap_interval(nrmse, rng)
        comparison_rows.append(
            {
                "baseline": baseline,
                "effect_recall_advantage": float(recall.mean()),
                "effect_recall_ci_low": recall_ci[0],
                "effect_recall_ci_high": recall_ci[1],
                "effect_recall_sign_flip_p": _paired_sign_flip(recall, seed),
                "nrmse_advantage": float(nrmse.mean()),
                "nrmse_ci_low": nrmse_ci[0],
                "nrmse_ci_high": nrmse_ci[1],
                "nrmse_sign_flip_p": _paired_sign_flip(nrmse, seed),
                "test_groups": len(common),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(output_dir / "tipping_fresh_traces.csv", index=False)
    decision_frame.to_csv(output_dir / "tipping_fresh_decisions.csv", index=False)
    comparisons.to_csv(output_dir / "tipping_fresh_comparisons.csv", index=False)
    (output_dir / "tipping_data_audit.json").write_text(
        json.dumps(tipping.audit, indent=2, sort_keys=True) + "\n"
    )
    risk = decision_frame.loc[decision_frame.policy.eq("risk_sensitive")]
    stops = risk.loc[risk.decision.eq("STOP")]
    summary = {
        "schema": "gates-act-v3-tipping-confirmation-v1",
        "dataset": "EPA Tipping Point longitudinal neural MEA source data",
        "source_doi": "10.23719/1407690",
        "fresh_compound_groups": int(risk.compound_group.nunique()),
        "fresh_runs": int(risk.run_id.nunique()),
        "risk_sensitive_decisions": risk.decision.value_counts().to_dict(),
        "risk_sensitive_stops": int(len(stops)),
        "risk_sensitive_unsafe_stops": int((~stops.stop_safe).sum()),
        "risk_sensitive_stop_effect_recall": float(stops.effect_recall.mean())
        if len(stops)
        else None,
        "risk_sensitive_missed_effect_cells_at_stop": int(stops.missed_effect_cells.sum()),
        "mean_recording_sessions": float(risk.recording_sessions.mean()),
        "recording_session_reduction": float(1 - risk.recording_sessions.mean() / 4),
        "measurement_cost_contract": (
            "24 physical wells allocated; savings count only avoided whole-plate DIV sessions"
        ),
        "model_lock_sha256": stop_lock["model_lock_sha256"],
        "stopping_lock_sha256": __import__("hashlib")
        .sha256(stopping_lock_path.read_bytes())
        .hexdigest(),
    }
    (output_dir / "tipping_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary

from __future__ import annotations

import numpy as np


def stopping_metrics(records: list[dict[str, object]], final_time: int) -> dict[str, float]:
    if not records:
        return {"eesr": 0.0, "coverage": 0.0, "observation_savings": 0.0}
    stopped = [record for record in records if record["decision"] == "STOP"]
    errors = sum(int(record["prediction"] != record["label"]) for record in stopped)
    savings = [1 - (float(record["decision_time"]) / final_time) for record in stopped]
    return {
        "eesr": errors / len(stopped) if stopped else 0.0,
        "coverage": len(stopped) / len(records),
        "observation_savings": float(np.mean(savings)) if savings else 0.0,
        "mean_decision_time": float(np.mean([float(record["decision_time"]) for record in stopped]))
        if stopped
        else float(final_time),
        "stopped": float(len(stopped)),
        "total": float(len(records)),
    }

from __future__ import annotations

from pathlib import Path

import numpy as np

from gates.act.benchmark_v2 import _metrics, _select_batch
from gates.act.model import AdaptiveResponseSurface
from gates.act.neural_mea import (
    _canonical_compound,
    _stable_split,
    load_neural_mea,
    neural_coordinates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_biological_repeat_aliases_share_split() -> None:
    assert _canonical_compound("Rotenone - ToxCast G-8") == _canonical_compound("Rotenone")
    assert _canonical_compound("Aspirin") == _canonical_compound("Acetylsalicylic acid")
    assert _stable_split(_canonical_compound("Aspirin")) == _stable_split(
        _canonical_compound("Acetylsalicylic acid")
    )


def test_chronological_batch_cannot_select_future_div() -> None:
    coordinates = neural_coordinates()
    training = np.zeros((3, 32))
    surface = AdaptiveResponseSurface(coordinates, training)
    observed = list(range(8))
    prediction = surface.fit_predict(observed, np.zeros(8))
    selected = _select_batch(
        surface,
        prediction,
        observed,
        time_rank=1,
        policy="act",
        rng=np.random.default_rng(1),
    )
    assert len(selected) == 4
    assert all(8 <= index < 16 for index in selected)


def test_effect_miss_is_counted_separately_from_error() -> None:
    truth = np.array([0.0, 0.8, 0.0])
    prediction = np.zeros(3)
    result = _metrics(prediction, truth, observed=[0], scale=1.0, effect_threshold=0.4)
    assert result["missed_effect_cells"] == 1
    assert result["compound_effect_missed"] == 1
    assert result["effect_recall"] == 0.0


def test_split_labels_have_expected_domain() -> None:
    labels = {_stable_split(f"compound-{index}") for index in range(100)}
    assert labels == {"train", "calibration", "test"}


def test_real_neural_groups_never_cross_splits() -> None:
    workbook = ROOT / "data" / "neural_mea_raw" / "All_Compound_Summary.xlsx"
    data = load_neural_mea(workbook)
    roles_per_group = data.surfaces.groupby("compound_group")["split"].nunique()
    assert roles_per_group.max() == 1
    assert data.audit["eligible_complete_runs"] == 135
    assert data.audit["split_groups"] == {"calibration": 21, "test": 26, "train": 80}

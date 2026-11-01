import numpy as np
import pandas as pd

from gates.act.vision import (
    brain_organoid_split,
    calibrate_selective_threshold,
    dice_score,
    one_sided_binomial_upper,
    ooc_chronological_split,
)


def test_brain_split_is_specimen_disjoint_and_clone_balanced() -> None:
    frame = pd.DataFrame(
        [
            {"org_id": org_id, "Clone": clone, "image": image}
            for clone_index, clone in enumerate(["a", "b", "c", "d"])
            for org_id in range(clone_index * 16 + 1, clone_index * 16 + 17)
            for image in range(3)
        ]
    )
    frame["split"] = brain_organoid_split(frame)
    assert frame.groupby("org_id")["split"].nunique().max() == 1
    counts = frame.drop_duplicates("org_id").groupby(["Clone", "split"]).size()
    assert set(counts.xs("train", level="split")) == {9}
    assert set(counts.xs("calibration", level="split")) == {3}
    assert set(counts.xs("test", level="split")) == {4}


def test_ooc_split_keeps_dates_intact() -> None:
    frame = pd.DataFrame(
        {"imageID": [f"{date:06d}_{image:02d}" for date in range(59) for image in range(2)]}
    )
    frame["split"] = ooc_chronological_split(frame)
    frame["date"] = frame.imageID.str.split("_").str[0]
    assert frame.groupby("date")["split"].nunique().max() == 1
    assert frame.groupby("split")["date"].nunique().to_dict() == {
        "calibration": 10,
        "test": 10,
        "train": 39,
    }


def test_dice_and_risk_calibration() -> None:
    assert dice_score(np.array([0, 1, 1]), np.array([0, 1, 0])) == 2 / 3
    assert one_sided_binomial_upper(0, 59) <= 0.05
    result = calibrate_selective_threshold(
        np.linspace(0, 1, 100),
        np.zeros(100, dtype=bool),
        maximum_risk=0.05,
    )
    assert result.authorized
    assert result.accepted == 100

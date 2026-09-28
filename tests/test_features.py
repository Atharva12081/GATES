import pandas as pd

from gates.features.prefix import prefix_features


def test_prefix_excludes_future_information() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": ["a", "a"],
            "dosage": [0.0, 0.0],
            "replicate": [0, 0],
            "time": [0, 4],
            "viability": [0.8, 0.8],
            "pi_fluorescence": [10.0, 999.0],
            "organoid_count": [1.0, 1.0],
            "organoid_fluorescence": [1.0, 1.0],
            "area_mean": [1.0, 1.0],
            "area_total": [1.0, 1.0],
            "circularity_mean": [1.0, 1.0],
            "solidity_mean": [1.0, 1.0],
            "eccentricity_mean": [1.0, 1.0],
        }
    )
    features = prefix_features(frame, 0)
    assert features.loc[0, "pi_fluorescence__last"] == 10.0
    assert features.loc[0, "observations_used"] == 1

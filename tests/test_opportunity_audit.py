import numpy as np
import pandas as pd

from gates.act.opportunity_audit import _agreement_score, _safe_binary_metrics


def test_agreement_score_requires_same_sign_and_uses_weaker_model():
    frame = pd.DataFrame(
        {
            "gates": [0.6, 0.6, -0.7, np.nan],
            "selected_residual": [0.5, -0.5, -0.4, 0.8],
        }
    )
    assert _agreement_score(frame).tolist() == [0.5, 0.0, 0.4, 0.0]


def test_binary_metrics_are_exact_for_perfect_ordering():
    result = _safe_binary_metrics(pd.Series([False, False, True]), pd.Series([0.1, 0.2, 0.9]))
    assert result == {"auroc": 1.0, "average_precision": 1.0}

from __future__ import annotations

import pandas as pd
import pytest

from gates.data.retinal import decode_binary_labels


def test_decode_binary_labels_normalizes_documented_values() -> None:
    result = decode_binary_labels(
        pd.Series(["yes", "no", "YES", "No"]), column="endpoint"
    )

    assert result.tolist() == [True, False, True, False]
    assert result.dtype == bool


def test_decode_binary_labels_rejects_unexpected_value() -> None:
    with pytest.raises(ValueError, match="unsupported labels.*maybe"):
        decode_binary_labels(pd.Series(["yes", "maybe"]), column="endpoint")

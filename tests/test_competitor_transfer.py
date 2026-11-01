import numpy as np

from gates.act.competitor_transfer import CONTEXTS, _features, _interpolate, _observed


def test_contexts_are_35_unique_sixteen_cell_designs():
    assert len(CONTEXTS) == 35
    assert len(set(CONTEXTS)) == 35
    for context in CONTEXTS:
        observed = _observed(context)
        assert len(observed) == len(set(observed)) == 16
        assert set(observed % 8) == {0, *context}


def test_interpolation_preserves_observed_and_is_finite():
    truth = np.linspace(-1, 1, 32)
    observed = _observed((1, 4, 7))
    prediction = _interpolate(truth, observed)
    np.testing.assert_allclose(prediction[observed], truth[observed])
    assert np.isfinite(prediction).all()


def test_features_have_one_row_per_hidden_cell():
    truth = np.sin(np.arange(32) / 4)
    observed = _observed((2, 4, 6))
    anchor = _interpolate(truth, observed)
    x, residual, hidden = _features(truth, observed, anchor, anchor, np.ones(32) * 0.2)
    assert x.shape == (16, 18)
    assert residual.shape == hidden.shape == (16,)
    assert np.isfinite(x).all()

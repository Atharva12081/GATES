import numpy as np

from gates.act.competitor_nfa import NFATask, _coordinates, _interpolation, contexts


def task() -> NFATask:
    mean = np.arange(7 * 4 * 17, dtype=float).reshape(7, 4, 17)
    return NFATask("example", 1, np.arange(7, dtype=float), mean, np.ones_like(mean, bool))


def test_coordinates_cover_seven_by_sixty_eight_grid():
    coordinates = _coordinates()
    assert coordinates.shape == (476, 2)
    np.testing.assert_allclose(coordinates.min(axis=0), 0)
    np.testing.assert_allclose(coordinates.max(axis=0), 1)


def test_contexts_are_deterministic_unique_triplets():
    first = contexts(task())
    second = contexts(task())
    assert len(first) == 5
    assert all(len(value) == 3 for value in first)
    assert [tuple(value) for value in first] == [tuple(value) for value in second]
    assert len({tuple(value) for value in first}) == 5


def test_interpolation_preserves_measured_levels():
    item = task()
    measured = np.array([True, False, False, True, False, False, True])
    prediction = _interpolation(item, measured)
    np.testing.assert_allclose(prediction[measured], item.mean[measured])

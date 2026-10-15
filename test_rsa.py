import numpy as np

from rsa import rsa


def test_rsa_identical_inputs():
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    assert np.isclose(rsa(X, X), 1.0)


def test_rsa_different_inputs_changes_score():
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    Y = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    score = rsa(X, Y)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0

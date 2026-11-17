"""Small Phase 4 helpers: error bars, calibration, and turning or sliding wafers."""
import numpy as np
import pandas as pd
from scipy import ndimage


def lot_bootstrap(lots, score, rounds=1000, seed=42):
    """Redraw whole lots with replacement `rounds` times and return score(rows) for each draw.

    We redraw lots, not single wafers, because wafers from one lot look alike.
    Same lots + same seed = same draws, so two models scored this way can be compared draw by draw.
    """
    groups = list(pd.Series(lots).groupby(lots).indices.values())
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        picked = rng.integers(len(groups), size=len(groups))
        scores.append(score(np.concatenate([groups[i] for i in picked])))
    return np.array(scores)


def expected_calibration_error(probs, labels, classes, bins=15):
    """Average gap between how sure the model is and how often it's right (0 = perfectly honest)."""
    confidence = probs.max(axis=1)
    correct = np.asarray(classes)[probs.argmax(axis=1)] == labels
    which_bin = np.minimum((confidence * bins).astype(int), bins - 1)
    error = 0.0
    for b in range(bins):
        inside = which_bin == b
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return error


def rotate_map(image, angle):
    """Turn an already-prepared 32x32 map by `angle` degrees, picking the nearest die (no blending).

    We rotate after preprocessing, so the denoising step sees the same wafer at every angle.
    Quarter turns are exact; anything that turns off the grid becomes 'outside the wafer' (0).
    """
    return ndimage.rotate(image, angle, order=0, reshape=False, cval=0)


def shift_map(image, dy, dx):
    """Slide a wafer image by (dy, dx) pixels. The empty strip becomes 'outside the wafer' (0)."""
    return ndimage.shift(image, (dy, dx), order=0, cval=0)

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


def rotate_map(raw, angle, size=32):
    """Turn a raw wafer map by `angle` degrees and shrink it to size x size, in one nearest-die step.

    Doing both at once means 0 degrees gives exactly the same picture as plain resizing.
    Dies that turn off the edge of the grid are dropped (0 = outside the wafer).
    """
    raw = np.asarray(raw)
    h, w = raw.shape
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    t = np.deg2rad(angle)
    from_y, from_x = np.cos(t) * y - np.sin(t) * x, np.sin(t) * y + np.cos(t) * x
    rows = np.floor((from_y + 1) / 2 * (h - 1) + 0.5).astype(int)
    cols = np.floor((from_x + 1) / 2 * (w - 1) + 0.5).astype(int)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.zeros((size, size), dtype=raw.dtype)
    out[inside] = raw[rows[inside], cols[inside]]
    return out


def shift_map(image, dy, dx):
    """Slide a wafer image by (dy, dx) pixels. The empty strip becomes 'outside the wafer' (0)."""
    return ndimage.shift(image, (dy, dx), order=0, cval=0)

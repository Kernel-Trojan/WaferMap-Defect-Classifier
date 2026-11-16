"""Step 2: clean up each wafer map and shrink it to a fixed size.

A raw map has 0 = off the wafer, 1 = passing die, 2 = failing die, and every
map has its own size. We turn each one into two small binary pictures:
  fail map  (1 = failing die)  and  wafer mask  (1 = on the wafer).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, median_filter

# 3x3 box with the middle switched off: counts the 8 neighbours of each die.
NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def denoise(fail: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """Remove speckle noise from a fail map.

    median   - 3x3 median filter. Strong, but it can wipe out 1-die-thin scratches.
    isolated - only flip a failing die if none of its 8 neighbours fail.
               Lines survive because every die in a line has a failing neighbour.
    none     - leave it alone.
    """
    if method == "none":
        return fail
    if method == "median":
        return median_filter(fail, size=3, mode="constant") * mask
    if method == "isolated":
        lonely = convolve(fail, NEIGHBOURS, mode="constant") == 0
        return fail * ~lonely
    raise ValueError(f"unknown denoise method {method}")


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize: just pick the closest die, never blend.

    Blending (bilinear) would invent "half failing" dies. Same index trick as
    the baselines, so both see the same 32x32 grid.
    """
    rows = np.floor(np.linspace(0, image.shape[0] - 1, size) + 0.5).astype(int)
    cols = np.floor(np.linspace(0, image.shape[1] - 1, size) + 0.5).astype(int)
    return image[rows[:, None], cols]


def prepare(wafer_maps, size: int = 32, method: str = "isolated"):
    """Turn raw wafer maps into (fail maps, masks, extra features).

    Extra features per wafer (worked out on the full-size map, before resizing):
      - fraction of on-wafer dies that fail
      - fraction of failing dies that denoising removed
    We keep these because the Random class is basically "lots of scattered noise",
    so denoising alone would erase the very thing that defines it.
    """
    fails, masks, extras = [], [], []
    for wafer_map in wafer_maps:
        raw = np.asarray(wafer_map)
        mask = (raw > 0).astype(np.uint8)
        fail = (raw == 2).astype(np.uint8)
        clean = denoise(fail, mask, method)
        # int() matters: these arrays are uint8, and the median filter can also
        # ADD failing dies (it fills holes), so "before - after" could go negative
        # and wrap around to a huge number. Count only dies that were switched off.
        n_fail = int(fail.sum())
        removed = int((fail & (clean == 0)).sum())
        extras.append([n_fail / max(1, int(mask.sum())), removed / max(1, n_fail)])
        fails.append(resize(clean, size))
        masks.append(resize(mask, size))
    return (np.stack(fails).astype(np.float32), np.stack(masks).astype(bool),
            np.asarray(extras, dtype=np.float32))

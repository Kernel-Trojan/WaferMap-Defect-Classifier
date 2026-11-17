"""Steps 3-6: a one-layer "CNN" where the filters are learned without backprop.

fit()       - grab 5x5 patches from TRAINING wafers, normalise them, and learn
              a filter bank from them (k-means, PCA or random).
transform() - slide those filters over every 5x5 spot on each wafer, apply a
              non-linearity, then average the responses inside concentric rings.

Everything learned here (patch stats, ZCA, filters) is fitted only on the wafers
passed to fit(). Call fit() on training folds only, or the test set leaks in.

Based on Coates, Ng and Lee, "An Analysis of Single-Layer Networks in Unsupervised
Feature Learning", AISTATS 2011 (https://proceedings.mlr.press/v15/coates11a.html):
ZCA-whitened patches, k-means filters, triangle activation, region pooling. Our
changes: rings instead of quadrants, and each filter tried at 4 rotations.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

P = 5  # patch size


def all_patches(fails: np.ndarray) -> np.ndarray:
    """Every 5x5 patch of every wafer, one per pixel (zero padding at the edges).

    Shape (n_wafers, H, W, 25). This is "im2col": once patches are rows,
    a convolution is just a matrix multiply.
    """
    padded = np.pad(fails, ((0, 0), (2, 2), (2, 2)))
    windows = sliding_window_view(padded, (P, P), axis=(1, 2))
    return windows.reshape(*fails.shape, P * P)


def normalise(patches: np.ndarray) -> np.ndarray:
    """Per-patch contrast normalisation: subtract the patch mean, divide by its spread.

    This makes a filter care about the *shape* inside a patch, not how many dies
    fail. The +0.01 stops all-pass or all-fail patches from blowing up to infinity.
    """
    patches = patches - patches.mean(-1, keepdims=True)
    return patches / np.sqrt(patches.var(-1, keepdims=True) + 0.01)


def rotations(filters: np.ndarray) -> list[np.ndarray]:
    """The filter bank turned by 0, 90, 180 and 270 degrees."""
    square = filters.reshape(-1, P, P)
    return [np.rot90(square, k, axes=(1, 2)).reshape(-1, P * P) for k in range(4)]


class ConvFeatures:
    def __init__(self, method="kmeans", k=32, rotate=True, pooling="rings",
                 n_patches=200_000, alpha=0.5, seed=42):
        self.method, self.k, self.rotate, self.pooling = method, k, rotate, pooling
        self.n_patches, self.alpha, self.seed = n_patches, alpha, seed

    # ---------- learning the filters ----------
    def fit(self, fails, masks, labels):
        rng = np.random.default_rng(self.seed)
        # Class-balanced sampling: pick wafers with weight 1/class size, so the
        # 149 Near-full wafers give as many patches as the 10,000 "none" ones.
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
        wafer = rng.choice(len(fails), size=self.n_patches, p=weights / weights.sum())
        # For each chosen wafer, a random on-wafer centre pixel.
        ys, xs = np.empty(self.n_patches, int), np.empty(self.n_patches, int)
        for i, w in enumerate(wafer):
            on = np.flatnonzero(masks[w])
            ys[i], xs[i] = divmod(rng.choice(on), masks.shape[2])
        padded = np.pad(fails, ((0, 0), (2, 2), (2, 2)))
        patches = np.stack([padded[w, y:y + P, x:x + P].ravel() for w, y, x in zip(wafer, ys, xs)])
        # Flat patches (all pass) carry no shape, and would just become one
        # wasted "blank" filter, so drop them.
        patches = normalise(patches[patches.std(1) > 0])

        if self.method == "kmeans":
            # ZCA whitening: undo the fact that neighbouring dies are strongly
            # correlated, so k-means finds edges/lines instead of blurry blobs.
            self.mean = patches.mean(0)
            values, vectors = np.linalg.eigh(np.cov(patches - self.mean, rowvar=False))
            self.zca = vectors @ np.diag(1 / np.sqrt(values + 0.1)) @ vectors.T
            white = (patches - self.mean) @ self.zca
            self.filters = KMeans(self.k, algorithm="lloyd", n_init=3,
                                  random_state=self.seed).fit(white).cluster_centers_
        elif self.method == "pca":
            # At most 25 components exist for 25-pixel patches.
            self.filters = PCA(min(self.k, P * P), random_state=self.seed).fit(patches).components_
        elif self.method == "random":
            # Control: if random filters do as well, the learning isn't helping.
            random = rng.standard_normal((self.k, P * P))
            self.filters = random / np.linalg.norm(random, axis=1, keepdims=True)
        else:
            raise ValueError(f"unknown method {self.method}")
        return self

    # ---------- using the filters ----------
    def activations(self, patches):
        """Filter responses at every pixel: (..., channels).

        With rotate=True each filter is tried at 4 orientations and we keep the
        strongest, so a vertical-line filter also fires on a horizontal scratch.
        """
        banks = rotations(self.filters) if self.rotate else [self.filters]
        if self.method == "kmeans":
            x = (patches - self.mean) @ self.zca
            # Triangle activation: distance to each centroid, keep only the ones
            # closer than average. Gives k channels, mostly zeros (sparse).
            out = None
            for c in banks:
                d = np.sqrt(np.maximum((x ** 2).sum(-1, keepdims=True) - 2 * x @ c.T + (c ** 2).sum(1), 0))
                a = np.maximum(0, d.mean(-1, keepdims=True) - d)
                out = a if out is None else np.maximum(out, a)  # max over 4 orientations
            return out
        # PCA / random: two-sided thresholded ReLU, one channel for a "match" and
        # one for an "inverted match", so 2k channels.
        pos = neg = None
        for f in banks:
            r = patches @ f.T
            p, n = np.maximum(0, r - self.alpha), np.maximum(0, -r - self.alpha)
            pos = p if pos is None else np.maximum(pos, p)
            neg = n if neg is None else np.maximum(neg, n)
        return np.concatenate([pos, neg], -1)

    def regions(self, size):
        """Boolean masks for the pooling regions on a size x size grid.

        rings:    4 rings at 0-25-50-75-100% of the radius, plus the whole wafer.
        quadrant: 4 quadrants plus the whole wafer (ablation).
        """
        y, x = np.mgrid[:size, :size] - (size - 1) / 2
        if self.pooling == "rings":
            r = np.sqrt(x ** 2 + y ** 2) / (size / 2)
            parts = [(r >= lo) & (r < hi) for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 9)]]
        else:
            parts = [(y < 0) & (x < 0), (y < 0) & (x >= 0), (y >= 0) & (x < 0), (y >= 0) & (x >= 0)]
        return np.stack(parts + [np.ones((size, size), bool)])

    def transform(self, fails, masks, batch=500):
        regions = self.regions(fails.shape[1])
        out = []
        for i in range(0, len(fails), batch):
            act = self.activations(normalise(all_patches(fails[i:i + batch])))  # (b, H, W, C)
            # Only count on-wafer pixels in each region (mask-aware mean), so the
            # empty corners of the square grid don't dilute the averages.
            weight = regions[None] & masks[i:i + batch, None]                 # (b, R, H, W)
            pooled = np.einsum("brhw,bhwc->brc", weight.astype(np.float32), act)
            pooled /= np.maximum(weight.sum((2, 3)), 1)[..., None]
            out.append(pooled.reshape(len(pooled), -1))
        return np.concatenate(out)

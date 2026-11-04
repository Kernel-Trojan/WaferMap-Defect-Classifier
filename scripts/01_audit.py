from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import first_label, load_data

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
NONE_SAMPLE_SIZE = 10000
EXPECTED_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def map_shape(value):
    try:
        array = np.asarray(value)
        return tuple(array.shape) if array.ndim == 2 else None
    except (TypeError, ValueError):
        return None


def map_hash(value):
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.int16))
        return hashlib.sha1(repr(array.shape).encode("ascii") + array.tobytes()).hexdigest()
    except (TypeError, ValueError):
        return None


def main():
    frame = load_data()
    required = {"failureType", "waferMap", "dieSize", "lotName"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")

    labels = frame["failureType"].map(first_label)
    print(f"Rows: {len(frame):,}; labeled: {labels.notna().sum():,}; "
          f"defect: {(labels.notna() & labels.ne('none')).sum():,}; "
          f"none: {(labels == 'none').sum():,}")
    expected = (811457, 172950, 25519, 147431)
    observed = (len(frame), int(labels.notna().sum()),
                int((labels.notna() & labels.ne("none")).sum()), int((labels == "none").sum()))
    if observed != expected:
        raise ValueError(f"Source row totals differ from WM-811K expectations: {observed} != {expected}")
    class_counts = labels.value_counts(dropna=False).to_dict()

    defect = frame.loc[labels.notna() & labels.ne("none")].copy()
    none = frame.loc[labels.eq("none")].sample(n=NONE_SAMPLE_SIZE, random_state=SEED)
    subset = pd.concat([defect, none]).sort_index().copy()
    subset.insert(0, "source_index", subset.index.astype("int64"))
    subset.insert(1, "label", labels.loc[subset.index].values)

    ROOT.joinpath("data").mkdir(exist_ok=True)
    subset.to_pickle(ROOT / "data" / "project_subset.pkl")
    subset["label"].value_counts().sort_index().to_csv(ROOT / "experiments" / "class_counts.csv", header=["count"])

    lots = frame["lotName"]
    lot_counts = frame.assign(audit_label=labels).groupby("audit_label")["lotName"].nunique().to_dict()
    missing_lots = int(lots.isna().sum())
    sizes = lots.value_counts(dropna=False)
    wafer_shapes = frame["waferMap"].map(map_shape)
    exact_square = sum(shape is not None and shape[0] == shape[1] for shape in wafer_shapes)
    dims_match = 0
    for die_size, wafer_map in zip(frame["dieSize"], frame["waferMap"]):
        try:
            dims_match += int(int(die_size) == int((np.asarray(wafer_map) > 0).sum()))
        except (TypeError, ValueError):
            pass

    hashes = frame["waferMap"].map(map_hash)
    duplicate_mask = hashes.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    hash_to_lots = defaultdict(set)
    for lot, digest in zip(frame["lotName"], hashes):
        if digest:
            hash_to_lots[digest].add(str(lot))
    duplicate_cross_lot = {digest: sorted(values) for digest, values in hash_to_lots.items() if len(values) > 1}

    report = {
        "seed": SEED,
        "source_rows": int(len(frame)),
        "labeled_rows": int(labels.notna().sum()),
        "defect_rows": int((labels.notna() & labels.ne("none")).sum()),
        "none_rows": int((labels == "none").sum()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "subset_rows": int(len(subset)),
        "subset_class_counts": {str(k): int(v) for k, v in subset["label"].value_counts().items()},
        "unique_lots_by_class": {str(k): int(v) for k, v in lot_counts.items()},
        "missing_lot_rows": missing_lots,
        "unique_lots": int(lots.nunique()),
        "lot_size_distribution": {str(k): int(v) for k, v in Counter(sizes).items()},
        "wafer_map_shapes_source": {str(k): int(v) for k, v in wafer_shapes.value_counts(dropna=False).items()},
        "exact_square_grids": int(exact_square),
        "die_size_matches_positive_map_cells": int(dims_match),
        "duplicate_grid_rows": duplicate_rows,
        "duplicate_hash_count": int(hashes.nunique()),
        "duplicate_hashes_spanning_multiple_lots": len(duplicate_cross_lot),
        "duplicate_hash_to_lots": duplicate_cross_lot,
        "output_subset": "data/project_subset.pkl",
    }
    ROOT.joinpath("experiments").mkdir(exist_ok=True)
    (ROOT / "experiments" / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    rng = np.random.default_rng(SEED)
    for ax, label in zip(axes.flat, EXPECTED_CLASSES):
        candidates = subset.loc[subset["label"] == label]
        ax.axis("off")
        ax.set_title(label.strip() or "(blank)")
        if candidates.empty:
            ax.text(.5, .5, "no sample", ha="center", va="center")
            continue
        row = candidates.iloc[int(rng.integers(len(candidates)))]
        image = np.asarray(row["waferMap"])
        ax.imshow(image, cmap="gray_r", interpolation="nearest")
    fig.tight_layout()
    figure_path = ROOT / "experiments" / "figures" / "01_sample_wafers_per_class.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Subset: {len(subset):,} wafers → data/project_subset.pkl")
    print(f"Audit summary: experiments/audit_summary.json")
    print(f"Sample figure: {figure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

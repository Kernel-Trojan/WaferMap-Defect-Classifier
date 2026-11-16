"""Phase 3 ablations: change ONE thing at a time from the default k-means setup
and see what it does to validation macro-F1 (logistic regression only).

Default: k-means K=32, rotated filters, ring pooling, 32x32, isolated-die denoising.
Still training folds only; the test split is never loaded.

Run: python main.py tuning   (needs the P3-kmeans-logreg row from `python main.py pipeline`)
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pipeline = importlib.import_module("scripts.03_pipeline")
ROOT = Path(__file__).resolve().parents[1]

ABLATIONS = {
    "k16": {"k": 16},
    "k64": {"k": 64},
    "denoise-median": {"denoise": "median"},
    "denoise-none": {"denoise": "none"},
    "no-rotation": {"rotate": False},
    "quadrant-pooling": {"pooling": "quadrant"},
    "size64": {"size": 64},
}


def main(only=None):
    """Run every ablation, or just the names in `only` (e.g. to redo one)."""
    data = pipeline.load_train()
    rows = []
    for name, change in ABLATIONS.items():
        if only and name not in only:
            continue
        config = {**pipeline.DEFAULT, **change}
        print(f"{name}:", flush=True)
        pred, calib = pipeline.run_cv(config, ["logreg"], data)["logreg"]
        rows.append(pipeline.result_row(f"P3-ablation-{name}", 3, config, "logreg", pred, data[1], data[2], calib,
                                        f" | ablation: {name}"))
        print(f"  macro-F1 {rows[-1]['macro_f1']:.4f}", flush=True)
        pipeline.write_log(rows)

    # Bar chart: default vs each ablation, all numbers read back from the log.
    log = {r["experiment_id"]: float(r["macro_f1"]) for r in csv.DictReader((ROOT / "experiments" / "log.csv").open())}
    names = ["default"] + list(ABLATIONS)
    scores = [log["P3-kmeans-logreg"]] + [log[f"P3-ablation-{n}"] for n in ABLATIONS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], scores[::-1], color=["#888"] * len(ABLATIONS) + ["#2a6"])
    for y, s in enumerate(scores[::-1]):
        ax.text(s, y, f" {s:.3f}", va="center")
    ax.set_xlim(min(scores) - 0.05, max(scores) + 0.03)
    ax.set_xlabel("validation macro-F1 (5-fold, lot-grouped)")
    ax.set_title("One change at a time vs default (k-means K=32, logistic regression)")
    fig.tight_layout()
    fig.savefig(ROOT / "experiments" / "figures" / "04_ablations_summary.png", dpi=160)


if __name__ == "__main__":
    # python -m scripts.04_tuning                -> all ablations
    # python -m scripts.04_tuning denoise-median -> just that one
    main(sys.argv[1:])

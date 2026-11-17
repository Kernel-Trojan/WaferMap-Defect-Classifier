"""Do the learned k-means filters actually beat random ones? A paired check on validation.

In Phase 3, random filters + logistic regression scored 0.770 against 0.754 for
k-means. Here we score both on the same folds, then redraw the same training lots
1,000 times, so we can see whether that gap is real or just noise.
Validation folds only; the test set is never loaded.

Run: python main.py filter-check
"""
import json
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation import lot_bootstrap

pipeline = import_module("scripts.03_pipeline")
ROOT, CLASSES = pipeline.ROOT, pipeline.CLASSES


def main():
    data = pipeline.load_train()
    labels, folds = data[1], data[2]
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    lots = split.loc[split["split"].eq("train"), "lotName"].to_numpy()

    predictions = {}
    for method in ("kmeans", "random"):
        print(f"{method} filters:", flush=True)
        config = {**pipeline.DEFAULT, "method": method}
        predictions[method] = pipeline.run_cv(config, ["logreg"], data)["logreg"][0]

    def per_class(method, rows):
        return f1_score(labels[rows], predictions[method][rows], average=None, labels=CLASSES, zero_division=0)

    # Same seed = same lots redrawn for both, so we compare them draw by draw.
    kmeans = lot_bootstrap(lots, lambda rows: per_class("kmeans", rows))
    random = lot_bootstrap(lots, lambda rows: per_class("random", rows))
    gap = random.mean(axis=1) - kmeans.mean(axis=1)
    everything = np.arange(len(labels))
    result = {
        "macro_f1": {m: float(per_class(m, everything).mean()) for m in predictions},
        "macro_f1_by_fold": {m: [float(per_class(m, np.flatnonzero(folds == f)).mean()) for f in range(5)]
                             for m in predictions},
        "random_minus_kmeans": [float(gap.mean()), *np.percentile(gap, [2.5, 97.5])],
        "random_beats_kmeans_in_share_of_draws": float((gap > 0).mean()),
        "per_class_f1": {c: {m: float(per_class(m, everything)[i]) for m in predictions} for i, c in enumerate(CLASSES)},
    }
    (ROOT / "experiments" / "filter_check.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"k-means {result['macro_f1']['kmeans']:.4f}   random {result['macro_f1']['random']:.4f}")
    print("random minus k-means: {:+.4f} (95% lot bootstrap {:+.4f} to {:+.4f})".format(*result["random_minus_kmeans"]))
    print(f"random wins in {result['random_beats_kmeans_in_share_of_draws']:.0%} of redraws")
    for f in range(5):
        print(f"  fold {f}: k-means {result['macro_f1_by_fold']['kmeans'][f]:.4f}  "
              f"random {result['macro_f1_by_fold']['random'][f]:.4f}")


if __name__ == "__main__":
    main()

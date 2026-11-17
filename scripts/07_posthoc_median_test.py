"""Post-hoc check, asked for AFTER the final test: isolated vs median denoising on the test set.

The primary model (isolated denoise) stays frozen from 05_final_test.py; its test
predictions are read back, not refit. Only the median version is newly fit, on all
training wafers. This is a second look at the test set, so it is logged as post-hoc
and must not be used to swap the primary model.

Run: python main.py posthoc-median
"""
import json
from datetime import date
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.evaluation import lot_bootstrap
from src.features import ConvFeatures
from src.models import make_model
from src.preprocessing import prepare

final = import_module("scripts.05_final_test")
pipeline = import_module("scripts.03_pipeline")
ROOT, CLASSES, SEED = final.ROOT, final.CLASSES, final.SEED


def macro_f1(labels, predicted):
    return f1_score(labels, predicted, average="macro", labels=CLASSES, zero_division=0)


def main():
    config = json.loads((ROOT / "experiments" / "final_config.json").read_text())
    recipe = {**config["recipe"], "denoise": "median"}
    subset = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    train = split["split"].eq("train").to_numpy()
    test = ~train
    labels = split["label"].to_numpy()

    # Primary: the frozen predictions from the one-time final test.
    saved = pd.read_csv(ROOT / "experiments" / "final_test_predictions.csv")
    if not (saved["source_index"].to_numpy() == split.loc[test, "source_index"].to_numpy()).all():
        raise ValueError("Saved test predictions don't line up with the split")
    primary = saved["final-logreg"].to_numpy()

    # Median version: same recipe, only the denoising changes. Fit on training wafers only.
    print("Fitting k-means K=32 + median denoise + logistic regression", flush=True)
    fails, masks, extras = prepare(subset["waferMap"], recipe["size"], recipe["denoise"])
    conv = ConvFeatures(recipe["method"], recipe["k"], recipe["rotate"], recipe["pooling"], seed=SEED)
    conv.fit(fails[train], masks[train], labels[train])
    x = np.hstack([conv.transform(fails, masks), extras])
    median = make_model("logreg").fit(x[train], labels[train]).predict(x[test])

    y, lots = labels[test], split.loc[test, "lotName"].to_numpy()
    table = {}
    for name, predicted in (("isolated (primary)", primary), ("median (post-hoc)", median)):
        p, r, f, support = precision_recall_fscore_support(y, predicted, labels=CLASSES, zero_division=0)
        table[name] = pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=CLASSES)
    table = pd.concat(table, axis=1)
    table.loc["macro avg"] = table.mean()
    table.insert(0, "test wafers", list(support) + [support.sum()])

    # Same seed = same redrawn lots for both, so the gap is compared draw by draw.
    gap = (lot_bootstrap(lots, lambda rows: macro_f1(y[rows], median[rows]))
           - lot_bootstrap(lots, lambda rows: macro_f1(y[rows], primary[rows])))
    low, high = np.percentile(gap, [2.5, 97.5])

    table.round(4).to_csv(ROOT / "experiments" / "test_denoise_comparison.csv")
    summary = {"median_minus_isolated_macro_f1": [macro_f1(y, median) - macro_f1(y, primary), low, high],
               "note": "Post-hoc second look at the test set, asked for after the final test. Primary stays isolated."}
    (ROOT / "experiments" / "test_denoise_comparison.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    pipeline.write_log([{
        "experiment_id": "P4-test-posthoc-median-logreg", "date": date.today().isoformat(), "phase": 4,
        "description": "Post-hoc test check (asked after the final test): median instead of isolated denoise",
        "split": "locked_test_20pct_lots", "seed": SEED, "model": "logreg", "parameters": json.dumps(recipe),
        "macro_f1": macro_f1(y, median), "macro_f1_fold_std": "",
        "weighted_f1": f1_score(y, median, average="weighted", labels=CLASSES, zero_division=0),
        "accuracy": float(np.mean(median == y)),
        "notes": f"median minus isolated macro-F1 95% lot bootstrap {low:+.4f} to {high:+.4f}; not used to pick a model"}])

    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    print(f"\nMedian minus isolated, macro-F1: {summary['median_minus_isolated_macro_f1'][0]:+.4f} "
          f"(95% lot bootstrap {low:+.4f} to {high:+.4f})")


if __name__ == "__main__":
    main()

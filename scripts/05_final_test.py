"""Phase 4, part 1: the ONE run on the locked test set.

The recipe comes from experiments/final_config.json, which was committed before
this script ever ran. Every learned piece (filters, ZCA, scaler, classifier) is fit
on all training wafers, then scored once on the test wafers. Error bars come from
redrawing whole test lots 1,000 times.

This script refuses to run a second time. Delete experiments/final_test.json only
if you really mean to re-test (and say so in the report).

Run: python main.py final-test
"""
import json
from datetime import date
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.evaluation import expected_calibration_error, lot_bootstrap
from src.features import ConvFeatures
from src.models import make_model
from src.preprocessing import prepare

baselines = import_module("scripts.03_baselines")
pipeline = import_module("scripts.03_pipeline")
ROOT, CLASSES, SEED = baselines.ROOT, np.array(baselines.CLASSES), baselines.SEED
RESULTS = ROOT / "experiments" / "final_test.json"
MODELS = ROOT / "data" / "final_models.joblib"  # gitignored; 06_analysis reuses it


def scores(labels, predicted, probs=None):
    """9 per-class F1s, accuracy, and if we have probabilities: ECE + 9 per-class PR-AUCs."""
    out = list(f1_score(labels, predicted, average=None, labels=CLASSES, zero_division=0))
    out.append(np.mean(labels == predicted))
    if probs is not None:
        out.append(expected_calibration_error(probs, labels, CLASSES))
        out += [average_precision_score(labels == c, probs[:, i]) for i, c in enumerate(CLASSES)]
    return np.array(out)


def summarise(point, draws):
    """Turn the scores + 1,000 bootstrap draws into [value, low, high] entries."""
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    macro = draws[:, :9].mean(axis=1)
    out = {"macro_f1": [point[:9].mean(), *np.percentile(macro, [2.5, 97.5])],
           "accuracy": [point[9], low[9], high[9]],
           "per_class_f1": {c: [point[i], low[i], high[i]] for i, c in enumerate(CLASSES)}}
    if len(point) > 10:
        out["ece"] = [point[10], low[10], high[10]]
        out["pr_auc"] = {c: [point[11 + i], low[11 + i], high[11 + i]] for i, c in enumerate(CLASSES)}
    return out


def main():
    if RESULTS.exists():
        raise SystemExit("The final test already ran (experiments/final_test.json). It only runs once.")
    config = json.loads((ROOT / "experiments" / "final_config.json").read_text())
    recipe = config["recipe"]

    subset = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    if not (subset["source_index"].to_numpy() == split["source_index"].to_numpy()).all():
        raise ValueError("Subset and split file are out of sync")
    train = split["split"].eq("train").to_numpy()
    test = ~train
    labels = split["label"].to_numpy()
    print(f"{train.sum():,} training wafers, {test.sum():,} test wafers", flush=True)

    # Our pipeline: everything learned from training wafers only.
    fails, masks, extras = prepare(subset["waferMap"], recipe["size"], recipe["denoise"])
    conv = ConvFeatures(recipe["method"], recipe["k"], recipe["rotate"], recipe["pooling"], seed=SEED)
    conv.fit(fails[train], masks[train], labels[train])
    features = np.hstack([conv.transform(fails, masks), extras])
    # B2 baselines: the same 1024 flat fail pixels as Phase 2.
    pixels = np.stack([baselines.resize_map(m) == 2 for m in subset["waferMap"]]).reshape(len(subset), -1)
    pixels = pixels.astype(np.float32)

    inputs = {"final-logreg": features, "final-svm": features,
              "B2-flat-logistic": pixels, "B2-flat-random-forest": pixels}
    makers = {"final-logreg": lambda: make_model("logreg"), "final-svm": lambda: make_model("svm"),
              "B2-flat-logistic": baselines.logistic_model, "B2-flat-random-forest": baselines.random_forest}
    models = {}
    for name in config["models"]:
        print(f"Fitting {name}", flush=True)
        models[name] = makers[name]().fit(inputs[name][train], labels[train])
    joblib.dump({"conv": conv, "models": models}, MODELS)

    # The one look at the test set.
    y, lots = labels[test], split.loc[test, "lotName"].to_numpy()
    results, draws, rows = {}, {}, []
    predictions = split.loc[test, ["source_index", "lotName", "label"]].copy()
    for name, model in models.items():
        x = inputs[name][test]
        predicted = model.predict(x)
        probs = model.predict_proba(x) if hasattr(model, "predict_proba") else None
        predictions[name] = predicted
        # Same seed for every model, so draw i uses the same lots for all of them.
        draws[name] = lot_bootstrap(lots, lambda r: scores(y[r], predicted[r], None if probs is None else probs[r]),
                                    config["bootstrap"]["rounds"], config["bootstrap"]["seed"])
        results[name] = summarise(scores(y, predicted, probs), draws[name])
        macro, low, high = results[name]["macro_f1"]
        print(f"{name:<24} macro-F1 {macro:.4f} ({low:.4f} to {high:.4f})", flush=True)
        rows.append({"experiment_id": f"P4-test-{name}", "date": date.today().isoformat(), "phase": 4,
                     "description": "Final locked test (run once)", "split": "locked_test_20pct_lots", "seed": SEED,
                     "model": name, "parameters": json.dumps(recipe) if name.startswith("final") else "same as Phase 2",
                     "macro_f1": macro, "macro_f1_fold_std": "",
                     "weighted_f1": f1_score(y, predicted, average="weighted", labels=CLASSES, zero_division=0),
                     "accuracy": results[name]["accuracy"][0],
                     "notes": f"95% lot bootstrap macro-F1 {low:.4f} to {high:.4f}"})

    # How much better is our main model than each other model, on the same redrawn lots?
    main_macro = draws["final-logreg"][:, :9].mean(axis=1)
    results["final_logreg_minus"] = {
        name: [results["final-logreg"]["macro_f1"][0] - results[name]["macro_f1"][0],
               *np.percentile(main_macro - draws[name][:, :9].mean(axis=1), [2.5, 97.5])]
        for name in models if name != "final-logreg"}
    for name, (gap, low, high) in results["final_logreg_minus"].items():
        print(f"final-logreg minus {name:<22} {gap:+.4f} ({low:+.4f} to {high:+.4f})")

    results["test_wafers"], results["test_lots"] = int(test.sum()), int(len(set(lots)))
    RESULTS.write_text(json.dumps(results, indent=2, default=float) + "\n")
    predictions.to_csv(ROOT / "experiments" / "final_test_predictions.csv", index=False)
    pipeline.write_log(rows)
    print("Saved experiments/final_test.json, final_test_predictions.csv and log.csv rows")


if __name__ == "__main__":
    main()

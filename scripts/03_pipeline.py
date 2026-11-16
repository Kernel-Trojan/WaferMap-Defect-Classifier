"""Phase 3 main run: 3 filter banks x 4 classifiers, 5-fold lot-grouped CV.

Only training-split wafers are used. The test split is not even loaded.
In every fold the filters/ZCA/scaler are fitted on the 4 training folds and
scored on the 5th, so validation wafers never shape the features.

Run: python main.py pipeline   (results go to experiments/log.csv)
"""
from __future__ import annotations

import csv
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features import ConvFeatures
from src.models import make_model
from src.preprocessing import prepare

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
DEFAULT = dict(method="kmeans", k=32, rotate=True, pooling="rings", size=32, denoise="isolated")
PREPARED = {}  # (size, denoise) -> prepared maps, reused across configs
LOG_FIELDS = ["experiment_id", "date", "phase", "description", "split", "seed", "model",
              "parameters", "macro_f1", "macro_f1_fold_std", "weighted_f1", "accuracy", "notes"]


def load_train():
    """Training-split wafers, their labels, and their fold numbers (0-4)."""
    subset = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    if not (subset["source_index"].to_numpy() == split["source_index"].to_numpy()).all():
        raise ValueError("Subset and split file are out of sync; rerun audit")
    train = (split["split"] == "train").to_numpy()
    return (subset.loc[train, "waferMap"].tolist(), split.loc[train, "label"].to_numpy(),
            split.loc[train, "fold"].to_numpy())


def ece(proba, labels, classes, bins=15):
    """Expected calibration error: how far confidence is from actual accuracy."""
    confidence = proba.max(1)
    correct = classes[proba.argmax(1)] == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return total


def run_cv(config, model_names, data):
    """5-fold CV for one feature config and several classifiers.

    Returns {model name: (predictions, ece or None)} over all training wafers.
    """
    maps, labels, folds = data
    key = (config["size"], config["denoise"])
    if key not in PREPARED:  # preprocessing has nothing learned, do it once
        PREPARED[key] = prepare(maps, config["size"], config["denoise"])
    fails, masks, extras = PREPARED[key]

    preds = {m: np.empty(len(labels), dtype=object) for m in model_names}
    probas = np.zeros((len(labels), len(CLASSES)))
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        conv = ConvFeatures(config["method"], config["k"], config["rotate"], config["pooling"], seed=SEED)
        conv.fit(fails[tr], masks[tr], labels[tr])
        X = np.hstack([conv.transform(fails, masks), extras])
        for m in model_names:
            model = make_model(m).fit(X[tr], labels[tr])
            preds[m][va] = model.predict(X[va])
            if m == "logreg":
                probas[va] = model.predict_proba(X[va])
        print(f"  fold {fold} done ({X.shape[1]} features)", flush=True)
    return {m: (preds[m], ece(probas, labels, np.array(CLASSES)) if m == "logreg" else None)
            for m in model_names}


def result_row(exp_id, phase, config, model, pred, labels, folds, calib, extra_note=""):
    note = f"ECE={calib:.4f}; " if calib is not None else ""
    per_class = f1_score(labels, pred, labels=CLASSES, average=None, zero_division=0)
    note += "per-class F1 " + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASSES, per_class))
    return {"experiment_id": exp_id, "date": date.today().isoformat(), "phase": phase,
            "description": f"Conv features ({config['method']}) + {model}" + extra_note,
            "split": "lot_grouped_5fold", "seed": SEED, "model": model,
            "parameters": ";".join(f"{k}={v}" for k, v in config.items()),
            "macro_f1": f1_score(labels, pred, average="macro", labels=CLASSES, zero_division=0),
            # How much macro-F1 jumps around between folds: a rough noise level.
            "macro_f1_fold_std": float(np.std([f1_score(labels[folds == f], pred[folds == f], average="macro",
                                                        labels=CLASSES, zero_division=0) for f in range(5)])),
            "weighted_f1": f1_score(labels, pred, average="weighted", labels=CLASSES, zero_division=0),
            "accuracy": float(np.mean(pred == labels)), "notes": note}


def write_log(rows):
    """Append rows to experiments/log.csv, replacing older rows with the same id."""
    path = ROOT / "experiments" / "log.csv"
    existing = list(csv.DictReader(path.open())) if path.exists() else []
    ids = {r["experiment_id"] for r in rows}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(r for r in existing if r["experiment_id"] not in ids)
        writer.writerows(rows)


def main():
    data = load_train()
    labels = data[1]
    rows = []
    for method in ["kmeans", "pca", "random"]:
        config = {**DEFAULT, "method": method}
        start = time.time()
        print(f"{method}:", flush=True)
        results = run_cv(config, ["logreg", "ridge", "svm", "rf"], data)
        for model, (pred, calib) in results.items():
            rows.append(result_row(f"P3-{method}-{model}", 3, config, model, pred, labels, data[2], calib))
            print(f"  {model}: macro-F1 {rows[-1]['macro_f1']:.4f}", flush=True)
        print(f"  ({time.time() - start:.0f}s)", flush=True)
        write_log(rows)  # save as we go, so a crash doesn't lose finished runs
    (ROOT / "experiments" / "phase3_summary.json").write_text(json.dumps(
        [{k: r[k] for k in ("experiment_id", "macro_f1", "weighted_f1", "accuracy", "notes")} for r in rows],
        indent=2) + "\n")


if __name__ == "__main__":
    main()

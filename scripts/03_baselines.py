"""Phase 2 baselines: the bars our CNN-style pipeline has to clear.

All scores are 5-fold, lot-grouped cross-validation on the TRAINING wafers only.
The locked test set is never loaded here. Each fold's model is fit on the other
four folds; we pool every wafer's out-of-fold prediction and score them together.

    B0  majority class / uniform random  -> the "no skill" floor
    B1  overall failing-die fraction     -> how far one number gets you
    B2  flattened 32x32 pixels           -> logistic regression and random forest,
                                            plus a naive random split to measure lot leakage
    B3  B2 logistic + 4 rotated copies   -> does teaching rotation help raw pixels?

Re-running replaces every phase-2 row in experiments/log.csv, so the log only
ever holds the current numbers. Older versions live in git history.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PHASE = "2"
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
LOG_FIELDS = ["experiment_id", "date", "phase", "description", "split", "seed", "model", "parameters",
              "macro_f1", "macro_f1_fold_std", "weighted_f1", "accuracy", "notes"]
LOT_SPLIT, RANDOM_SPLIT = "lot_grouped_5fold", "naive_random_5fold"


def resize_map(value: object, size: int = 32) -> np.ndarray:
    """Shrink or stretch a raw 0/1/2 wafer map to size x size by picking the nearest cell."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError("Every waferMap must be a non-empty 2D array")
    rows = np.floor(np.linspace(0, array.shape[0] - 1, size) + 0.5).astype(int)
    cols = np.floor(np.linspace(0, array.shape[1] - 1, size) + 0.5).astype(int)
    return array[rows[:, None], cols]


def logistic_model():
    """Standardise each feature, then class-balanced multinomial logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))


def random_forest():
    return RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=2, random_state=SEED,
                                  min_samples_leaf=2)


def record(experiment_id: str, model: str, parameters: str, labels: np.ndarray, predictions: np.ndarray,
           folds: np.ndarray, notes: str, split: str = LOT_SPLIT) -> dict[str, object]:
    """Score pooled out-of-fold predictions and build one log row.

    macro_f1_fold_std is how much macro-F1 moves from fold to fold. It's a rough
    sense of the noise: a gap between two models smaller than that isn't worth much.
    """
    per_fold = [f1_score(labels[folds == f], predictions[folds == f], average="macro", labels=CLASSES,
                         zero_division=0) for f in range(5)]
    return {"experiment_id": experiment_id, "phase": PHASE, "description": "Phase 2 baseline", "split": split,
            "seed": SEED, "model": model, "parameters": parameters,
            "macro_f1": f1_score(labels, predictions, average="macro", labels=CLASSES, zero_division=0),
            "macro_f1_fold_std": float(np.std(per_fold)),
            "weighted_f1": f1_score(labels, predictions, average="weighted", labels=CLASSES, zero_division=0),
            "accuracy": float(np.mean(labels == predictions)), "notes": notes}


def cross_val_predict(make_estimator, features: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Fit a fresh model on 4 folds, predict the 5th, repeat. Returns one prediction per wafer."""
    predictions = np.empty(labels.shape, dtype=object)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        predictions[valid] = make_estimator().fit(features[train], labels[train]).predict(features[valid])
    return predictions


def rotation_augmented_predict(fail: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """B3: same model as B2 logistic, but each training wafer also appears turned 90, 180 and 270 degrees.

    Only the training folds get rotated copies. Validation wafers are scored as
    they are, in the exact same 0/1 pixel format the model trained on.
    """
    predictions = np.empty(labels.shape, dtype=object)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        rotated = np.concatenate([np.rot90(fail[train], k, axes=(1, 2)) for k in range(4)])
        model = logistic_model().fit(rotated.reshape(len(rotated), -1).astype(np.float32), np.tile(labels[train], 4))
        predictions[valid] = model.predict(fail[valid].reshape(int(valid.sum()), -1).astype(np.float32))
    return predictions


def write_outputs(results: list[dict[str, object]]) -> None:
    """Replace phase-2 rows in the log (keep other phases), and refresh the summary JSON."""
    log_path = ROOT / "experiments" / "log.csv"
    others = [row for row in csv.DictReader(log_path.open())] if log_path.exists() else []
    others = [row for row in others if row.get("phase") != PHASE]
    with log_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(others)
        writer.writerows({"date": date.today().isoformat(), **row} for row in results)
    keep = ("experiment_id", "split", "macro_f1", "macro_f1_fold_std", "weighted_f1", "accuracy")
    summary = [{k: row[k] for k in keep} for row in results]
    (ROOT / "experiments" / "phase2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    subset_path = ROOT / "data" / "project_subset.pkl"
    split_path = ROOT / "splits" / "project_splits.csv"
    if not subset_path.exists() or not split_path.exists():
        raise FileNotFoundError("Run `python main.py audit` and `python main.py splits` first")
    subset = pd.read_pickle(subset_path).reset_index(drop=True)
    split = pd.read_csv(split_path)
    train_mask = split["split"].eq("train").to_numpy()
    train = subset.loc[train_mask].reset_index(drop=True)
    labels = train["label"].astype(str).to_numpy()
    folds = split.loc[train_mask, "fold"].to_numpy(dtype=int)
    random_folds = split.loc[train_mask, "val_fold_random"].to_numpy(dtype=int)
    if len(train) != len(labels) or (folds < 0).any() or (random_folds < 0).any():
        raise ValueError("Frozen split is malformed")
    results: list[dict[str, object]] = []

    # B0: no-skill floor.
    majority = np.empty(labels.shape, dtype=object)
    for fold in range(5):
        majority[folds == fold] = pd.Series(labels[folds != fold]).value_counts().index[0]
    results.append(record("B0-majority", "majority_class", "most common class in the training folds",
                          labels, majority, folds, "Always predicts the biggest training class"))
    uniform = np.random.default_rng(SEED).choice(CLASSES, size=len(labels))
    results.append(record("B0-uniform", "uniform_random", "uniform over 9 classes", labels, uniform, folds,
                          "One seeded random draw"))

    print(f"Resizing {len(train):,} training wafers to 32x32", flush=True)
    images = np.stack([resize_map(value) for value in train["waferMap"]])
    fail, mask = (images == 2).astype(np.uint8), (images > 0).astype(np.uint8)   # 1 = failing die / 1 = on wafer
    pixels = fail.reshape(len(train), -1).astype(np.float32)

    # B1: one number per wafer, the failing share of the 32x32 grid.
    fractions = (fail.reshape(len(train), -1).sum(axis=1) / np.maximum(1, mask.reshape(len(train), -1).sum(axis=1)))
    fractions = fractions.astype(np.float32)[:, None]
    results.append(record("B1-failing-fraction", "logistic_regression", "1 feature; standardised; balanced",
                          labels, cross_val_predict(logistic_model, fractions, labels, folds), folds,
                          "Overall failing-die fraction on the 32x32 grid"))
    print("Done B0, B1", flush=True)

    # B2: raw pixels, on the honest lot split and the leaky random split.
    flat = "1024 binary fail pixels (32x32, nearest-neighbour)"
    for split_name, split_folds, suffix, note in ((LOT_SPLIT, folds, "", "Flattened 32x32 fail map"),
                                                  (RANDOM_SPLIT, random_folds, "-random-split",
                                                   "Leakage check only: naive random folds ignore lots")):
        results.append(record(f"B2-flat-logistic{suffix}", "logistic_regression", f"{flat}; standardised; balanced",
                              labels, cross_val_predict(logistic_model, pixels, labels, split_folds), split_folds,
                              note, split_name))
        results.append(record(f"B2-flat-random-forest{suffix}", "random_forest",
                              f"{flat}; 100 trees; min_samples_leaf=2; balanced",
                              labels, cross_val_predict(random_forest, pixels, labels, split_folds), split_folds,
                              note, split_name))
    print("Done B2", flush=True)

    # B3: B2 logistic + rotated copies of the training wafers.
    results.append(record("B3-rotation-pixel-logistic", "logistic_regression",
                          f"{flat}; train folds x4 (0/90/180/270 deg); standardised; balanced",
                          labels, rotation_augmented_predict(fail, labels, folds), folds,
                          "Same model as B2-flat-logistic; only the training folds get rotated copies"))
    print("Done B3", flush=True)

    write_outputs(results)
    for row in results:
        print(f"{row['experiment_id']:<36} macro-F1={row['macro_f1']:.4f} (fold std {row['macro_f1_fold_std']:.4f})"
              f"  weighted-F1={row['weighted_f1']:.4f}  accuracy={row['accuracy']:.4f}")
    print("Saved experiments/log.csv and experiments/phase2_summary.json")


if __name__ == "__main__":
    main()

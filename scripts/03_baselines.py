from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def resize_map(value: object, size: int = 32) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError("Every waferMap must be a non-empty 2D array")
    rows = np.floor(np.linspace(0, array.shape[0] - 1, size) + 0.5).astype(int)
    cols = np.floor(np.linspace(0, array.shape[1] - 1, size) + 0.5).astype(int)
    return array[rows[:, None], cols]


def logistic_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def score_cv(name: str, estimator, features: np.ndarray, labels: np.ndarray,
             folds: np.ndarray, notes: str = "", split_name: str = "lot_grouped_5fold") -> dict[str, object]:
    predictions = np.empty(labels.shape, dtype=object)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        estimator.fit(features[train], labels[train])
        predictions[valid] = estimator.predict(features[valid])
    return {"experiment_id": name, "phase": 2, "split": split_name,
            "seed": SEED, "model": name, "parameters": "frozen baseline configuration",
            "macro_f1": f1_score(labels, predictions, average="macro", labels=CLASSES, zero_division=0),
            "weighted_f1": f1_score(labels, predictions, average="weighted", labels=CLASSES, zero_division=0),
            "accuracy": float(np.mean(labels == predictions)), "notes": notes}


def write_summary(results: list[dict[str, object]]) -> None:
    summary = [{k: v for k, v in row.items() if k in ("experiment_id", "macro_f1", "weighted_f1", "accuracy")} for row in results]
    (ROOT / "experiments" / "phase2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def write_log(results: list[dict[str, object]]) -> None:
    log_path = ROOT / "experiments" / "log.csv"
    fields = ["experiment_id", "date", "phase", "description", "split", "seed", "model", "parameters", "macro_f1", "weighted_f1", "accuracy", "notes"]
    existing = list(csv.DictReader(log_path.open())) if log_path.exists() else []
    ids = {str(row["experiment_id"]) for row in results}
    with log_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in existing if row.get("experiment_id") not in ids)
        for row in results:
            writer.writerow({"date": date.today().isoformat(), "description": "Phase 2 baseline", **row})


def main() -> None:
    subset_path = ROOT / "data" / "project_subset.pkl"
    split_path = ROOT / "splits" / "project_splits.csv"
    if not subset_path.exists() or not split_path.exists():
        raise FileNotFoundError("Run scripts/01_audit.py and scripts/02_freeze_splits.py first")
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
    majority_predictions = np.empty(labels.shape, dtype=object)
    for fold in range(5):
        train_fold, valid_fold = folds != fold, folds == fold
        majority = pd.Series(labels[train_fold]).value_counts().index[0]
        majority_predictions[valid_fold] = majority
    results.append({"experiment_id": "B0-majority", "phase": 2, "split": "lot_grouped_5fold",
                    "seed": SEED, "model": "majority_class", "parameters": "most common training class per fold",
                    "macro_f1": f1_score(labels, majority_predictions, average="macro", labels=CLASSES, zero_division=0),
                    "weighted_f1": f1_score(labels, majority_predictions, average="weighted", labels=CLASSES, zero_division=0),
                    "accuracy": float(np.mean(labels == majority_predictions)), "notes": "Majority class selected from each training fold"})
    rng = np.random.default_rng(SEED)
    uniform = rng.choice(CLASSES, size=len(labels))
    results.append({"experiment_id": "B0-uniform", "phase": 2, "split": "lot_grouped_5fold",
                    "seed": SEED, "model": "uniform_random", "parameters": "uniform over 9 classes",
                    "macro_f1": f1_score(labels, uniform, average="macro", labels=CLASSES, zero_division=0),
                    "weighted_f1": f1_score(labels, uniform, average="weighted", labels=CLASSES, zero_division=0),
                    "accuracy": float(np.mean(labels == uniform)), "notes": "Seeded single uniform-random reference draw"})
    write_summary(results)

    print(f"Preparing {len(train):,} training maps at 32x32", flush=True)
    images = np.stack([resize_map(value) for value in train["waferMap"]])
    fractions = np.asarray([((image == 2).sum() / max(1, (image > 0).sum())) for image in images], dtype=np.float32)[:, None]
    results.append(score_cv("B1-failing-fraction", logistic_model(), fractions, labels, folds,
                            "Single overall failing-die fraction feature"))
    print("Completed B0 and B1", flush=True)
    pixels = (images == 2).astype(np.float32).reshape(len(images), -1)
    flat_lr = logistic_model()
    results.append(score_cv("B2-flat-logistic", flat_lr, pixels, labels, folds, "Nearest-neighbor 32x32 maps, flattened"))
    print("Completed B2 logistic", flush=True)
    results.append(score_cv("B2-flat-logistic-random-split", logistic_model(), pixels, labels, random_folds,
                            "Naive random-fold validation; included only to measure lot leakage inflation", "naive_random_5fold"))
    results.append(score_cv("B2-flat-random-forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=2, random_state=SEED, min_samples_leaf=2), pixels, labels, folds, "Nearest-neighbor 32x32 maps, flattened; 100 trees"))
    print("Completed B2 random forest", flush=True)
    results.append(score_cv("B2-flat-random-forest-random-split", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=2, random_state=SEED, min_samples_leaf=2), pixels, labels, random_folds,
                            "Naive random-fold validation; included only to measure lot leakage inflation", "naive_random_5fold"))
    print("Completed B2", flush=True)
    write_summary(results)

    rot_predictions = np.empty(labels.shape, dtype=object)
    for fold in range(5):
        train_rows = folds != fold
        valid = folds == fold
        train_labels = labels[train_rows]
        counts = np.asarray([(train_labels == label).sum() for label in CLASSES])
        weights = len(train_labels) / (len(CLASSES) * np.maximum(counts, 1))
        class_weights = dict(zip(CLASSES, weights))
        model = SGDClassifier(loss="log_loss", class_weight=class_weights, random_state=SEED,
                              learning_rate="optimal", average=True)
        for epoch in range(5):
            for rotation in range(4):
                rotated = np.rot90(images[train_rows], rotation, axes=(1, 2)).copy().reshape(-1, 1024)
                if epoch == 0 and rotation == 0:
                    model.partial_fit(rotated, train_labels, classes=np.asarray(CLASSES))
                else:
                    model.partial_fit(rotated, train_labels)
                del rotated
        rot_predictions[valid] = model.predict(pixels[valid])
    results.append({"experiment_id": "B3-rotation-pixel-logistic", "phase": 2, "split": "lot_grouped_5fold",
                    "seed": SEED, "model": "rotation_augmented_logistic_sgd", "parameters": "32x32; 5 epochs x 4 quarter turns; balanced",
                    "macro_f1": f1_score(labels, rot_predictions, average="macro", labels=CLASSES, zero_division=0),
                    "weighted_f1": f1_score(labels, rot_predictions, average="weighted", zero_division=0),
                    "accuracy": float(np.mean(labels == rot_predictions)), "notes": "Only training-fold wafers augmented"})
    print("Completed B3", flush=True)
    write_summary(results)

    write_log(results)
    summary = [{k: v for k, v in row.items() if k in ("experiment_id", "macro_f1", "weighted_f1", "accuracy")} for row in results]
    for row in summary:
        print(f"{row['experiment_id']}: macro-F1={row['macro_f1']:.4f}, weighted-F1={row['weighted_f1']:.4f}, accuracy={row['accuracy']:.4f}")
    print("Saved experiments/log.csv and experiments/phase2_summary.json")


if __name__ == "__main__":
    main()

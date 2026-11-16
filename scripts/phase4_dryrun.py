"""Phase 4 dry run: try every Phase 4 tool on the validation folds, using the B2 baselines.

The test set is never loaded here. When the Phase 3 pipeline lands, we swap it in for
B2, check it with these same tools, and only then run the real test once.
"""
import json
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from src.evaluation import expected_calibration_error, lot_bootstrap, rotate_map, shift_map

baselines = import_module("scripts.03_baselines")
ROOT, CLASSES = baselines.ROOT, np.array(baselines.CLASSES)
FIGURES = ROOT / "experiments" / "figures"


def fail_pixels(images):
    """32x32 maps -> one row of 1024 fail pixels per wafer (same input as B2)."""
    return (np.stack(images) == 2).reshape(len(images), -1).astype(np.float32)


def out_of_fold(make_model, pixels, labels, folds):
    """Same as B2's cross-validation, but keeps the class probabilities and each fold's model."""
    probs, models = np.zeros((len(labels), len(CLASSES))), []
    for fold in range(5):
        model = make_model().fit(pixels[folds != fold], labels[folds != fold])
        probs[folds == fold] = model.predict_proba(pixels[folds == fold])
        models.append(model)
    return probs, models


def fold_macro_f1(models, pixels, labels, folds):
    """Score each fold's model on its own validation wafers, pooled like the log does."""
    predictions = np.empty(len(labels), dtype=object)
    for fold in range(5):
        predictions[folds == fold] = models[fold].predict(pixels[folds == fold])
    return f1_score(labels, predictions, average="macro", labels=CLASSES, zero_division=0)


def main():
    subset = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    train = split["split"].eq("train").to_numpy()
    raw = subset.loc[train, "waferMap"].tolist()
    labels = split.loc[train, "label"].to_numpy()
    lots = split.loc[train, "lotName"].to_numpy()
    folds = split.loc[train, "fold"].to_numpy()
    images = [baselines.resize_map(m) for m in raw]
    pixels = fail_pixels(images)
    summary = {"note": "Validation folds only (lot-grouped 5-fold). Dry run of the Phase 4 tools on B2."}

    # 1. Scores with 95% ranges from redrawing lots. Both models use the same seed,
    #    so draw i picks the same lots for both and we can subtract them draw by draw.
    per_draw = {}
    for name, make_model in (("B2-flat-logistic", baselines.logistic_model),
                             ("B2-flat-random-forest", baselines.random_forest)):
        print(f"Fitting {name}", flush=True)
        probs, models = out_of_fold(make_model, pixels, labels, folds)
        assert list(models[0].classes_) == list(CLASSES)
        predicted = CLASSES[probs.argmax(axis=1)]
        per_draw[name] = lot_bootstrap(lots, lambda rows: f1_score(
            labels[rows], predicted[rows], average=None, labels=CLASSES, zero_division=0))
        per_class = f1_score(labels, predicted, average=None, labels=CLASSES, zero_division=0)
        summary[name] = {
            "macro_f1": per_class.mean(),
            "macro_f1_95": np.percentile(per_draw[name].mean(axis=1), [2.5, 97.5]).tolist(),
            "per_class_f1": dict(zip(CLASSES, per_class)),
            "per_class_f1_95": dict(zip(CLASSES, np.percentile(per_draw[name], [2.5, 97.5], axis=0).T.tolist())),
            "pr_auc": {c: average_precision_score(labels == c, probs[:, i]) for i, c in enumerate(CLASSES)},
            "ece": expected_calibration_error(probs, labels, CLASSES),
        }
        if name == "B2-flat-logistic":
            logistic_models, logistic_probs = models, probs
    gap = per_draw["B2-flat-random-forest"].mean(axis=1) - per_draw["B2-flat-logistic"].mean(axis=1)
    summary["forest_minus_logistic_95"] = np.percentile(gap, [2.5, 97.5]).tolist()

    # 2. Rotation curve: turn every validation wafer, score the already-trained fold models.
    print("Rotation curve", flush=True)
    angles = list(range(0, 360, 15))
    rotation = [fold_macro_f1(logistic_models, fail_pixels([rotate_map(m, a) for m in raw]), labels, folds)
                for a in angles]

    # 3. Shift curve: slide up, down, left and right by d pixels, average the four scores.
    print("Shift curve", flush=True)
    shifts = [0, 1, 2, 3]
    shift = [np.mean([fold_macro_f1(logistic_models, fail_pixels([shift_map(i, dy, dx) for i in images]),
                                    labels, folds) for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d))])
             for d in shifts]

    # 4. Learning curve: train on a share of each fold's training LOTS (3 random picks below 100%).
    print("Learning curve", flush=True)
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    learning = []
    for fraction in fractions:
        scores = []
        for seed in range(1 if fraction == 1.0 else 3):
            rng = np.random.default_rng(seed)
            predictions = np.empty(len(labels), dtype=object)
            for fold in range(5):
                fold_lots = np.unique(lots[folds != fold])
                keep = rng.choice(fold_lots, size=max(1, round(fraction * len(fold_lots))), replace=False)
                fit_rows = (folds != fold) & np.isin(lots, keep)
                model = baselines.logistic_model().fit(pixels[fit_rows], labels[fit_rows])
                predictions[folds == fold] = model.predict(pixels[folds == fold])
            scores.append(f1_score(labels, predictions, average="macro", labels=CLASSES, zero_division=0))
        learning.append(float(np.mean(scores)))
        print(f"  {fraction:.0%}: {learning[-1]:.4f}", flush=True)
    summary["curves"] = {"rotation": dict(zip(angles, rotation)), "shift": dict(zip(shifts, shift)),
                         "learning": dict(zip(fractions, learning))}

    # 5. Error analysis on the logistic model: where do the mistakes go?
    predicted = CLASSES[logistic_probs.argmax(axis=1)]
    matrix = confusion_matrix(labels, predicted, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(9), yticks=range(9), xlabel="predicted", ylabel="true",
           title="B2 logistic, validation (each row adds to 1)")
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_dryrun_confusion.png", dpi=120)

    # The 6 biggest mix-ups, and the 2 most confident mistakes from each = 12 wafers to look at.
    off_diagonal = matrix - np.diag(np.diag(matrix))
    top_pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1][:6], matrix.shape))[0]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for k, (t, p) in enumerate(top_pairs):
        wrong = np.flatnonzero((labels == CLASSES[t]) & (predicted == CLASSES[p]))
        for n, row in enumerate(wrong[np.argsort(-logistic_probs[wrong].max(axis=1))][:2]):
            ax = axes.flat[2 * k + n]
            ax.imshow(images[row], cmap="viridis", vmin=0, vmax=2)
            ax.set_title(f"{CLASSES[t]} -> {CLASSES[p]}\n{logistic_probs[row].max():.0%} sure", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_dryrun_mistakes.png", dpi=120)

    # 6. One picture with the three curves.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(angles, rotation, marker="o")
    axes[0].set(xlabel="rotation (degrees)", ylabel="macro-F1", title="Rotation", xticks=range(0, 360, 45))
    axes[1].plot(shifts, shift, marker="o")
    axes[1].set(xlabel="shift (pixels)", title="Shift", xticks=shifts)
    axes[2].plot(fractions, learning, marker="o")
    axes[2].set(xlabel="share of training lots", title="Learning curve", xscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("B2 logistic on validation folds (dry run)")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_dryrun_curves.png", dpi=120)

    (ROOT / "experiments" / "phase4_dryrun.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    for name in ("B2-flat-logistic", "B2-flat-random-forest"):
        low, high = summary[name]["macro_f1_95"]
        print(f"{name:<24} macro-F1 {summary[name]['macro_f1']:.4f} ({low:.4f} to {high:.4f})"
              f"  ECE {summary[name]['ece']:.4f}")
    low, high = summary["forest_minus_logistic_95"]
    print(f"Forest minus logistic: {low:+.4f} to {high:+.4f}")
    print("Saved experiments/phase4_dryrun.json and 3 figures in experiments/figures/")


if __name__ == "__main__":
    main()

"""Phase 4, part 2: why the final model works, and where it breaks.

Uses the models frozen by 05_final_test.py (nothing here changes them):
  1. rotation curve  - test wafers turned 0..345 degrees in 15 degree steps
  2. shift curve     - test wafers slid 0..3 pixels up/down/left/right
  3. learning curve  - VALIDATION folds only: retrain on 1%..100% of training lots
  4. error analysis  - test confusion matrix + the 12 most telling mistakes
  5. filter picture  - the 32 learned 5x5 k-means filters

Run: python main.py analysis   (after python main.py final-test)
"""
import json
from importlib import import_module

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.evaluation import rotate_map, shift_map
from src.features import ConvFeatures
from src.models import make_model
from src.preprocessing import prepare

baselines = import_module("scripts.03_baselines")
ROOT, CLASSES, SEED = baselines.ROOT, np.array(baselines.CLASSES), baselines.SEED
FIGURES = ROOT / "experiments" / "figures"


def macro_f1(labels, predicted):
    return f1_score(labels, predicted, average="macro", labels=CLASSES, zero_division=0)


def fail_pixels(images):
    """32x32 maps (0/1/2) -> 1024 flat fail pixels, the B2 input."""
    return (np.stack(images) == 2).reshape(len(images), -1).astype(np.float32)


def main():
    saved = joblib.load(ROOT / "data" / "final_models.joblib")
    conv, models = saved["conv"], saved["models"]
    recipe = json.loads((ROOT / "experiments" / "final_config.json").read_text())["recipe"]
    subset = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    labels, lots, folds = split["label"].to_numpy(), split["lotName"].to_numpy(), split["fold"].to_numpy()
    test = split["split"].eq("test").to_numpy()
    fails, masks, extras = prepare(subset["waferMap"], recipe["size"], recipe["denoise"])
    images = np.stack([baselines.resize_map(m) for m in subset["waferMap"]])  # B2's 0/1/2 pictures
    y = labels[test]

    def score_all(change):
        """Apply `change` to every test wafer's 32x32 maps, then score the frozen models."""
        f = np.stack([change(m) for m in fails[test]])
        m = np.stack([change(m) for m in masks[test]])
        pixels = fail_pixels([change(i) for i in images[test]])
        x = np.hstack([conv.transform(f, m), extras[test]])
        return {"final-logreg": macro_f1(y, models["final-logreg"].predict(x)),
                "B2-flat-logistic": macro_f1(y, models["B2-flat-logistic"].predict(pixels)),
                "B2-flat-random-forest": macro_f1(y, models["B2-flat-random-forest"].predict(pixels))}

    # 1. Rotation (test). 0 degrees reproduces the final test scores exactly.
    print("Rotation curve", flush=True)
    angles = list(range(0, 360, 15))
    rotation = {a: score_all(lambda m: rotate_map(m, a)) for a in angles}

    # 2. Shift (test): average of up, down, left and right.
    print("Shift curve", flush=True)
    shift = {}
    for d in range(4):
        runs = [score_all(lambda m: shift_map(m, dy, dx)) for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d))]
        shift[d] = {name: float(np.mean([r[name] for r in runs])) for name in runs[0]}

    # 3. Learning curve (validation folds only, never test). Filters are relearned
    #    from the smaller sample each time, or the 1% model would secretly use 100%.
    print("Learning curve", flush=True)
    train = ~test
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    learning = {}
    for fraction in fractions:
        ours, flat = [], []
        for seed in range(1 if fraction == 1.0 else 3):
            rng = np.random.default_rng(seed)
            pred_ours, pred_flat = np.empty(train.sum(), dtype=object), np.empty(train.sum(), dtype=object)
            for fold in range(5):
                valid = train & (folds == fold)
                fold_lots = np.unique(lots[train & (folds != fold)])
                keep = rng.choice(fold_lots, size=max(1, round(fraction * len(fold_lots))), replace=False)
                fit = train & (folds != fold) & np.isin(lots, keep)
                c = ConvFeatures(recipe["method"], recipe["k"], recipe["rotate"], recipe["pooling"], seed=SEED)
                c.fit(fails[fit], masks[fit], labels[fit])
                x_fit = np.hstack([c.transform(fails[fit], masks[fit]), extras[fit]])
                x_valid = np.hstack([c.transform(fails[valid], masks[valid]), extras[valid]])
                # Plain logistic regression: temperature scaling never changes the predicted class.
                at = folds[train] == fold
                pred_ours[at] = make_model("logreg").model.fit(x_fit, labels[fit]).predict(x_valid)
                b2 = baselines.logistic_model().fit(fail_pixels(images[fit]), labels[fit])
                pred_flat[at] = b2.predict(fail_pixels(images[valid]))
            ours.append(macro_f1(labels[train], pred_ours))
            flat.append(macro_f1(labels[train], pred_flat))
        learning[fraction] = {"final-logreg": float(np.mean(ours)), "B2-flat-logistic": float(np.mean(flat))}
        print(f"  {fraction:.0%}: ours {learning[fraction]['final-logreg']:.4f}, "
              f"flat {learning[fraction]['B2-flat-logistic']:.4f}", flush=True)

    # 4. Error analysis (test).
    x = np.hstack([conv.transform(fails[test], masks[test]), extras[test]])
    probs = models["final-logreg"].predict_proba(x)
    predicted = CLASSES[probs.argmax(axis=1)]
    matrix = confusion_matrix(y, predicted, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(9), yticks=range(9), xlabel="predicted", ylabel="true",
           title="Final model on the test set (each row adds to 1)")
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_confusion_matrix.png", dpi=120)

    # The 6 biggest mix-ups, and the 2 most confident mistakes from each = 12 wafers.
    off_diagonal = matrix - np.diag(np.diag(matrix))
    top_pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1][:6], matrix.shape))[0]
    test_images = images[test]
    mistakes = []
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for k, (t, p) in enumerate(top_pairs):
        wrong = np.flatnonzero((y == CLASSES[t]) & (predicted == CLASSES[p]))
        for n, row in enumerate(wrong[np.argsort(-probs[wrong].max(axis=1))][:2]):
            ax = axes.flat[2 * k + n]
            ax.imshow(test_images[row], cmap="viridis", vmin=0, vmax=2)
            ax.set_title(f"{CLASSES[t]} -> {CLASSES[p]}\n{probs[row].max():.0%} sure", fontsize=8)
            mistakes.append({"source_index": int(split.loc[test, "source_index"].iloc[row]), "true": CLASSES[t],
                             "predicted": CLASSES[p], "confidence": float(probs[row].max())})
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Most confident test mistakes (purple = off wafer, teal = pass, yellow = fail)")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_test_mistakes.png", dpi=120)

    # 5. The 32 learned filters (shown in whitened space, as the model uses them).
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    for ax, f in zip(axes.flat, conv.filters):
        ax.imshow(f.reshape(5, 5), cmap="gray")
        ax.axis("off")
    fig.suptitle("The 32 k-means filters (5x5, learned without backprop)")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_learned_filters.png", dpi=120)

    # 6. One picture with the three curves.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name in ("final-logreg", "B2-flat-random-forest", "B2-flat-logistic"):
        axes[0].plot(angles, [rotation[a][name] for a in angles], marker="o", label=name)
        axes[1].plot(list(shift), [shift[d][name] for d in shift], marker="o", label=name)
        if name in learning[1.0]:
            axes[2].plot(fractions, [learning[f][name] for f in fractions], marker="o", label=name)
    axes[0].set(xlabel="rotation (degrees)", ylabel="macro-F1", title="Rotation (test)", xticks=range(0, 360, 45))
    axes[1].set(xlabel="shift (pixels)", title="Shift (test)", xticks=list(shift))
    axes[2].set(xlabel="share of training lots", title="Learning curve (validation)", xscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_curves.png", dpi=120)

    summary = {"rotation_test": rotation, "shift_test": shift, "learning_validation": learning,
               "confusion_test": {c: dict(zip(CLASSES, row)) for c, row in zip(CLASSES, matrix)},
               "mistakes_test": mistakes}
    (ROOT / "experiments" / "phase4_analysis.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    for a in (0, 15, 45, 90, 180):
        print(f"rotation {a:>3}: " + "  ".join(f"{k} {v:.4f}" for k, v in rotation[a].items()))
    for d in shift:
        print(f"shift {d}: " + "  ".join(f"{k} {v:.4f}" for k, v in shift[d].items()))
    print("Saved experiments/phase4_analysis.json and 4 figures (06_*.png)")


if __name__ == "__main__":
    main()

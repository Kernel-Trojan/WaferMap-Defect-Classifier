"""Reference: one basic textbook CNN trained WITH backprop, to compare against our no-backprop pipeline.

Kept fair on purpose:
  - same lot-grouped split, and the same input our pipeline sees: the 32x32
    isolated-denoised fail map + wafer mask (2 channels)
  - balanced class weights, like our logistic regression
  - the number of epochs is picked on the 5 validation folds only; then we train
    on all training wafers and score the locked test set once (refuses to run twice)

Run: python main.py reference-cnn
"""
import json
import time
from datetime import date
from importlib import import_module

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn

from src.evaluation import lot_bootstrap, rotate_map, shift_map
from src.preprocessing import prepare

pipeline = import_module("scripts.03_pipeline")
ROOT, SEED, CLASSES = pipeline.ROOT, pipeline.SEED, np.array(pipeline.CLASSES)
RESULTS = ROOT / "experiments" / "reference_cnn.json"
MAX_EPOCHS, BATCH = 30, 128


def make_cnn():
    """Textbook small CNN: 3 x (3x3 conv -> ReLU -> 2x2 max-pool), then two dense layers."""
    return nn.Sequential(
        nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32 -> 16x16
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16 -> 8x8
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8 -> 4x4
        nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 9))


def to_tensor(fails, masks):
    return torch.tensor(np.stack([fails, masks], axis=1), dtype=torch.float32)


def train(x, y, epochs, check=None):
    """Adam + class-weighted cross-entropy. check(model) runs after every epoch."""
    torch.manual_seed(SEED)
    model = make_cnn()
    counts = np.bincount(y, minlength=9)
    weights = torch.tensor(len(y) / (9 * np.maximum(counts, 1)), dtype=torch.float32)  # "balanced"
    loss_fn, optimiser = nn.CrossEntropyLoss(weight=weights), torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.tensor(y)
    history = []
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(y)).split(BATCH):
            optimiser.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            optimiser.step()
        if check:
            history.append(check(model))
    return model, history


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(b) for b in x.split(2048)]).argmax(1).numpy()


def macro_f1(y, predicted):
    return f1_score(y, predicted, average="macro", labels=range(9), zero_division=0)


def main():
    if RESULTS.exists():
        raise SystemExit("The reference CNN was already scored on the test set (experiments/reference_cnn.json).")
    subset = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    train_rows = np.flatnonzero(split["split"].eq("train"))
    test_rows = np.flatnonzero(split["split"].eq("test"))
    y = np.searchsorted(CLASSES, split["label"].to_numpy())  # class names -> 0..8
    folds = split["fold"].to_numpy()
    fails, masks, _ = prepare(subset["waferMap"], 32, "isolated")
    x = to_tensor(fails, masks)

    # 1. Pick the number of epochs on the validation folds (test never touched here).
    by_epoch = np.zeros((MAX_EPOCHS, len(train_rows)), dtype=int)
    for fold in range(5):
        fit, valid = train_rows[folds[train_rows] != fold], train_rows[folds[train_rows] == fold]
        start = time.time()
        _, history = train(x[fit], y[fit], MAX_EPOCHS, lambda model: predict(model, x[valid]))
        by_epoch[:, folds[train_rows] == fold] = np.array(history)
        print(f"fold {fold} trained ({time.time() - start:.0f}s)", flush=True)
    val_curve = [macro_f1(y[train_rows], by_epoch[e]) for e in range(MAX_EPOCHS)]
    epochs = int(np.argmax(val_curve)) + 1
    val_pred = by_epoch[epochs - 1]
    val_fold_std = np.std([macro_f1(y[train_rows][folds[train_rows] == f], val_pred[folds[train_rows] == f])
                           for f in range(5)])
    print(f"Best on validation: {epochs} epochs, macro-F1 {val_curve[epochs - 1]:.4f}", flush=True)

    # 2. Train once on all training wafers, then the one look at the test set.
    start = time.time()
    model, _ = train(x[train_rows], y[train_rows], epochs)
    train_seconds = time.time() - start
    torch.save(model.state_dict(), ROOT / "data" / "reference_cnn.pt")
    cnn = predict(model, x[test_rows])
    saved = pd.read_csv(ROOT / "experiments" / "final_test_predictions.csv")
    if not (saved["source_index"].to_numpy() == split.loc[test_rows, "source_index"].to_numpy()).all():
        raise ValueError("Saved test predictions don't line up with the split")
    ours = np.searchsorted(CLASSES, saved["final-logreg"].to_numpy())
    y_test, lots = y[test_rows], split.loc[test_rows, "lotName"].to_numpy()

    # Same seed = same redrawn lots for both models, so the gap is compared draw by draw.
    cnn_draws = lot_bootstrap(lots, lambda r: macro_f1(y_test[r], cnn[r]))
    ours_draws = lot_bootstrap(lots, lambda r: macro_f1(y_test[r], ours[r]))

    table = {}
    for name, predicted in (("ours: k-means + logreg (no backprop)", ours), ("textbook CNN (backprop)", cnn)):
        p, r, f, support = precision_recall_fscore_support(y_test, predicted, labels=range(9), zero_division=0)
        table[name] = pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=CLASSES)
    table = pd.concat(table, axis=1)
    table.loc["macro avg"] = table.mean()
    table.insert(0, "test wafers", list(support) + [support.sum()])
    table.round(4).to_csv(ROOT / "experiments" / "reference_cnn_comparison.csv")

    # 3. Rotation and shift on the test set (no rotated copies were used in training).
    def cnn_score(change):
        f = np.stack([change(m) for m in fails[test_rows]])
        m = np.stack([change(m) for m in masks[test_rows]])
        return macro_f1(y_test, predict(model, to_tensor(f, m)))

    angles = list(range(0, 360, 15))
    rotation = {a: cnn_score(lambda m: rotate_map(m, a)) for a in angles}
    shift = {d: float(np.mean([cnn_score(lambda m: shift_map(m, dy, dx))
                               for dy, dx in ((d, 0), (-d, 0), (0, d), (0, -d))])) for d in range(4)}
    analysis = json.loads((ROOT / "experiments" / "phase4_analysis.json").read_text())
    ours_rotation = {int(a): v["final-logreg"] for a, v in analysis["rotation_test"].items()}
    ours_shift = {int(d): v["final-logreg"] for d, v in analysis["shift_test"].items()}

    # 4. How many numbers each model had to learn.
    final = joblib.load(ROOT / "data" / "final_models.joblib")
    conv, logreg = final["conv"], final["models"]["final-logreg"]
    scaler, linear = logreg.model[0], logreg.model[-1]
    ours_params = (conv.filters.size + conv.zca.size + conv.mean.size + scaler.mean_.size + scaler.scale_.size
                   + linear.coef_.size + linear.intercept_.size + 1)  # +1 = temperature
    cnn_params = sum(p.numel() for p in model.parameters())

    summary = {
        "architecture": str(make_cnn()), "epochs_picked_on_validation": epochs, "max_epochs": MAX_EPOCHS,
        "validation_macro_f1_by_epoch": val_curve, "cnn_train_seconds_all_training_wafers": train_seconds,
        "test_macro_f1": {"textbook CNN": [macro_f1(y_test, cnn), *np.percentile(cnn_draws, [2.5, 97.5])],
                          "ours": [macro_f1(y_test, ours), *np.percentile(ours_draws, [2.5, 97.5])]},
        "ours_minus_cnn": [macro_f1(y_test, ours) - macro_f1(y_test, cnn),
                           *np.percentile(ours_draws - cnn_draws, [2.5, 97.5])],
        "test_accuracy": {"textbook CNN": float(np.mean(cnn == y_test)), "ours": float(np.mean(ours == y_test))},
        "rotation_test": {a: {"textbook CNN": rotation[a], "ours": ours_rotation[a]} for a in angles},
        "shift_test": {d: {"textbook CNN": shift[d], "ours": ours_shift[d]} for d in shift},
        "learned_numbers": {"textbook CNN": cnn_params, "ours": int(ours_params)},
    }
    RESULTS.write_text(json.dumps(summary, indent=2, default=float) + "\n")

    row = {"date": date.today().isoformat(), "phase": "ref", "seed": SEED, "model": "textbook_cnn",
           "parameters": f"3x(conv3x3+ReLU+maxpool) 16/32/64, dense 128, dropout 0.3, Adam 1e-3, batch {BATCH}, "
                         f"balanced weights, {epochs} epochs (picked on validation), input isolated fail map + mask"}
    pipeline.write_log([
        {**row, "experiment_id": "REF-cnn-validation", "description": "Reference textbook CNN (backprop), validation",
         "split": "lot_grouped_5fold", "macro_f1": val_curve[epochs - 1], "macro_f1_fold_std": val_fold_std,
         "weighted_f1": f1_score(y[train_rows], val_pred, average="weighted", labels=range(9), zero_division=0),
         "accuracy": float(np.mean(val_pred == y[train_rows])),
         "notes": "epoch count picked on these same folds, so slightly optimistic"},
        {**row, "experiment_id": "REF-cnn-test", "description": "Reference textbook CNN (backprop), locked test once",
         "split": "locked_test_20pct_lots", "macro_f1": macro_f1(y_test, cnn), "macro_f1_fold_std": "",
         "weighted_f1": f1_score(y_test, cnn, average="weighted", labels=range(9), zero_division=0),
         "accuracy": float(np.mean(cnn == y_test)),
         "notes": "95% lot bootstrap macro-F1 {:.4f} to {:.4f}".format(*np.percentile(cnn_draws, [2.5, 97.5]))}])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(range(1, MAX_EPOCHS + 1), val_curve, marker="o", ms=3, label="textbook CNN")
    axes[0].axhline(float(pd.read_csv(ROOT / "experiments" / "log.csv").set_index("experiment_id")
                          .loc["P3-kmeans-logreg", "macro_f1"]), color="C1", ls="--", label="ours")
    axes[0].set(xlabel="epoch", ylabel="macro-F1", title="Validation (5-fold) while training")
    axes[1].plot(angles, [rotation[a] for a in angles], marker="o", label="textbook CNN")
    axes[1].plot(angles, [ours_rotation[a] for a in angles], marker="o", label="ours")
    axes[1].set(xlabel="rotation (degrees)", title="Rotation (test)", xticks=range(0, 360, 45))
    axes[2].plot(list(shift), list(shift.values()), marker="o", label="textbook CNN")
    axes[2].plot(list(shift), [ours_shift[d] for d in shift], marker="o", label="ours")
    axes[2].set(xlabel="shift (pixels)", title="Shift (test)", xticks=list(shift))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "experiments" / "figures" / "08_reference_cnn.png", dpi=120)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    for name, (value, low, high) in summary["test_macro_f1"].items():
        print(f"{name:<14} test macro-F1 {value:.4f} ({low:.4f} to {high:.4f})")
    print("ours minus CNN: {:+.4f} ({:+.4f} to {:+.4f})".format(*summary["ours_minus_cnn"]))
    for a in (0, 45, 90, 180):
        print(f"rotation {a:>3}: CNN {rotation[a]:.4f}  ours {ours_rotation[a]:.4f}")
    print(f"shift 3: CNN {shift[3]:.4f}  ours {ours_shift[3]:.4f}")
    print(f"learned numbers: CNN {cnn_params:,}  ours {int(ours_params):,}")


if __name__ == "__main__":
    main()

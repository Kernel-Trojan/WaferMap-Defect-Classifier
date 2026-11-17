"""Export every wafer and number that the website in site/ shows.

Run with: python main.py site-data

The site is static (HTML, CSS, JS), so everything it draws is baked into a few data
files. They are written as small JavaScript files (window.SITE.<name> = {...}) rather
than .json, so the page also works when index.html is opened straight from disk.

  site/data/results.js   every number on the page (read from experiments/ and splits/)
  site/data/wafers.js    real wafer maps for the pictures and interactions
  site/data/model.js     the CNN-style pipeline's numbers, for the in-browser demos
  experiments/site_stats.json   a copy of results.js, logged next to the other results

What this script does NOT do: it never scores the test set and never chooses anything
by looking at test labels. The only test labels it reads are the ones already saved in
experiments/final_test_predictions.csv by the one final run.

The model for the demos (filter explorer, rotation playground, draw-a-wafer, confidence
in the error explorer) comes from data/final_models.joblib when that file exists. It is
gitignored and only exists where `final-test` was run. Without it, the frozen recipe is
refit on the training lots with the same seed (never on test wafers), and every demo is
labelled as using this refit. As a check, the refit's predictions on the test wafers are
compared with the saved predictions; only the agreement rate is recorded, no score.
"""
from __future__ import annotations

import base64
import csv
import html
import json
import math
import re
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.evaluation import rotate_map
from src.features import ConvFeatures
from src.models import make_model
from src.preprocessing import denoise, prepare, resize

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SIZE = 32
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
EXP = ROOT / "experiments"

# Wafers chosen by eye from contact sheets, by source_index (row number in LSWMD.pkl).
# Everything in the data story (problem, data, split, preprocessing) is a training wafer.
STORY_EXAMPLES = {  # first one per class is the "headline" example
    "Center": [394490], "Donut": [377785], "Edge-Loc": [718371], "Edge-Ring": [21297], "Loc": [270477],
    "Near-full": [245311], "Random": [774230, 243283], "Scratch": [174905, 781498], "none": [722666],
}
EXAMPLES_PER_CLASS = 6                   # gallery and stepper cycle through this many
SCRATCH_DEMOS = [174905, 781498, 788675, 381283]   # thin scratches the median filter erases
NOISE_DEMO = 717909                      # a healthy wafer with scattered lonely dies
SIZE_EXTREMES = (8430, 244294)           # smallest and largest grid among our wafers
SAME_LOT = {"lot": "lot11879", "label": "Edge-Ring",
            "others": ["lot11589", "lot44802", "lot21580", "lot11631", "lot6697", "lot25392"]}
EXTRA_LOTS = [("lot5447", "Loc"), ("lot7747", "Donut"), ("lot10942", "Edge-Loc"), ("lot16072", "Center")]
SPLIT_DEMO_LOTS = 40                     # lots in the "split it yourself" demo
ROTATION_DEMO_PER_CLASS = 2              # test wafers per class in the rotation playground
CONFUSION_EXAMPLES = 6                   # test wafers per confusion-matrix cell
NEW_FILES = ["experiments/site_stats.json", "scripts/11_site_data.py"]


# --------------------------------------------------------------------- small helpers
def r4(value) -> float:
    """Keep 6 decimals: the page rounds to 3, and rounding twice (4 then 3) can be off by one."""
    return round(float(value), 6)


def triple(values) -> list[float]:
    return [r4(v) for v in values]


def pack32(fail: np.ndarray, mask: np.ndarray) -> str:
    """A 32x32 map as 0 off / 1 pass / 2 fail, 2 bits per die, base64 (344 characters)."""
    mask = mask.astype(np.uint8)
    codes = (mask + (fail.astype(np.uint8) & mask)).ravel()
    quads = codes.reshape(-1, 4)
    packed = quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
    return base64.b64encode(packed.astype(np.uint8).tobytes()).decode("ascii")


def weighted_sum(image: np.ndarray) -> int:
    """Tiny fingerprint so the page can prove its JavaScript matches this Python."""
    flat = image.ravel().astype(np.int64)
    return int((flat * (np.arange(flat.size) % 97 + 1)).sum())


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def first_commit(path: str) -> dict:
    line = git("log", "--diff-filter=A", "--format=%h|%ad", "--date=iso", "--", path).splitlines()[-1]
    commit, when = line.split("|")
    return {"commit": commit, "time": when[11:16], "date": when[:10]}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def macro_f1(truth, predicted) -> float:
    _, _, f1, _ = precision_recall_fscore_support(truth, predicted, labels=CLASSES, zero_division=0)
    return float(f1.mean())


def exact_prf(precision: float, recall: float, n: int) -> dict:
    """Rebuild exact precision, recall and F1 from 4-decimal values in a CSV.

    recall x (wafers of the class) gives the true positives, and true positives /
    precision gives how many wafers were called that class. Both are whole numbers, so
    the exact fractions come back and the page never rounds a rounded number again.
    """
    tp = round(recall * n)
    called = round(tp / precision) if precision else 0
    p = tp / called if called else 0.0
    r = tp / n
    return {"precision": r4(p), "recall": r4(r), "f1": r4(2 * tp / (n + called) if n + called else 0.0)}


def exact_table(rows: dict, counts: dict) -> dict:
    out = {c: exact_prf(rows[c]["precision"], rows[c]["recall"], counts[c]) for c in CLASSES}
    out["macro avg"] = {k: r4(np.mean([out[c][k] for c in CLASSES])) for k in ("precision", "recall", "f1")}
    return out


# ------------------------------------------------------------------- numbers: data
def data_numbers(data, split, fails32):
    audit = json.loads((EXP / "audit_summary.json").read_text())
    summary = json.loads((ROOT / "splits" / "split_summary.json").read_text())
    shapes = {eval(key): count for key, count in audit["wafer_map_shapes_source"].items()}
    by_area = sorted(shapes, key=lambda s: s[0] * s[1])
    longest = np.maximum(data["h"], data["w"])
    small, large = (data.index[data["source_index"] == i][0] for i in SIZE_EXTREMES)
    labeled = {c: audit["class_counts"][c] for c in CLASSES}
    builtin = audit["builtin_split_check"]

    # The two leak numbers (handout section 5). Validation folds: share of each fold's
    # wafers whose lot also appears in the other folds, for our folds and naive ones.
    train = split[split["split"] == "train"]
    fold_leak = {col: float(np.mean([train[train[col] == f]["lotName"].isin(set(train[train[col] != f]["lotName"])).mean()
                                     for f in range(5)])) for col in ("fold", "val_fold_random")}
    # The one-line split the site shows, run on our 35,519 wafers.
    positions = np.arange(len(split))
    naive_train, naive_test = train_test_split(positions, test_size=0.20, shuffle=True, random_state=SEED)
    lots = split["lotName"].to_numpy()
    naive_test_leak = float(np.isin(lots[naive_test], list(set(lots[naive_train]))).mean())

    # How alike are lot-mates? IoU of failing dies on cleaned 32x32 maps, training wafers:
    # consecutive wafers of one (lot, class) against 4,000 random same-class pairs from different lots.
    rng = np.random.default_rng(0)
    is_train = (split["split"] == "train").to_numpy()
    overlap = {}
    for label in CLASSES[:-1]:
        rows = np.flatnonzero(is_train & (split["label"] == label).to_numpy())
        same = []
        for _, members in split.iloc[rows].groupby("lotName").indices.items():
            members = rows[members]
            same += [iou(fails32[a], fails32[b]) for a, b in zip(members[:-1], members[1:])]
        other = []
        while len(other) < 4000:
            a, b = rng.choice(rows, 2, replace=False)
            if lots[a] != lots[b]:
                other.append(iou(fails32[a], fails32[b]))
        if len(same) >= 20:
            overlap[label] = {"same_lot": r4(np.median(same)), "other_lots": r4(np.median(other)), "pairs": len(same)}

    return {
        "audit": {
            "source_rows": audit["source_rows"], "labeled_rows": audit["labeled_rows"],
            "unlabeled_rows": audit["source_rows"] - audit["labeled_rows"],
            "unlabeled_share": (audit["source_rows"] - audit["labeled_rows"]) / audit["source_rows"],
            "labeled_share": audit["labeled_rows"] / audit["source_rows"],
            "none_rows": audit["none_rows"], "none_share_of_labeled": audit["none_rows"] / audit["labeled_rows"],
            "defect_rows": audit["defect_rows"], "defect_share_of_labeled": audit["defect_rows"] / audit["labeled_rows"],
            "none_kept": audit["subset_class_counts"]["none"],
            "none_dropped": audit["none_rows"] - audit["subset_class_counts"]["none"],
            "subset_rows": audit["subset_rows"],
            "subset_class_counts": {c: audit["subset_class_counts"][c] for c in CLASSES},
            "labeled_class_counts": labeled,
            "rarest_to_commonest": max(labeled[c] for c in CLASSES[:-1]) / min(labeled[c] for c in CLASSES[:-1]),
            "unique_lots": audit["unique_lots"], "full_lots": audit["lot_size_distribution"]["25"],
            "missing_lot_rows": audit["missing_lot_rows"], "die_count_matches": audit["die_size_matches_positive_map_cells"],
            "builtin": {"lot_overlap": builtin["lot_overlap"],
                        "edge_ring_train": builtin["training_share_by_class"]["Edge-Ring"],
                        "none_train": builtin["training_share_by_class"]["none"],
                        "none_test": 1 - builtin["training_share_by_class"]["none"]},
        },
        "sizes": {
            "file_shapes": len(shapes), "file_smallest": list(by_area[0]), "file_largest": list(by_area[-1]),
            "subset_shapes": int(data.groupby(["h", "w"]).ngroups),
            "subset_min_side": int(min(data["h"].min(), data["w"].min())), "subset_max_side": int(longest.max()),
            "median_longest_side": float(np.median(longest)),
            "small": {"h": int(data.loc[small, "h"]), "w": int(data.loc[small, "w"]), "cells": int(data.loc[small, "h"] * data.loc[small, "w"])},
            "large": {"h": int(data.loc[large, "h"]), "w": int(data.loc[large, "w"]), "cells": int(data.loc[large, "h"] * data.loc[large, "w"])},
            "ratio": float(data.loc[large, "h"] * data.loc[large, "w"] / (data.loc[small, "h"] * data.loc[small, "w"])),
            "share_side_under_32": float(((data["h"] < SIZE) | (data["w"] < SIZE)).mean()),
        },
        "split": {key: summary[key] for key in ("rows", "groups", "test_rows", "test_fraction", "test_lots",
                                                "train_rows", "train_lots", "test_class_counts", "group_overlap_train_test",
                                                "groups_joining_multiple_lots", "max_lots_in_one_group")}
                 | {"group_sizes": summary["group_sizes"],
                    "duplicates": {k: v for k, v in summary["duplicate_maps"].items() if k != "label_conflicts"},
                    "fold_rows": [summary["folds"][str(f)]["validation_rows"] for f in range(5)],
                    "fold_lots": [summary["folds"][str(f)]["validation_lots"] for f in range(5)],
                    "fold_class_counts": [summary["folds"][str(f)]["class_counts"] for f in range(5)]},
        "leak": {"validation_naive": fold_leak["val_fold_random"], "validation_ours": fold_leak["fold"],
                 "test_naive": naive_test_leak, "test_ours": 0.0},
        "overlap": overlap,
    }


def summary_numbers(data, split):
    """Median per class, training wafers, of the two saved numbers under each cleanup."""
    train = (split["split"] == "train").to_numpy()
    _, _, iso = prepare(data.loc[train, "waferMap"], SIZE, "isolated")
    _, _, med = prepare(data.loc[train, "waferMap"], SIZE, "median")
    labels = split.loc[train, "label"].to_numpy()
    return {c: {"failing_share": r4(np.median(iso[labels == c, 0])),
                "removed_isolated": r4(np.median(iso[labels == c, 1])),
                "removed_median": r4(np.median(med[labels == c, 1]))} for c in CLASSES}


# ---------------------------------------------------------------- numbers: results
def parse_per_class(notes: str) -> dict:
    return {c: float(v) for c, v in re.findall(r"(\S+)=([\d.]+)", notes) if c in CLASSES}


def results_numbers(predictions):
    with (EXP / "log.csv").open() as stream:
        log = {row["experiment_id"]: row for row in csv.DictReader(stream)}
    final = json.loads((EXP / "final_test.json").read_text())
    analysis = json.loads((EXP / "phase4_analysis.json").read_text())
    cnn = json.loads((EXP / "reference_cnn.json").read_text())
    filters = json.loads((EXP / "filter_check.json").read_text())
    cnn_table = pd.read_csv(EXP / "reference_cnn_comparison.csv", header=[0, 1], index_col=0)
    denoise_table = pd.read_csv(EXP / "test_denoise_comparison.csv", header=[0, 1], index_col=0)
    denoise_gap = json.loads((EXP / "test_denoise_comparison.json").read_text())["median_minus_isolated_macro_f1"]
    real = pd.read_csv(EXP / "real_world_precision.csv", index_col=0)

    def row(name):
        r = log[name]
        out = {"macro_f1": r4(r["macro_f1"]), "accuracy": r4(r["accuracy"])}
        if r["macro_f1_fold_std"]:
            out["fold_std"] = r4(r["macro_f1_fold_std"])
        if "per-class F1" in r["notes"]:
            out["per_class"] = parse_per_class(r["notes"])
        return out

    validation = {name: row(name) for name in log if log[name]["split"] in ("lot_grouped_5fold", "naive_random_5fold")}

    # Per-class precision / recall for every model, recomputed from the saved test predictions.
    names = {"ours": "final-logreg", "svm": "final-svm", "forest": "B2-flat-random-forest", "logistic": "B2-flat-logistic"}
    per_class = {}
    for key, column in names.items():
        p, r, f, _ = precision_recall_fscore_support(predictions["label"], predictions[column], labels=CLASSES, zero_division=0)
        per_class[key] = {c: {"precision": r4(p[i]), "recall": r4(r[i]), "f1": r4(f[i]),
                              "range": triple(final[column]["per_class_f1"][c][1:])} for i, c in enumerate(CLASSES)}
    cnn_name = next(c for c in cnn_table.columns.levels[0] if "CNN" in c)
    test_counts = predictions["label"].value_counts().to_dict()
    per_class["cnn"] = exact_table({c: {k: float(cnn_table.loc[c, (cnn_name, k)]) for k in ("precision", "recall")} for c in CLASSES},
                                   test_counts)
    per_class["cnn"].pop("macro avg")
    check = macro_f1(predictions["label"], predictions["final-logreg"])
    assert abs(check - final["final-logreg"]["macro_f1"][0]) < 1e-9, "saved predictions do not match final_test.json"

    counts = pd.crosstab(predictions["label"], predictions["final-logreg"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    angles = sorted(analysis["rotation_test"], key=int)
    shifts = sorted(analysis["shift_test"], key=int)
    fractions = sorted(analysis["learning_validation"], key=float)

    def curve(block, keys, name):
        return [r4(block[k][name]) for k in keys]

    model_block = lambda column: {"macro_f1": triple(final[column]["macro_f1"]), "accuracy": r4(final[column]["accuracy"][0]),
                                  **({"ece": triple(final[column]["ece"])} if "ece" in final[column] else {})}
    return {
        "validation": validation,
        "filter_check": {"kmeans": r4(filters["macro_f1"]["kmeans"]), "random": r4(filters["macro_f1"]["random"]),
                         "gap": triple(filters["random_minus_kmeans"]),
                         "folds_random_better": int(sum(r > k for r, k in zip(filters["macro_f1_by_fold"]["random"],
                                                                               filters["macro_f1_by_fold"]["kmeans"]))),
                         "share_of_draws": r4(filters["random_beats_kmeans_in_share_of_draws"]),
                         "per_class": filters["per_class_f1"]},
        "test": {
            "wafers": final["test_wafers"], "lots": final["test_lots"],
            "models": {"ours": model_block("final-logreg"), "svm": model_block("final-svm"),
                       "forest": model_block("B2-flat-random-forest"), "logistic": model_block("B2-flat-logistic"),
                       "cnn": {"macro_f1": triple(cnn["test_macro_f1"]["textbook CNN"]), "accuracy": r4(cnn["test_accuracy"]["textbook CNN"])}},
            "gaps": {"logistic": triple(final["final_logreg_minus"]["B2-flat-logistic"]),
                     "forest": triple(final["final_logreg_minus"]["B2-flat-random-forest"]),
                     "svm": triple(final["final_logreg_minus"]["final-svm"]), "cnn": triple(cnn["ours_minus_cnn"])},
            "per_class": per_class,
            "pr_auc": {c: triple(final["final-logreg"]["pr_auc"][c]) for c in CLASSES},
            "confusion_counts": counts.to_numpy().tolist(),
            "mistakes": analysis["mistakes_test"],
        },
        "rotation": {"angles": [int(a) for a in angles],
                     "ours": curve(analysis["rotation_test"], angles, "final-logreg"),
                     "forest": curve(analysis["rotation_test"], angles, "B2-flat-random-forest"),
                     "logistic": curve(analysis["rotation_test"], angles, "B2-flat-logistic"),
                     "cnn": curve(cnn["rotation_test"], angles, "textbook CNN")},
        "shift": {"pixels": [int(d) for d in shifts],
                  "ours": curve(analysis["shift_test"], shifts, "final-logreg"),
                  "forest": curve(analysis["shift_test"], shifts, "B2-flat-random-forest"),
                  "logistic": curve(analysis["shift_test"], shifts, "B2-flat-logistic"),
                  "cnn": curve(cnn["shift_test"], shifts, "textbook CNN")},
        "learning": {"fractions": [float(f) for f in fractions],
                     "ours": curve(analysis["learning_validation"], fractions, "final-logreg"),
                     "logistic": curve(analysis["learning_validation"], fractions, "B2-flat-logistic")},
        "cnn": {"learned_numbers": cnn["learned_numbers"], "epoch_picked": cnn["epochs_picked_on_validation"],
                "max_epochs": cnn["max_epochs"], "train_seconds": round(cnn["cnn_train_seconds_all_training_wafers"]),
                "validation_by_epoch": triple(cnn["validation_macro_f1_by_epoch"]),
                "learned_ratio": cnn["learned_numbers"]["textbook CNN"] / cnn["learned_numbers"]["ours"],
                "score_ratio": cnn["test_macro_f1"]["ours"][0] / cnn["test_macro_f1"]["textbook CNN"][0]},
        "denoise": {"gap": triple(denoise_gap),
                    "isolated": exact_table({c: {k: float(denoise_table.loc[c, ("isolated (primary)", k)]) for k in ("precision", "recall")}
                                             for c in CLASSES}, test_counts),
                    "median": exact_table({c: {k: float(denoise_table.loc[c, ("median (post-hoc)", k)]) for k in ("precision", "recall")}
                                           for c in CLASSES}, test_counts),
                    "median_accuracy": r4(log["P4-test-posthoc-median-logreg"]["accuracy"])},
        "real_world": real_world(counts, real),
    }


def real_world(counts: pd.DataFrame, logged: pd.DataFrame) -> dict:
    """Same replay as scripts/10_real_world_precision.py, at full precision (the CSV keeps 4 decimals)."""
    rates = counts.to_numpy() / counts.to_numpy().sum(axis=1, keepdims=True)
    recall = np.diag(rates)
    test_share = counts.to_numpy().sum(axis=1) / counts.to_numpy().sum()
    real_counts = pd.read_csv(EXP / "class_counts.csv").set_index("label")["count"]
    real_counts["none"] = json.loads((EXP / "audit_summary.json").read_text())["none_rows"]
    real_share = real_counts[CLASSES].to_numpy() / real_counts[CLASSES].sum()

    def precision_at(shares):
        expected = shares[:, None] * rates
        return np.diag(expected) / expected.sum(axis=0)

    table = {}
    for name, shares in (("on test", test_share), ("at real shares", real_share)):
        precision = precision_at(shares)
        f1 = 2 * precision * recall / (precision + recall)
        for i, c in enumerate(CLASSES):
            table.setdefault(c, {})[f"precision {name}"] = r4(precision[i])
            table[c][f"F1 {name}"] = r4(f1[i])
        table.setdefault("macro avg", {})[f"precision {name}"] = r4(precision.mean())
        table["macro avg"][f"F1 {name}"] = r4(f1.mean())
    check = logged.loc["macro avg", "F1 at real shares"]
    assert abs(table["macro avg"]["F1 at real shares"] - check) < 1e-4, "real-world replay disagrees with real_world_precision.csv"
    return {"test_share": [r4(v) for v in test_share], "real_share": [r4(v) for v in real_share], "table": table,
            "clean_flagged": r4(1 - counts.loc["none", "none"] / counts.loc["none"].sum())}


# ------------------------------------------------------------------- the demo model
def demo_model(split, fails, masks, extras, predictions):
    """Return (conv, logreg, info) for the in-browser demos."""
    saved = ROOT / "data" / "final_models.joblib"
    if saved.exists():
        bundle = joblib.load(saved)
        return bundle["conv"], bundle["models"]["final-logreg"], {"source": "saved", "agreement": 1.0}
    cache = ROOT / "data" / "site_demo_refit.joblib"   # gitignored; delete it to refit again
    if cache.exists():
        bundle = joblib.load(cache)
        return bundle["conv"], bundle["model"], {"source": "refit"}
    train = (split["split"] == "train").to_numpy()
    recipe = json.loads((EXP / "final_config.json").read_text())["recipe"]
    print("data/final_models.joblib not found: refitting the frozen recipe on the training lots", flush=True)
    conv = ConvFeatures(recipe["method"], recipe["k"], recipe["rotate"], recipe["pooling"], seed=SEED)
    conv.fit(fails[train], masks[train], split.loc[train, "label"].to_numpy())
    model = make_model("logreg").fit(np.hstack([conv.transform(fails[train], masks[train]), extras[train]]),
                                     split.loc[train, "label"].to_numpy())
    joblib.dump({"conv": conv, "model": model}, cache)
    return conv, model, {"source": "refit"}


def model_block(conv, model, info, data, split, fails, masks, extras, examples):
    recipe = json.loads((EXP / "final_config.json").read_text())["recipe"]
    train = (split["split"] == "train").to_numpy()
    scaler, logreg = model.model.steps[0][1], model.model.steps[1][1]
    assert list(model.classes_) == CLASSES
    rand = ConvFeatures("random", recipe["k"], recipe["rotate"], recipe["pooling"], seed=SEED)
    rand.fit(fails[train], masks[train], split.loc[train, "label"].to_numpy())
    checks = {}
    for source_index in examples:
        at = data.index[data["source_index"] == source_index][0]
        x = np.hstack([conv.transform(fails[at:at + 1], masks[at:at + 1]), extras[at:at + 1]])
        checks[str(source_index)] = {"features": [round(float(v), 5) for v in x[0]],
                                     "probs": [r4(v) for v in model.predict_proba(x)[0]]}
    return {
        **info, "recipe": recipe, "classes": CLASSES,
        "mean": [round(float(v), 6) for v in conv.mean], "zca": [[round(float(v), 6) for v in r] for r in conv.zca],
        "filters": [[round(float(v), 5) for v in r] for r in conv.filters],
        "random_filters": [[round(float(v), 5) for v in r] for r in rand.filters],
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_], "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [[round(float(v), 6) for v in r] for r in logreg.coef_], "intercept": [round(float(v), 6) for v in logreg.intercept_],
        "temperature": round(float(model.temperature), 6),
        "check": checks,
    }


# ----------------------------------------------------------- numbers into the HTML
def round_half(value: float, digits: int) -> float:
    """Round half away from zero, the same way assets/core.js does, so both agree."""
    scale = 10 ** digits
    out = math.copysign(math.floor(abs(value) * scale + 0.5 + 1e-6) / scale, value)
    return 0.0 if out == 0 else out


FORMATS = {
    "int": lambda v: f"{math.floor(v + 0.5):,}",
    "f3": lambda v: f"{round_half(v, 3):.3f}",
    "f2": lambda v: f"{round_half(v, 2):.2f}",
    "pct": lambda v: f"{round_half(v * 100, 0):.0f}%",
    "pct1": lambda v: f"{round_half(v * 100, 1):.1f}%",
    "sf3": lambda v: ("+" if round_half(v, 3) > 0 else "−" if round_half(v, 3) < 0 else "") + f"{abs(round_half(v, 3)):.3f}",
    "af2": lambda v: f"{round_half(abs(v), 2):.2f}",
    "af3": lambda v: f"{round_half(abs(v), 3):.3f}",
    "x": lambda v: f"{round_half(v, 0):.0f}×",
    "per100": lambda v: f"{round_half(v * 100, 0):.0f}",
    "inv": lambda v: f"{round_half(1 / v, 0):.0f}",
    "thousand": lambda v: f"{math.floor(v / 1000):,} thousand",
    "raw": str,
}


def lookup(results: dict, path: str):
    node = results
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def fill_html(results: dict | None = None) -> None:
    """Write every data-key number, source chip and commit link into site/index.html.

    The page works without JavaScript this way, and still no number is typed by hand:
    each one comes from `results` (the same data as site/data/results.js).
    """
    if results is None:
        results = json.loads((EXP / "site_stats.json").read_text())
    meta = results["meta"]
    page = ROOT / "site" / "index.html"
    html_text = page.read_text()
    missing = []

    def number(match):
        attrs, key = match["attrs"], match["key"]
        fmt = re.search(r'data-f="([^"]+)"', attrs)
        try:
            text = FORMATS[fmt[1] if fmt else "raw"](lookup(results, key))
        except (KeyError, IndexError, TypeError):
            missing.append(key)
            return match[0]
        return f"<{match['tag']}{attrs}>{html.escape(text)}</{match['tag']}>"

    html_text = re.sub(r'<(?P<tag>\w+)(?P<attrs>[^<>]*?\bdata-key="(?P<key>[^"]+)"[^<>]*)>[^<]*</(?P=tag)>', number, html_text)

    def chip(match):
        path = match["path"]
        ref = "main" if path in meta["new_files"] else meta["commit"]
        return f'<a class="src" data-src="{path}" href="{meta["repo"]}/blob/{ref}/{path}">'

    html_text = re.sub(r'<a class="src" data-src="(?P<path>[^"]+)"(?: href="[^"]*")?>', chip, html_text)
    commits = {"head": meta["commit"], "freeze": meta["freeze"]["commit"], "results": meta["results"]["commit"]}
    html_text = re.sub(r'<a data-commit="(\w+)" class="commit"(?: href="[^"]*")?>[^<]*</a>',
                       lambda m: f'<a data-commit="{m[1]}" class="commit" href="{meta["repo"]}/commit/{commits[m[1]]}">{commits[m[1]]}</a>',
                       html_text)
    html_text = re.sub(r'<a id="(repo-link|footer-repo)" class="ext"(?: href="[^"]*")?>',
                       lambda m: f'<a id="{m[1]}" class="ext" href="{meta["repo"]}">', html_text)
    page.write_text(html_text)
    if missing:
        raise KeyError(f"site/index.html asks for numbers that results.js does not have: {sorted(set(missing))}")
    print(f"Filled the numbers into {page.relative_to(ROOT)}")


# ---------------------------------------------------------------------------- main
def main() -> None:
    data = pd.read_pickle(ROOT / "data" / "project_subset.pkl").reset_index(drop=True)
    split = pd.read_csv(ROOT / "splits" / "project_splits.csv")
    assert (split["source_index"].to_numpy() == data["source_index"].to_numpy()).all()
    predictions = pd.read_csv(EXP / "final_test_predictions.csv")
    shapes = data["waferMap"].map(lambda m: np.asarray(m).shape)
    data["h"], data["w"] = shapes.map(lambda s: s[0]), shapes.map(lambda s: s[1])
    position = pd.Series(data.index, index=data["source_index"])
    is_train = (split["split"] == "train").to_numpy()

    recipe = json.loads((EXP / "final_config.json").read_text())["recipe"]
    print("Preparing all wafers", flush=True)
    fails, masks, extras = prepare(data["waferMap"], recipe["size"], recipe["denoise"])

    results = {"meta": {"repo": git("remote", "get-url", "origin").removesuffix(".git"), "commit": git("rev-parse", "--short", "HEAD"),
                        "new_files": NEW_FILES, "freeze": first_commit("experiments/final_config.json"),
                        "results": first_commit("experiments/final_test.json"), "classes": CLASSES},
               "recipe": recipe}
    results |= data_numbers(data, split, fails.astype(bool))
    results["summary_numbers"] = summary_numbers(data, split)
    results |= results_numbers(predictions)
    rot, shift = results["rotation"], results["shift"]
    quarter = [rot["angles"].index(a) for a in (90, 180, 270)]
    results["derived"] = {   # small differences the text quotes, worked out here so none is typed by hand
        "shift3_loss": {m: r4(shift[m][0] - shift[m][-1]) for m in ("ours", "forest", "logistic", "cnn")},
        "rot45_cost": r4(rot["ours"][0] - rot["ours"][rot["angles"].index(45)]),
        "worst_angle_cost": r4(rot["ours"][0] - min(rot["ours"])),
        "quarter_move": r4(max(abs(rot["ours"][k] - rot["ours"][0]) for k in quarter)),
        "labeled_vs_sample": r4(results["audit"]["labeled_rows"] / results["audit"]["subset_rows"]),
    }

    conv, model, info = demo_model(split, fails, masks, extras, predictions)

    # ---- wafers: full raw maps for the data story (training wafers only) --------------------------
    wafers: dict[str, dict] = {}

    def add_raw(source_index: int) -> str:
        at = position[source_index]
        assert is_train[at], f"{source_index} is a test wafer; the data story uses training wafers only"
        key = str(source_index)
        if key not in wafers:
            raw = np.asarray(data.loc[at, "waferMap"])
            mask, fail = (raw > 0).astype(np.uint8), (raw == 2).astype(np.uint8)
            iso, med = denoise(fail, mask, "isolated"), denoise(fail, mask, "median")
            wafers[key] = {"label": data.loc[at, "label"], "lot": data.loc[at, "lotName"], "h": int(raw.shape[0]), "w": int(raw.shape[1]),
                           "map": "".join(str(int(v)) for v in raw.ravel()),
                           "check": {"fail": int(fail.sum()), "isolated": int(iso.sum()), "median": int(med.sum()),
                                     "isolated32": weighted_sum(resize(iso, SIZE)), "mask32": weighted_sum(resize(mask, SIZE)),
                                     "median32": weighted_sum(resize(med, SIZE)), "none32": weighted_sum(resize(fail, SIZE))}}
        return key

    rng = np.random.default_rng(SEED)
    examples = {}
    for label in CLASSES:
        chosen = list(STORY_EXAMPLES[label])
        pool = data.index[is_train & (data["label"] == label).to_numpy() & (np.maximum(data["h"], data["w"]) <= 64).to_numpy()]
        for at in rng.permutation(pool):
            if len(chosen) >= EXAMPLES_PER_CLASS:
                break
            if int(data.loc[at, "source_index"]) not in chosen:
                chosen.append(int(data.loc[at, "source_index"]))
        examples[label] = [add_raw(i) for i in chosen]

    def lot_wafers(lot, label, n=6):
        rows = data.index[(data["lotName"] == lot) & (data["label"] == label)][:n]
        return [add_raw(int(data.loc[at, "source_index"])) for at in rows]

    same_lot = [{"lot": SAME_LOT["lot"], "label": SAME_LOT["label"], "wafers": lot_wafers(SAME_LOT["lot"], SAME_LOT["label"]),
                 "strangers": [lot_wafers(lot, SAME_LOT["label"], 1)[0] for lot in SAME_LOT["others"]]}]
    for lot, label in EXTRA_LOTS:
        h, w = data.loc[data["lotName"] == lot, ["h", "w"]].iloc[0]
        others = data[is_train & (data["label"] == label).to_numpy() & (data["lotName"] != lot).to_numpy()]
        for tolerance in (8, 16, 32, 400):      # widen until six different lots qualify
            pool = others[others["h"].between(h - tolerance, h + tolerance)
                          & others["w"].between(w - tolerance, w + tolerance)].groupby("lotName").head(1)
            if len(pool) >= 6:
                break
        picks = rng.choice(pool.index, 6, replace=False)
        same_lot.append({"lot": lot, "label": label, "wafers": lot_wafers(lot, label),
                         "strangers": [add_raw(int(data.loc[at, "source_index"])) for at in picks]})

    story = {"examples": examples, "scratches": [add_raw(i) for i in SCRATCH_DEMOS], "noise": add_raw(NOISE_DEMO),
             "small": add_raw(SIZE_EXTREMES[0]), "large": add_raw(SIZE_EXTREMES[1]), "same_lot": same_lot}

    # ---- packed 32x32 maps: split demo (training), rotation and error explorer (test) -------------
    def packed(at) -> str:
        return pack32(fails[at], masks[at])

    lot_sizes = split[is_train].groupby("lotName").size()
    demo_lots = rng.choice(lot_sizes.index[(lot_sizes >= 3) & (lot_sizes <= 6)], SPLIT_DEMO_LOTS, replace=False)
    split_demo = [{"lot": lot, "maps": [packed(at) for at in split.index[split["lotName"] == lot]],
                   "labels": split.loc[split["lotName"] == lot, "label"].tolist()} for lot in sorted(demo_lots)]

    test_rows = split.index[~is_train]
    saved_pred = predictions.set_index("source_index")["final-logreg"]
    print("Applying the demo model to the test wafers (no scores computed)", flush=True)
    x_test = np.hstack([conv.transform(fails[test_rows], masks[test_rows]), extras[test_rows]])
    probs_test = model.predict_proba(x_test)
    demo_pred = np.asarray(CLASSES)[probs_test.argmax(1)]
    saved_test = saved_pred.loc[split.loc[test_rows, "source_index"]].to_numpy()
    agrees = demo_pred == saved_test
    info["agreement"] = float(agrees.mean())
    truth = split.loc[test_rows, "label"].to_numpy()

    rotation = []
    for label in CLASSES:
        here = np.flatnonzero((truth == label) & agrees)
        right, wrong = here[saved_test[here] == label], here[saved_test[here] != label]
        picks = [rng.choice(right), rng.choice(wrong) if len(wrong) else rng.choice(right)]
        for k in picks[:ROTATION_DEMO_PER_CLASS]:
            at = test_rows[k]
            frames, probs = [], []
            for angle in results["rotation"]["angles"]:
                f = rotate_map(fails[at], angle)
                m = rotate_map(masks[at].astype(np.uint8), angle).astype(bool)
                frames.append(pack32(f, m))
                x = np.hstack([conv.transform(f[None], m[None]), extras[at:at + 1]])
                probs.append([round(float(v), 3) for v in model.predict_proba(x)[0]])
            rotation.append({"id": int(split.loc[at, "source_index"]), "lot": split.loc[at, "lotName"], "label": label,
                             "saved_prediction": saved_test[k], "frames": frames, "probs": probs})

    confusion = {}
    for i, t in enumerate(CLASSES):
        for j, p in enumerate(CLASSES):
            cell = np.flatnonzero((truth == t) & (saved_test == p) & agrees)
            if len(cell) == 0:
                continue
            chosen = rng.choice(cell, min(len(cell), CONFUSION_EXAMPLES), replace=False)
            confusion[f"{i},{j}"] = [{"id": int(split.loc[test_rows[k], "source_index"]), "lot": split.loc[test_rows[k], "lotName"],
                                      "map": packed(test_rows[k]), "confidence": round(float(probs_test[k, j]), 3)} for k in chosen]

    mistakes = [{**m, "lot": data.loc[position[m["source_index"]], "lotName"], "map": packed(position[m["source_index"]])}
                for m in results["test"]["mistakes"]]

    lot_index = {lot: n for n, lot in enumerate(predictions["lotName"].unique())}
    bootstrap = {"lot": [lot_index[l] for l in predictions["lotName"]],
                 "truth": [CLASSES.index(c) for c in predictions["label"]],
                 "ours": [CLASSES.index(c) for c in predictions["final-logreg"]], "lots": len(lot_index)}

    draw_mask = masks[position[STORY_EXAMPLES["none"][0]]]
    model_data = model_block(conv, model, info, data, split, fails, masks, extras, [STORY_EXAMPLES[c][0] for c in CLASSES])
    model_data["draw_mask"] = pack32(np.zeros_like(draw_mask), draw_mask)
    results["demo_model"] = {"source": info["source"], "agreement": r4(info["agreement"])}
    print(f"Demo model ({info['source']}) agrees with the saved final-test predictions on {info['agreement']:.2%} of test wafers")

    out = ROOT / "site" / "data"
    out.mkdir(parents=True, exist_ok=True)
    header = "// Written by scripts/11_site_data.py (python main.py site-data). Do not edit by hand.\n"

    def write(name, payload):
        (out / f"{name}.js").write_text(header + f"(window.SITE = window.SITE || {{}}).{name} = "
                                        + json.dumps(payload, separators=(",", ":")) + ";\n")

    write("results", results)
    write("wafers", {"raw": wafers, "story": story, "split_demo": split_demo, "rotation": rotation,
                     "confusion": confusion, "mistakes": mistakes, "bootstrap": bootstrap})
    write("model", model_data)
    (EXP / "site_stats.json").write_text(json.dumps(results, indent=2) + "\n")
    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB" for p in sorted(out.glob("*.js"))}
    print(f"Wrote {sizes}, {len(wafers)} raw maps, and experiments/site_stats.json")
    fill_html(results)


if __name__ == "__main__":
    # python -m scripts.11_site_data --fill-only : refresh the numbers in index.html from site_stats.json
    import sys
    fill_html() if "--fill-only" in sys.argv else main()

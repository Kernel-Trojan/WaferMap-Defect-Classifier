"""Leak checks for the frozen split. Run with: python main.py check

Our whole evaluation rests on one promise: the model never gets graded on a wafer
it (or a lot-mate, or an identical copy) saw during training. This script re-checks
that promise straight from splits/project_splits.csv, and exits with an error if
anything leaks. Run it any time the split or the subset might have changed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import wafer_hash

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def shared(left: pd.Series, right: pd.Series) -> set:
    """Values that show up on both sides."""
    return set(left) & set(right)


def main() -> None:
    split_path = ROOT / "splits" / "project_splits.csv"
    subset_path = ROOT / "data" / "project_subset.pkl"
    if not split_path.exists() or not subset_path.exists():
        raise FileNotFoundError("Run `python main.py audit` and `python main.py splits` first")
    split = pd.read_csv(split_path)
    subset = pd.read_pickle(subset_path).reset_index(drop=True)

    failures: list[str] = []
    passed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        if ok:
            passed += 1
        else:
            failures.append(name)

    # 1. The split file and the subset have to describe the same wafers, in the same order.
    same_rows = len(split) == len(subset) and (split["source_index"].to_numpy() == subset["source_index"].to_numpy()).all()
    check("split rows line up with the subset", bool(same_rows), f"{len(split):,} rows")
    if not same_rows:
        raise SystemExit("Split and subset don't match; the other checks would be meaningless.")
    check("labels line up with the subset", bool((split["label"].to_numpy() == subset["label"].astype(str).to_numpy()).all()))
    split["hash"] = subset["waferMap"].map(wafer_hash).to_numpy()

    # 2. Bookkeeping: test rows carry no fold, train rows carry folds 0-4.
    train, test = split[split["split"] == "train"], split[split["split"] == "test"]
    check("only 'train' and 'test' in the split column", set(split["split"]) == {"train", "test"})
    check("test rows have no fold number", bool((test["fold"] == -1).all() and (test["val_fold_random"] == -1).all()))
    check("every train row has folds 0-4",
          bool(train["fold"].between(0, 4).all() and train["val_fold_random"].between(0, 4).all()))
    fraction = len(test) / len(split)
    check("test set is about 20% of wafers", 0.18 <= fraction <= 0.22, f"{fraction:.1%}")

    # 3. The big one: nothing crosses from train to test.
    for column, meaning in (("lotName", "lot"), ("group_id", "group"), ("hash", "identical map")):
        count = len(shared(train[column], test[column]))
        check(f"no {meaning} on both train and test", count == 0, f"{count} shared")

    # 4. Same promise inside each validation fold, since we tune on these.
    for fold in range(5):
        valid, rest = train[train["fold"] == fold], train[train["fold"] != fold]
        lots, maps = len(shared(valid["lotName"], rest["lotName"])), len(shared(valid["hash"], rest["hash"]))
        check(f"fold {fold}: no lot or identical map shared with its training part", lots == 0 and maps == 0,
              f"{lots} lots, {maps} maps")

    # 5. Every class has to show up everywhere we score, or its F1 is undefined there.
    for name, rows in [("test", test)] + [(f"fold {f}", train[train["fold"] == f]) for f in range(5)]:
        missing = sorted(set(CLASSES) - set(rows["label"]))
        check(f"{name}: all 9 classes present", not missing, f"missing {missing}" if missing else "")

    # 6. Check the checker. The naive random folds ignore lots on purpose, so the
    #    exact same lot test MUST find leaks there. If it didn't, our "0 shared"
    #    results above could just mean the test is broken.
    valid, rest = train[train["val_fold_random"] == 0], train[train["val_fold_random"] != 0]
    leaked = len(shared(valid["lotName"], rest["lotName"]))
    check("checker catches a real leak (naive random fold 0 shares lots, as expected)", leaked > 0,
          f"{leaked:,} of {valid['lotName'].nunique():,} validation lots also in training")

    if failures:
        raise SystemExit(f"\n{len(failures)} check(s) FAILED: " + "; ".join(failures))
    print(f"\nAll {passed} checks passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def main():
    subset_path = ROOT / "data" / "project_subset.pkl"
    if not subset_path.exists():
        raise FileNotFoundError("Run python -m scripts.01_audit first to create data/project_subset.pkl")
    data = pd.read_pickle(subset_path).reset_index(drop=True)
    if data["lotName"].isna().any():
        raise ValueError("Rows with missing lotName cannot be safely grouped")

    n = len(data)
    uf = UnionFind(n)
    first_by_lot, first_by_hash = {}, {}
    for i, row in data.iterrows():
        lot = str(row["lotName"])
        previous = first_by_lot.setdefault(lot, i)
        uf.union(i, previous)
        wafer = np.ascontiguousarray(np.asarray(row["waferMap"], dtype=np.int16))
        digest = hashlib.sha1(repr(wafer.shape).encode("ascii") + wafer.tobytes()).hexdigest()
        previous = first_by_hash.setdefault(digest, i)
        uf.union(i, previous)
    groups = np.asarray([uf.find(i) for i in range(n)])
    labels = data["label"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(splitter.split(np.zeros(n), labels, groups))
    fold_assignment = np.full(n, -1, dtype=int)
    fold_maker = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(fold_maker.split(train_idx, labels[train_idx], groups[train_idx])):
        fold_assignment[train_idx[val_idx]] = fold

    random_assignment = np.full(n, -1, dtype=int)
    random_maker = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(random_maker.split(train_idx, labels[train_idx])):
        random_assignment[train_idx[val_idx]] = fold

    result = pd.DataFrame({
        "source_index": data["source_index"].astype("int64"),
        "label": labels,
        "lotName": data["lotName"].astype(str),
        "group_id": groups.astype(str),
        "split": np.where(np.isin(np.arange(n), test_idx), "test", "train"),
        "fold": fold_assignment,
        "val_fold_random": random_assignment,
    })
    train_groups = set(result.loc[result.split == "train", "group_id"])
    test_groups = set(result.loc[result.split == "test", "group_id"])
    assert not train_groups.intersection(test_groups)
    assert (result.loc[result.split == "test", "fold"] == -1).all()
    assert (result.loc[result.split == "test", "val_fold_random"] == -1).all()
    assert (result.loc[result.split == "train", "fold"] >= 0).all()

    out = ROOT / "splits"
    out.mkdir(exist_ok=True)
    result.to_csv(out / "project_splits.csv", index=False)
    summary = {
        "seed": SEED,
        "group_definition": "connected components of lotName and exact waferMap hash",
        "rows": n,
        "groups": int(pd.Series(groups).nunique()),
        "test_rows": int((result.split == "test").sum()),
        "test_fraction": float((result.split == "test").mean()),
        "test_lots": int(result.loc[result.split == "test", "lotName"].nunique()),
        "train_rows": int((result.split == "train").sum()),
        "train_lots": int(result.loc[result.split == "train", "lotName"].nunique()),
        "folds": {},
        "test_class_counts": {str(k): int(v) for k, v in result.loc[result.split == "test", "label"].value_counts().items()},
        "group_overlap_train_test": len(train_groups.intersection(test_groups)),
    }
    for fold in range(5):
        validation = result.loc[result.fold == fold]
        summary["folds"][str(fold)] = {
            "validation_rows": int(len(validation)),
            "validation_lots": int(validation.lotName.nunique()),
            "class_counts": {str(k): int(v) for k, v in validation.label.value_counts().items()},
        }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Frozen {len(result):,} rows: {summary['test_rows']:,} test rows "
          f"({summary['test_fraction']:.1%}), {summary['groups']:,} connected groups")
    print("Wrote splits/project_splits.csv and splits/split_summary.json")


if __name__ == "__main__":
    main()

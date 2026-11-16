from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

from src.data_loader import wafer_hash

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


def group_and_duplicate_stats(result: pd.DataFrame, digests: list[str]) -> dict:
    """Describe the groups and duplicate maps so we can see the split is sane.

    This only reads the finished split; it never changes who goes where.
    - Group sizes: one huge group would make the 20% test cut lumpy.
    - Duplicate maps: identical maps must sit on the same side, or the model
      gets graded on wafers it has literally seen.
    - Label conflicts: the same exact map with two different labels is label
      noise that no model can get fully right.
    """
    frame = result.assign(hash=digests)
    groups = frame.groupby("group_id")
    sizes = groups.size()
    lots_per_group = groups["lotName"].nunique()
    duplicates = frame[frame["hash"].duplicated(keep=False)]
    per_map = duplicates.groupby("hash").agg(lots=("lotName", "nunique"), labels=("label", "nunique"),
                                             sides=("split", "nunique"))
    conflicts = [
        {"source_index": sorted(int(i) for i in rows["source_index"]), "labels": sorted(set(rows["label"]))}
        for digest, rows in duplicates[duplicates["hash"].isin(per_map.index[per_map["labels"] > 1])].groupby("hash")
    ]
    return {
        "group_sizes": {"median": float(sizes.median()), "max": int(sizes.max()),
                        "single_wafer_groups": int((sizes == 1).sum())},
        "groups_joining_multiple_lots": int((lots_per_group > 1).sum()),
        "max_lots_in_one_group": int(lots_per_group.max()),
        "duplicate_maps": {
            "rows_sharing_a_map": int(len(duplicates)),
            "distinct_maps": int(len(per_map)),
            "maps_spanning_multiple_lots": int((per_map["lots"] > 1).sum()),
            "maps_split_across_train_test": int((per_map["sides"] > 1).sum()),
            "maps_with_conflicting_labels": len(conflicts),
            "label_conflicts": sorted(conflicts, key=lambda c: c["source_index"]),
        },
    }


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
    digests = []
    for i, row in data.iterrows():
        lot = str(row["lotName"])
        previous = first_by_lot.setdefault(lot, i)
        uf.union(i, previous)
        digest = wafer_hash(row["waferMap"])
        digests.append(digest)
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
        **group_and_duplicate_stats(result, digests),
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

"""What would our precision look like in a real fab's class mix?

Our test set is about 27% 'none', because we kept only 10,000 of the 147,431 clean
labeled wafers. In the full labeled data 'none' is about 85%. Recall doesn't care
about that mix (it only looks at wafers of one true class), but precision does:
more clean wafers means more chances for a false alarm.

So we take the test-set confusion rates (of the real Scratch wafers, what share did
we call Scratch, Loc, ...) and replay them at the real class shares. No model is
refit and nothing new is predicted: this only re-reads final_test_predictions.csv.

Run: python main.py real-world-precision
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def precision_at(shares, rates):
    """Precision per predicted class if true classes appear with these shares."""
    expected = shares[:, None] * rates  # expected[i, j] = share of wafers that are i and get called j
    return np.diag(expected) / expected.sum(axis=0)


def main():
    test = pd.read_csv(ROOT / "experiments" / "final_test_predictions.csv")
    counts = confusion_matrix(test["label"], test["final-logreg"], labels=CLASSES)
    rates = counts / counts.sum(axis=1, keepdims=True)  # row i: where the real class-i wafers ended up
    recall = np.diag(rates)

    test_shares = counts.sum(axis=1) / counts.sum()
    # Real shares among labeled WM-811K wafers: we kept every defect wafer, so their
    # counts are exact; 'none' has 147,431 labeled wafers (from the audit).
    real = pd.read_csv(ROOT / "experiments" / "class_counts.csv").set_index("label")["count"]
    real["none"] = json.loads((ROOT / "experiments" / "audit_summary.json").read_text())["none_rows"]
    real_shares = real[CLASSES].to_numpy() / real.sum()

    table = pd.DataFrame({"test share": test_shares, "real share": real_shares, "recall": recall,
                          "precision on test": precision_at(test_shares, rates),
                          "precision at real shares": precision_at(real_shares, rates)}, index=CLASSES)
    for name in ("on test", "at real shares"):
        p = table[f"precision {name}"]
        table[f"F1 {name}"] = 2 * p * recall / (p + recall)
    table.loc["macro avg"] = table.mean()
    table.loc["macro avg", ["test share", "real share"]] = 1.0

    table.round(4).to_csv(ROOT / "experiments" / "real_world_precision.csv")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()

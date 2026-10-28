from src.data_loader import first_label, load_data

def verify() -> dict[str, int]:
    frame = load_data()
    labels = frame["failureType"].map(first_label)
    counts = {
        "total_rows": len(frame),
        "labeled_rows": int(labels.notna().sum()),
        "defect_rows": int((labels.notna() & labels.ne("none")).sum()),
        "none_rows": int((labels == "none").sum()),
    }
    expected = {"total_rows": 811457, "labeled_rows": 172950,
                "defect_rows": 25519, "none_rows": 147431}
    print("WM-811K source row-count checks")
    for key, value in counts.items():
        print(f"{key}: {value:,} (expected {expected[key]:,})")
    if counts != expected:
        raise ValueError("Dataset row counts do not match the expected WM-811K counts")
    return counts

if __name__ == "__main__":
    verify()

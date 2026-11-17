import argparse
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="Wafer map project tools")
    parser.add_argument("action", choices=("verify", "audit", "splits", "check", "baselines", "pipeline",
                                           "tuning", "final-test", "analysis", "posthoc-median", "reference-cnn", "filter-check",
                                           "real-world-precision"))
    args = parser.parse_args()
    action = args.action
    if action == "verify":
        from scripts.data_verifier import verify
        verify()
    elif action == "audit":
        import_module("scripts.01_audit").main()
    elif action == "splits":
        import_module("scripts.02_freeze_splits").main()
    elif action == "check":
        import_module("scripts.check_splits").main()
    elif action == "baselines":
        import_module("scripts.03_baselines").main()
    elif action == "pipeline":
        import_module("scripts.03_pipeline").main()
    elif action == "tuning":
        import_module("scripts.04_tuning").main()
    elif action == "final-test":
        import_module("scripts.05_final_test").main()
    elif action == "analysis":
        import_module("scripts.06_analysis").main()
    elif action == "posthoc-median":
        import_module("scripts.07_posthoc_median_test").main()
    elif action == "reference-cnn":
        import_module("scripts.08_reference_cnn").main()
    elif action == "filter-check":
        import_module("scripts.09_filter_check").main()
    else:
        import_module("scripts.10_real_world_precision").main()


if __name__ == "__main__":
    main()

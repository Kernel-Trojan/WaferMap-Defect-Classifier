import argparse
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="Wafer map project tools")
    parser.add_argument("action", choices=("verify", "audit", "splits", "check", "baselines", "pipeline",
                                           "tuning", "phase4-dryrun"))
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
    else:
        import_module("scripts.phase4_dryrun").main()


if __name__ == "__main__":
    main()

import argparse
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="Wafer map project tools")
    parser.add_argument("action", choices=("verify", "audit", "splits", "baselines"))
    args = parser.parse_args()
    action = args.action
    if action == "verify":
        from scripts.data_verifier import verify
        verify()
    elif action == "audit":
        import_module("scripts.01_audit").main()
    elif action == "splits":
        import_module("scripts.02_freeze_splits").main()
    else:
        import_module("scripts.03_baselines").main()


if __name__ == "__main__":
    main()

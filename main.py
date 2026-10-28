import argparse
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="Wafer map project tools")
    parser.add_argument("action", choices=("verify", "audit", "splits"))
    action = parser.parse_args().action
    if action == "verify":
        from scripts.data_verifier import verify
        verify()
    elif action == "audit":
        import_module("scripts.01_audit").main()
    else:
        import_module("scripts.02_freeze_splits").main()


if __name__ == "__main__":
    main()

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "LSWMD.pkl"


class LegacyPandasUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        return super().find_class(module, name)


_cache = None


def load_data(path: str | Path | None = None, *, use_cache: bool = True):
    global _cache
    source = Path(path) if path is not None else DATA_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"WM-811K dataset not found at {source}. Place LSWMD.pkl in data/ "
            "or pass its path to load_data()."
        )
    if path is None and use_cache and _cache is not None:
        return _cache
    with source.open("rb") as stream:
        frame = LegacyPandasUnpickler(stream, encoding="latin1").load()
    if path is None and use_cache:
        _cache = frame
    return frame


def first_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if len(value) == 0:
            return None
        item = value[0]
        while isinstance(item, (list, tuple)) or hasattr(item, "ndim") and getattr(item, "ndim", 0) > 0:
            if len(item) == 0:
                return None
            item = item[0]
        if item is None:
            return None
        try:
            if item != item:
                return None
        except (TypeError, ValueError):
            pass
        return str(item)
    except (TypeError, IndexError, KeyError):
        return None

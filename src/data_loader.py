from __future__ import annotations
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

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


def wafer_hash(wafer_map: Any) -> str:
    """Fingerprint of a wafer map: two maps get the same hash only if every die matches.

    The shape goes into the hash too, so a 30x30 map and a 25x36 map with the same
    900 values never collide. The split script and the split checker both use this,
    so they always agree on what counts as a duplicate.
    """
    wafer = np.ascontiguousarray(np.asarray(wafer_map, dtype=np.int16))
    return hashlib.sha1(repr(wafer.shape).encode("ascii") + wafer.tobytes()).hexdigest()


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

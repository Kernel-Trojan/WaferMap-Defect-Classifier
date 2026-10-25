import pickle
from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data" / "LSWMD.pkl"

# We are renaming the module as pandas.indexes Module is deprecated and we can't use it in modern python. 
# It forces us to use python2 but using python2 will cause conflicts with other packages.
class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        return super().find_class(module, name)

_cache = None

# This function will be called to get data.
def load_data():
    global _cache
    if _cache is None:
        with open(DATA_PATH, "rb") as f:
            _cache = RenamingUnpickler(f, encoding="latin1").load()
    return _cache
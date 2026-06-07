"""
dataset.py  (crossformer)

Thin re-export of the shared WeatherDataset so that local imports work
when running scripts from src/model/crossformer/.
"""
import importlib, sys, os

_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))

# Load the *utils* dataset module under an unambiguous name to avoid
# shadowing the local dataset.py during import.
_spec = importlib.util.spec_from_file_location(
    "_weather_dataset_shared",
    os.path.join(_utils_path, "dataset.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

WeatherDataset = _mod.WeatherDataset
Dataset = WeatherDataset   # legacy alias
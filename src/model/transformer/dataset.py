"""
dataset.py  (transformer)
"""
import importlib, sys, os

_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils"))

_spec = importlib.util.spec_from_file_location(
    "_weather_dataset_shared",
    os.path.join(_utils_path, "dataset.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

WeatherDataset = _mod.WeatherDataset
Dataset = WeatherDataset
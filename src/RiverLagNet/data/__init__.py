"""Data structures and loaders for RiverLagNet."""

from .hydrowq_import import HydroWQChinaCatalog, HydroWQChinaSample
from .real_daily import load_real_daily_dataset, prepare_china_real_daily
from .schema import RiverGraph, TARGET_NAMES, TimeSeriesData

__all__ = [
    "HydroWQChinaCatalog",
    "HydroWQChinaSample",
    "load_real_daily_dataset",
    "prepare_china_real_daily",
    "RiverGraph",
    "TARGET_NAMES",
    "TimeSeriesData",
]

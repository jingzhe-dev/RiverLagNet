"""Data structures and loaders for RiverLagNet."""

from .hydrowq_import import HydroWQChinaCatalog, HydroWQChinaSample
from .contracted_real_daily import prepare_china_contracted_real_daily
from .feature_roles import FeatureRoles, resolve_feature_roles
from .real_daily import load_real_daily_dataset, prepare_china_real_daily
from .schema import LocalForecastShapeContract, RiverGraph, TARGET_NAMES, TimeSeriesData

__all__ = [
    "HydroWQChinaCatalog",
    "HydroWQChinaSample",
    "FeatureRoles",
    "LocalForecastShapeContract",
    "load_real_daily_dataset",
    "prepare_china_contracted_real_daily",
    "prepare_china_real_daily",
    "resolve_feature_roles",
    "RiverGraph",
    "TARGET_NAMES",
    "TimeSeriesData",
]

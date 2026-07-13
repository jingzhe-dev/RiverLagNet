"""Verified import and access helpers for the HydroWQ China sample bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor

from .schema import RiverGraph, TARGET_NAMES


MANIFEST_NAME = "sample-bundles-china-multibasin-v0.1.json"
DATASET_RELATIVE_PATH = Path("datasets") / "china-multibasin-v0.1"
METADATA_NAMES = ("sample-index.parquet", "normalization-train.json", "build-report.json")
EXCLUDED_CATEGORIES = ("raw", "cache", "logs", "runs")


@dataclass(frozen=True)
class HydroWQImportSummary:
    """Summary of a verified manifest-driven local data import."""

    source: Path
    destination: Path
    manifest_name: str
    manifest_sha256: str
    asset_count: int
    file_count: int
    total_bytes: int
    receipt_path: Path


@dataclass(frozen=True)
class HydroWQChinaSample:
    """One source-normalized sample in RiverLagNet target order."""

    sample_id: str
    basin_id: str
    x: Tensor
    x_mask: Tensor
    x_quality: Tensor | None
    y: Tensor
    y_mask: Tensor
    target_names: tuple[str, ...] = TARGET_NAMES
    source_normalized: bool = True


@dataclass(frozen=True)
class HydroWQCompatibility:
    """Compatibility between imported and requested temporal windows."""

    compatible: bool
    source_history: int
    source_forecast: int
    required_history: int
    required_forecast: int
    reasons: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_references(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for bundle in manifest.get("bundles", []):
        for key in ("graph_ref", "static_ref"):
            reference = bundle.get(key)
            if isinstance(reference, dict):
                yield reference
        for scale in bundle.get("scales", {}).values():
            if not isinstance(scale, dict):
                continue
            for key, reference in scale.items():
                if key.endswith("_ref") and isinstance(reference, dict):
                    yield reference


def _safe_source_path(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"asset path escapes data root: {relative_path}")
    return candidate


def _load_manifest(source_processed: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = source_processed / "_manifests" / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HydroWQ manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("bundles"):
        raise ValueError("HydroWQ manifest contains no sample bundles")
    if not manifest.get("data_root"):
        raise ValueError("HydroWQ manifest does not declare data_root")
    required = {"NH3N", "CODMn", "TP"}
    for bundle in manifest["bundles"]:
        daily = bundle.get("scales", {}).get("daily", {})
        if not required.issubset(set(daily.get("variables", []))):
            raise ValueError(
                f"sample {bundle.get('sample_id', '<unknown>')} lacks required targets"
            )
    return manifest_path, manifest


def import_hydrowq_china(
    source_processed: Path, destination: Path
) -> HydroWQImportSummary:
    """Copy the HydroWQ China manifest whitelist after verifying every asset hash.

    Raw rasters, archives, caches, logs, and run artifacts are intentionally not
    traversed. The copied sample values remain normalized with the source
    project's training-basin statistics and must not be normalized a second time.
    """
    source_processed = source_processed.expanduser().resolve()
    destination = destination.expanduser().resolve()
    manifest_path, manifest = _load_manifest(source_processed)
    source_data_root = source_processed / manifest["data_root"]

    assets: dict[str, tuple[Path, str]] = {}
    for reference in _asset_references(manifest):
        relative_path = reference.get("relative_path")
        expected_sha = reference.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
            raise ValueError("every imported asset reference needs relative_path and sha256")
        source_path = _safe_source_path(source_data_root, relative_path)
        previous = assets.get(relative_path)
        if previous is not None and previous[1] != expected_sha:
            raise ValueError(f"conflicting checksums for asset: {relative_path}")
        assets[relative_path] = (source_path, expected_sha)

    for relative_path, (source_path, expected_sha) in assets.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"referenced asset not found: {source_path}")
        if _sha256(source_path) != expected_sha:
            raise ValueError(f"checksum mismatch for asset: {relative_path}")

    metadata: list[tuple[Path, Path]] = []
    metadata_root = source_data_root / DATASET_RELATIVE_PATH
    for name in METADATA_NAMES:
        source_path = metadata_root / name
        if source_path.is_file():
            metadata.append((source_path, DATASET_RELATIVE_PATH / name))

    copied: list[Path] = []
    copied_manifest = destination / "_manifests" / MANIFEST_NAME
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, copied_manifest)
    copied.append(copied_manifest)

    copied_data_root = destination / manifest["data_root"]
    for relative_path, (source_path, expected_sha) in assets.items():
        target_path = copied_data_root / Path(relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        if _sha256(target_path) != expected_sha:
            raise OSError(f"copied asset failed checksum verification: {relative_path}")
        copied.append(target_path)

    for source_path, relative_path in metadata:
        target_path = copied_data_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        if _sha256(target_path) != _sha256(source_path):
            raise OSError(f"copied metadata failed checksum verification: {relative_path}")
        copied.append(target_path)

    receipt_path = destination / "_import_receipt.json"
    receipt = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_processed),
        "destination": str(destination),
        "manifest_name": MANIFEST_NAME,
        "manifest_sha256": _sha256(manifest_path),
        "asset_count": len(assets),
        "file_count": len(copied),
        "total_bytes": sum(path.stat().st_size for path in copied),
        "source_normalized": True,
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "compatibility_note": (
            "Source samples use 45 history days and 46 forecast days; they are not "
            "directly compatible with RiverLagNet's default 90-to-30 experiment."
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return HydroWQImportSummary(
        source=source_processed,
        destination=destination,
        manifest_name=MANIFEST_NAME,
        manifest_sha256=receipt["manifest_sha256"],
        asset_count=len(assets),
        file_count=len(copied),
        total_bytes=receipt["total_bytes"],
        receipt_path=receipt_path,
    )


class HydroWQChinaCatalog:
    """Read verified HydroWQ China bundles without altering source normalization."""

    def __init__(self, import_root: Path) -> None:
        self.import_root = import_root.expanduser().resolve()
        _, self.manifest = _load_manifest(self.import_root)
        self.data_root = self.import_root / self.manifest["data_root"]
        self._samples = {
            str(bundle["sample_id"]): bundle for bundle in self.manifest["bundles"]
        }
        self._basins: dict[str, dict[str, Any]] = {}
        for bundle in self.manifest["bundles"]:
            self._basins.setdefault(str(bundle["basin_id"]), bundle)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return sample identifiers in manifest order."""
        return tuple(self._samples)

    @property
    def basin_ids(self) -> tuple[str, ...]:
        """Return unique basin identifiers in manifest order."""
        return tuple(self._basins)

    def _asset_path(self, reference: dict[str, Any]) -> Path:
        relative_path = reference.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError("asset reference lacks relative_path")
        path = _safe_source_path(self.data_root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"imported asset not found: {path}")
        return path

    def _load_array(self, reference: dict[str, Any], default_key: str) -> np.ndarray:
        key = reference.get("metadata", {}).get("npz_key", default_key)
        with np.load(self._asset_path(reference), allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"NPZ asset does not contain key {key!r}")
            return np.asarray(archive[key]).copy()

    def load_graph(self, basin_id: str) -> RiverGraph:
        """Load a basin graph, retaining upstream-to-downstream edge orientation."""
        try:
            bundle = self._basins[basin_id]
        except KeyError as exc:
            raise KeyError(f"unknown HydroWQ basin: {basin_id}") from exc
        edge_index = self._load_array(bundle["graph_ref"], "edge_index")
        graph_path = self._asset_path(bundle["graph_ref"])
        with np.load(graph_path, allow_pickle=False) as archive:
            if "edge_attr" not in archive:
                raise KeyError("graph NPZ asset does not contain 'edge_attr'")
            edge_attr = np.asarray(archive["edge_attr"]).copy()
        static = self._load_array(bundle["static_ref"], "static_node")
        graph = RiverGraph(
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr).float(),
            static=torch.from_numpy(static).float(),
        )
        graph.validate()
        return graph

    def load_sample(self, sample_id: str) -> HydroWQChinaSample:
        """Load three water-quality targets in fixed NH3N, CODMn, TP order."""
        try:
            bundle = self._samples[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown HydroWQ sample: {sample_id}") from exc
        daily = bundle["scales"]["daily"]
        input_variables = list(daily["variables"])
        target_variables = list(daily.get("target_variables", input_variables))
        input_indices = [input_variables.index(name) for name in TARGET_NAMES]
        target_indices = [target_variables.index(name) for name in TARGET_NAMES]

        x = self._load_array(daily["x_obs_ref"], "x_obs")[..., input_indices]
        x_mask = self._load_array(daily["mask_obs_ref"], "mask_obs")[..., input_indices]
        y = self._load_array(daily["target_ref"], "target")[..., target_indices]
        y_mask = self._load_array(daily["target_mask_ref"], "target_mask")[
            ..., target_indices
        ]
        quality_ref = daily.get("quality_code_ref")
        quality = (
            self._load_array(quality_ref, "quality_code")[..., input_indices]
            if isinstance(quality_ref, dict)
            else None
        )
        return HydroWQChinaSample(
            sample_id=sample_id,
            basin_id=str(bundle["basin_id"]),
            x=torch.from_numpy(x).float(),
            x_mask=torch.from_numpy(x_mask).bool(),
            x_quality=torch.from_numpy(quality).float() if quality is not None else None,
            y=torch.from_numpy(y).float(),
            y_mask=torch.from_numpy(y_mask).bool(),
        )

    def compatibility(
        self, required_history: int = 90, required_forecast: int = 30
    ) -> HydroWQCompatibility:
        """Compare the uniform source window against a requested model contract."""
        windows = {
            (
                int(bundle["scales"]["daily"]["history_length"]),
                int(bundle["scales"]["daily"]["forecast_length"]),
            )
            for bundle in self.manifest["bundles"]
        }
        if len(windows) != 1:
            raise ValueError("HydroWQ bundles do not share one daily window contract")
        source_history, source_forecast = next(iter(windows))
        reasons: list[str] = []
        if source_history != required_history:
            reasons.append(
                f"history length is {source_history}, required {required_history}"
            )
        if source_forecast != required_forecast:
            reasons.append(
                f"forecast length is {source_forecast}, required {required_forecast}"
            )
        return HydroWQCompatibility(
            compatible=not reasons,
            source_history=source_history,
            source_forecast=source_forecast,
            required_history=required_history,
            required_forecast=required_forecast,
            reasons=tuple(reasons),
        )


def summary_as_dict(summary: HydroWQImportSummary) -> dict[str, object]:
    """Convert a summary to a JSON-safe dictionary for the command line."""
    result = asdict(summary)
    for key in ("source", "destination", "receipt_path"):
        result[key] = str(result[key])
    return result

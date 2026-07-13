"""Verified import and access helpers for the HydroWQ China sample bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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


def summary_as_dict(summary: HydroWQImportSummary) -> dict[str, object]:
    """Convert a summary to a JSON-safe dictionary for the command line."""
    result = asdict(summary)
    for key in ("source", "destination", "receipt_path"):
        result[key] = str(result[key])
    return result

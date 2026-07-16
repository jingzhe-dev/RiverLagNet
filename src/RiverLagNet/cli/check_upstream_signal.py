"""Evaluate the preregistered RiverLagNet v0.2 upstream signal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from RiverLagNet.analysis.upstream_signal_gate import (
    CONTROLS,
    evaluate_probe_bundle,
    render_signal_report_markdown,
    signal_report_as_dict,
)


def main() -> None:
    """Read frozen local-probe predictions and write paired gate reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--controls", nargs="+", default=list(CONTROLS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probe-bundle",
        type=Path,
        help="Override the probe_bundle path declared by the selected config.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = OmegaConf.load(args.checkpoint_config)
    configured_bundle = OmegaConf.select(config, "probe_bundle")
    bundle = args.probe_bundle or (
        Path(str(configured_bundle)) if configured_bundle is not None else None
    )
    if bundle is None:
        raise ValueError(
            "selected checkpoint config must declare probe_bundle or receive --probe-bundle"
        )
    if not bundle.is_file():
        raise FileNotFoundError(f"probe bundle not found: {bundle}")
    reports, decision, metadata = evaluate_probe_bundle(
        str(bundle), folds=args.folds, controls=args.controls, seed=args.seed
    )
    payload = signal_report_as_dict(reports, decision)
    payload["metadata"] = {
        **metadata,
        "checkpoint_config": str(args.checkpoint_config),
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(
        render_signal_report_markdown(payload), encoding="utf-8"
    )
    print(f"signal_gate_advance={decision.advance}")
    print(f"json={args.output}")
    print(f"markdown={markdown_path}")


if __name__ == "__main__":
    main()

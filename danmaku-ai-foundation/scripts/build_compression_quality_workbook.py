from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_image_compression import (
    CompressionResult,
    write_compression_quality_workbook,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build compression_quality_comparison.xlsx from a compression "
            "benchmark's incremental variant logs."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Compression run directory containing variants/*/comments.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    log_paths = sorted((run_dir / "variants").glob("*/comments.jsonl"))
    if not log_paths:
        raise FileNotFoundError(
            f"no incremental variant logs found under "
            f"{run_dir / 'variants'}"
        )

    field_names = {field.name for field in fields(CompressionResult)}
    results: list[CompressionResult] = []
    for log_path in log_paths:
        with log_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    record = json.loads(clean)
                    values = {
                        name: record[name]
                        for name in field_names
                    }
                    results.append(CompressionResult(**values))
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid result at {log_path}:{line_number}"
                    ) from exc

    if not results:
        raise ValueError(f"no compression results found under {run_dir}")

    variant_names = {path.parent.name for path in log_paths}
    ordered_variants = sorted(
        variant_names,
        key=lambda value: (
            value == "original",
            int(value) if value.isdigit() else float("inf"),
        ),
    )
    variant_order = {
        variant: index for index, variant in enumerate(ordered_variants)
    }
    results.sort(
        key=lambda result: (
            variant_order.get(result.variant, len(variant_order)),
            result.frame_index,
        )
    )
    workbook_path = write_compression_quality_workbook(
        results,
        run_dir,
    )
    print(f"[done] wrote {workbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

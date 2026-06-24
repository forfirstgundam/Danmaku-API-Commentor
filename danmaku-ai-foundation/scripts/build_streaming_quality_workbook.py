from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_streaming import (
    StreamingResult,
    write_streaming_quality_workbook,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build streaming_quality_comparison.xlsx from a streaming "
            "benchmark's incremental variant logs."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Streaming run directory containing variants/*/comments.jsonl.",
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

    field_names = {field.name for field in fields(StreamingResult)}
    results: list[StreamingResult] = []
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
                    results.append(StreamingResult(**values))
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid result at {log_path}:{line_number}"
                    ) from exc

    if not results:
        raise ValueError(f"no streaming results found under {run_dir}")

    ordered_variant_results = sorted(
        {
            result.variant: result
            for result in results
        }.values(),
        key=lambda result: (
            result.image_max_dimension > 0,
            result.image_max_dimension,
            result.streaming,
        ),
    )
    variants = [result.variant for result in ordered_variant_results]
    variant_order = {
        variant: index for index, variant in enumerate(variants)
    }
    results.sort(
        key=lambda result: (
            variant_order[result.variant],
            result.frame_index,
        )
    )
    workbook_path = write_streaming_quality_workbook(results, run_dir)
    print(f"[done] wrote {workbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

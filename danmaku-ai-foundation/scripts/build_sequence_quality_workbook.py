from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_danmaku_sequence import (
    SequenceResult,
    write_sequence_quality_workbook,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build comment_quality_comparison.xlsx from a sequence "
            "benchmark's incremental model logs."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Sequence run directory containing models/*/comments.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    log_paths = sorted((run_dir / "models").glob("*/comments.jsonl"))
    if not log_paths:
        raise FileNotFoundError(
            f"no incremental model logs found under {run_dir / 'models'}"
        )

    field_names = {field.name for field in fields(SequenceResult)}
    results: list[SequenceResult] = []
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
                    results.append(SequenceResult(**values))
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid result at {log_path}:{line_number}"
                    ) from exc

    if not results:
        raise ValueError(f"no sequence results found under {run_dir}")

    results.sort(
        key=lambda result: (
            result.provider,
            result.model,
            result.frame_index,
        )
    )
    workbook_path = write_sequence_quality_workbook(results, run_dir)
    print(f"[done] wrote {workbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

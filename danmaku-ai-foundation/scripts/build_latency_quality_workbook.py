from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_model_latency import (
    BenchmarkResult,
    write_quality_workbook,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build comment_quality_comparison.xlsx from a model-latency "
            "run's incremental latency_results.jsonl."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Latency run directory containing latency_results.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    results_path = run_dir / "latency_results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"results file not found: {results_path}")

    results: list[BenchmarkResult] = []
    with results_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean = line.strip()
            if not clean:
                continue
            try:
                results.append(BenchmarkResult(**json.loads(clean)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid result at {results_path}:{line_number}"
                ) from exc

    if not results:
        raise ValueError(f"no benchmark results found in {results_path}")

    workbook_path = write_quality_workbook(results, run_dir)
    print(f"[done] wrote {workbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

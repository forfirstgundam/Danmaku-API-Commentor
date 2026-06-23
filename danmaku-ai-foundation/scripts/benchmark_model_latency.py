from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from danmaku.api.llm_client import LLMClient
from danmaku.models import CaptureFrame


API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}


MODEL_PRESETS = {
    "core": [
        "gemini:gemini-2.5-flash-lite",
        "openai:gpt-5.4-mini",
        "anthropic:claude-haiku-4-5",
    ],
    "requested": [
        "deepinfra:Qwen/Qwen2.5-VL-7B-Instruct",
        "together:Qwen/Qwen2.5-VL-7B-Instruct",
        "mistral:pixtral-large-latest",
        "mistral:pixtral-12b-latest",
        "openai:gpt-5.4-nano",
        "openai:gpt-5.4-mini",
        "gemini:gemini-3.5-flash",
        "gemini:gemini-2.0-flash",
        "groq:llama-3.2-90b-vision-preview",
        "groq:llama-3.2-11b-vision-preview",
        "anthropic:claude-haiku-4-5",
        "xai:grok-3-mini",
    ],
    "current": [
        "together:Qwen/Qwen3.5-9B",
        "mistral:mistral-large-2512",
        "mistral:mistral-small-2506",
        "openai:gpt-5.4-nano",
        "openai:gpt-5.4-mini",
        "gemini:gemini-3.5-flash",
        "gemini:gemini-2.0-flash",
        "groq:meta-llama/llama-4-scout-17b-16e-instruct",
        "groq:qwen/qwen3.6-27b",
        "anthropic:claude-haiku-4-5",
        "xai:grok-4.3",
    ],
}


@dataclass(slots=True)
class BenchmarkResult:
    logged_at: str
    provider: str
    model: str
    image_path: str
    image_bytes: int
    run_index: int
    latency_sec: float
    ok: bool
    comments_count: int
    long_comments_count: int
    summary_chars: int
    error_message: str


def parse_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "models must use provider:model, for example openai:gpt-5-mini"
        )

    provider, model = value.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()

    if provider not in API_KEY_ENV:
        supported = ", ".join(sorted(API_KEY_ENV))
        raise argparse.ArgumentTypeError(f"provider must be one of: {supported}")

    if not model:
        raise argparse.ArgumentTypeError("model name cannot be empty")

    return provider, model


def collect_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        images = [Path(path) for path in args.image]
    else:
        image_dir = Path(args.image_dir)
        suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        images = [
            path
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in suffixes
        ]

    if args.limit:
        images = images[: args.limit]

    missing = [path for path in images if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"image path(s) not found:\n{missing_text}")

    if not images:
        raise FileNotFoundError("no benchmark images found")

    return images


def build_client(provider: str, model: str, max_output_tokens: int) -> LLMClient | None:
    api_key = os.getenv(API_KEY_ENV[provider], "").strip()

    if not api_key:
        print(f"[skip] {provider}:{model} missing {API_KEY_ENV[provider]}")
        return None

    return LLMClient(
        api_key=api_key,
        api_provider=provider,
        model_name=model,
        use_dummy_api=False,
        send_screenshot=True,
        image_max_dimension=0,
        image_jpeg_quality=95,
        max_output_tokens=max_output_tokens,
        save_api_images=False,
    )


def run_once(
    client: LLMClient,
    provider: str,
    model: str,
    image_path: Path,
    run_index: int,
) -> BenchmarkResult:
    frame = CaptureFrame(
        image_path=image_path,
        timestamp=time.time(),
        ocr_text="",
    )

    started = time.perf_counter()
    batch = client.generate_comments(
        frame=frame,
        previous_summary="",
        previous_comments=[],
        use_streaming=False,
        on_comment=None,
    )
    latency_sec = round(time.perf_counter() - started, 3)

    return BenchmarkResult(
        logged_at=datetime.now().isoformat(timespec="seconds"),
        provider=provider,
        model=model,
        image_path=str(image_path),
        image_bytes=image_path.stat().st_size,
        run_index=run_index,
        latency_sec=latency_sec,
        ok=not batch.is_error,
        comments_count=len(batch.comments),
        long_comments_count=len(batch.long_comments),
        summary_chars=len(batch.summary),
        error_message=batch.error_message,
    )


def write_outputs(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "latency_results.jsonl"
    csv_path = output_dir / "latency_results.csv"
    summary_path = output_dir / "latency_summary.csv"
    run_summary_path = output_dir / "run_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    fieldnames = list(asdict(results[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    grouped: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.provider, result.model), []).append(result)

    summary_rows: list[dict[str, object]] = []
    for (provider, model), model_results in grouped.items():
        successful = [result for result in model_results if result.ok]
        latencies = [
            result.latency_sec
            for result in successful
        ] or [result.latency_sec for result in model_results]
        summary_rows.append(
            {
                "provider": provider,
                "model": model,
                "attempts": len(model_results),
                "successes": len(successful),
                "avg_latency_sec": round(sum(latencies) / len(latencies), 3),
                "min_latency_sec": min(latencies),
                "max_latency_sec": max(latencies),
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "provider",
            "model",
            "attempts",
            "successes",
            "avg_latency_sec",
            "min_latency_sec",
            "max_latency_sec",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with run_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {run_summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark non-streaming full API response latency across models. "
            "Images are sent as original bytes without resizing or JPEG recompression."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        default=[],
        help="Provider/model pair, for example anthropic:claude-sonnet-4-5",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        default="requested",
        help=(
            "Model set to use when --model is not supplied. "
            "requested mirrors the candidate list; current uses newer documented IDs."
        ),
    )
    parser.add_argument(
        "--image",
        action="append",
        help="Specific image path. Can be passed more than once.",
    )
    parser.add_argument(
        "--image-dir",
        default="temp_captures",
        help="Directory of images to benchmark when --image is not supplied.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum images to use from --image-dir. Use 0 for all.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of calls per model/image pair.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output-dir",
        default="logs/latency_benchmarks",
        help="Base output directory.",
    )
    return parser


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    parser = build_parser()
    args = parser.parse_args()
    model_specs = args.model or [
        parse_model(value) for value in MODEL_PRESETS[args.preset]
    ]
    images = collect_images(args)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    results: list[BenchmarkResult] = []

    for provider, model in model_specs:
        client = build_client(provider, model, args.max_output_tokens)

        if client is None:
            continue

        for image_path in images:
            for run_index in range(1, args.repeat + 1):
                print(f"[run] {provider}:{model} {image_path} #{run_index}")
                result = run_once(client, provider, model, image_path, run_index)
                results.append(result)
                status = "ok" if result.ok else "error"
                print(f"[result] {status} {result.latency_sec}s")

    if not results:
        print("[done] no results; check API keys and image paths")
        return 1

    write_outputs(results, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

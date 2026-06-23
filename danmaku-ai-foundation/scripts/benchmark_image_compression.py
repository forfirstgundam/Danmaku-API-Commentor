from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from benchmark_danmaku_sequence import (
    ModelContext,
    append_jsonl,
    capture_timestamp,
    collect_images,
)
from benchmark_model_latency import percentile
from danmaku.api.llm_client import LLMClient
from danmaku.models import CaptureFrame


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_DIMENSIONS = [512, 768, 1024, 0]


@dataclass(slots=True)
class CompressionResult:
    logged_at: str
    variant: str
    image_max_dimension: int
    jpeg_quality: int
    frame_index: int
    capture_timestamp: float
    image_path: str
    source_width: int
    source_height: int
    source_bytes: int
    request_width: int
    request_height: int
    request_bytes: int
    request_size_ratio: float
    model: str
    latency_sec: float
    first_attempt_sec: float
    retry_used: bool
    retry_duration_sec: float | None
    ok: bool
    comments: list[str]
    long_comments: list[str]
    summary: str
    scene_change_detected: bool
    context_sent: str
    recent_comments_sent: list[str]
    summary_history: list[str]
    recent_comment_history: list[str]
    consecutive_api_failures: int
    error_message: str


def variant_name(max_dimension: int) -> str:
    return "original" if max_dimension <= 0 else str(max_dimension)


def measure_request_image(
    image_path: Path,
    max_dimension: int,
    jpeg_quality: int,
) -> tuple[int, int, int, int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        source_width, source_height = image.size
        if max_dimension <= 0:
            return (
                source_width,
                source_height,
                source_width,
                source_height,
                image_path.stat().st_size,
            )

        image = image.convert("RGB")
        image.thumbnail(
            (max_dimension, max_dimension),
            Image.Resampling.LANCZOS,
        )
        request_width, request_height = image.size
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=max(20, min(95, jpeg_quality)),
            optimize=True,
        )
        return (
            source_width,
            source_height,
            request_width,
            request_height,
            len(buffer.getvalue()),
        )


def build_client(
    model: str,
    max_dimension: int,
    jpeg_quality: int,
    max_output_tokens: int,
) -> LLMClient | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[skip] missing GEMINI_API_KEY")
        return None

    return LLMClient(
        api_key=api_key,
        api_provider="gemini",
        model_name=model,
        use_dummy_api=False,
        send_screenshot=True,
        image_max_dimension=max_dimension,
        image_jpeg_quality=jpeg_quality,
        max_output_tokens=max_output_tokens,
        save_api_images=False,
    )


def generate_frame(
    client: LLMClient,
    model: str,
    max_dimension: int,
    jpeg_quality: int,
    image_path: Path,
    frame_index: int,
    context: ModelContext,
) -> CompressionResult:
    frame = CaptureFrame(
        image_path=image_path,
        timestamp=capture_timestamp(image_path, frame_index),
        ocr_text="",
    )
    (
        source_width,
        source_height,
        request_width,
        request_height,
        request_bytes,
    ) = measure_request_image(image_path, max_dimension, jpeg_quality)
    source_bytes = image_path.stat().st_size
    context_sent = context.build_context_summary()
    recent_comments_sent = context.recent_comment_history[-12:]

    total_started = time.perf_counter()
    first_started = time.perf_counter()
    batch = client.generate_comments(
        frame=frame,
        previous_summary=context_sent,
        previous_comments=recent_comments_sent,
        use_streaming=False,
        on_comment=None,
    )
    first_attempt_sec = round(time.perf_counter() - first_started, 3)

    retry_used = False
    retry_duration_sec: float | None = None
    if batch.is_error:
        retry_used = True
        retry_started = time.perf_counter()
        batch = client.generate_comments(
            frame=frame,
            previous_summary=context_sent,
            previous_comments=recent_comments_sent,
            use_streaming=False,
            on_comment=None,
        )
        retry_duration_sec = round(time.perf_counter() - retry_started, 3)

    latency_sec = round(time.perf_counter() - total_started, 3)
    scene_changed = context.apply_batch(batch)

    return CompressionResult(
        logged_at=datetime.now().isoformat(timespec="seconds"),
        variant=variant_name(max_dimension),
        image_max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
        frame_index=frame_index,
        capture_timestamp=frame.timestamp,
        image_path=str(image_path),
        source_width=source_width,
        source_height=source_height,
        source_bytes=source_bytes,
        request_width=request_width,
        request_height=request_height,
        request_bytes=request_bytes,
        request_size_ratio=round(request_bytes / source_bytes, 4),
        model=model,
        latency_sec=latency_sec,
        first_attempt_sec=first_attempt_sec,
        retry_used=retry_used,
        retry_duration_sec=retry_duration_sec,
        ok=not batch.is_error,
        comments=batch.comments,
        long_comments=batch.long_comments,
        summary=batch.summary,
        scene_change_detected=scene_changed,
        context_sent=context_sent,
        recent_comments_sent=recent_comments_sent,
        summary_history=context.summary_history[-2:],
        recent_comment_history=context.recent_comment_history[-12:],
        consecutive_api_failures=context.consecutive_api_failures,
        error_message=batch.error_message,
    )


def write_variant_log(
    result: CompressionResult,
    variant_dir: Path,
    max_output_tokens: int,
) -> None:
    record = asdict(result)
    record.update(
        {
            "ocr_text": "",
            "is_error": not result.ok,
            "api_provider": "gemini",
            "send_screenshot_to_api": True,
            "api_max_output_tokens": max_output_tokens,
            "use_streaming_api": False,
            "timing": {
                "api_duration_sec": result.latency_sec,
                "first_attempt_sec": result.first_attempt_sec,
                "retry_used": result.retry_used,
                "retry_duration_sec": result.retry_duration_sec,
            },
        }
    )
    append_jsonl(variant_dir / "comments.jsonl", record)


def write_outputs(
    results: list[CompressionResult],
    output_dir: Path,
    image_count: int,
    max_output_tokens: int,
) -> None:
    frame_results_path = output_dir / "frame_results.csv"
    comments_path = output_dir / "all_comments.csv"
    comparison_path = output_dir / "compression_comparison.csv"
    quality_path = output_dir / "quality_review.csv"
    summary_path = output_dir / "run_summary.csv"

    fields = list(asdict(results[0]).keys())
    with frame_results_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for field in (
                "comments",
                "long_comments",
                "recent_comments_sent",
                "summary_history",
                "recent_comment_history",
            ):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)

    comment_rows: list[dict[str, object]] = []
    for result in results:
        comments = [
            ("short", index, comment)
            for index, comment in enumerate(result.comments, start=1)
        ]
        comments.extend(
            ("long", index, comment)
            for index, comment in enumerate(result.long_comments, start=1)
        )
        for comment_type, comment_index, comment in comments:
            comment_rows.append(
                {
                    "variant": result.variant,
                    "frame_index": result.frame_index,
                    "image_path": result.image_path,
                    "comment_type": comment_type,
                    "comment_index": comment_index,
                    "comment": comment,
                    "summary": result.summary,
                    "latency_sec": result.latency_sec,
                    "request_bytes": result.request_bytes,
                }
            )

    comment_fields = [
        "variant",
        "frame_index",
        "image_path",
        "comment_type",
        "comment_index",
        "comment",
        "summary",
        "latency_sec",
        "request_bytes",
    ]
    with comments_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=comment_fields)
        writer.writeheader()
        writer.writerows(comment_rows)

    quality_fields = [
        *comment_fields,
        "relevance_score",
        "naturalness_score",
        "context_score",
        "overall_score",
        "review_notes",
    ]
    with quality_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=quality_fields)
        writer.writeheader()
        for row in comment_rows:
            writer.writerow(row)

    variants = list(dict.fromkeys(result.variant for result in results))
    by_frame: dict[int, dict[str, CompressionResult]] = {}
    for result in results:
        by_frame.setdefault(result.frame_index, {})[result.variant] = result

    comparison_fields = ["frame_index", "image_path", *variants]
    with comparison_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=comparison_fields)
        writer.writeheader()
        for frame_index in sorted(by_frame):
            per_variant = by_frame[frame_index]
            first_result = next(iter(per_variant.values()))
            row: dict[str, object] = {
                "frame_index": frame_index,
                "image_path": first_result.image_path,
            }
            for variant in variants:
                result = per_variant.get(variant)
                if result is None:
                    row[variant] = "NO RESULT"
                elif not result.ok:
                    row[variant] = f"ERROR\n{result.error_message}"
                else:
                    sections = []
                    if result.comments:
                        sections.append(
                            "COMMENTS\n"
                            + "\n".join(f"- {item}" for item in result.comments)
                        )
                    if result.long_comments:
                        sections.append(
                            "LONG COMMENTS\n"
                            + "\n".join(
                                f"- {item}" for item in result.long_comments
                            )
                        )
                    sections.append(f"SUMMARY\n{result.summary or '(empty)'}")
                    row[variant] = "\n\n".join(sections)
            writer.writerow(row)

    grouped: dict[str, list[CompressionResult]] = {}
    for result in results:
        grouped.setdefault(result.variant, []).append(result)

    summary_rows: list[dict[str, object]] = []
    for variant, variant_results in grouped.items():
        successful = [result for result in variant_results if result.ok]
        latencies = sorted(result.latency_sec for result in successful)
        request_sizes = [result.request_bytes for result in variant_results]
        source_sizes = [result.source_bytes for result in variant_results]
        comments_generated = sum(
            len(result.comments) + len(result.long_comments)
            for result in successful
        )
        summary_rows.append(
            {
                "variant": variant,
                "image_max_dimension": variant_results[0].image_max_dimension,
                "jpeg_quality": variant_results[0].jpeg_quality,
                "frames_expected": image_count,
                "frames_attempted": len(variant_results),
                "successes": len(successful),
                "failures": len(variant_results) - len(successful),
                "success_rate": round(
                    len(successful) / len(variant_results),
                    3,
                ),
                "comments_generated": comments_generated,
                "avg_comments_per_success": (
                    round(comments_generated / len(successful), 3)
                    if successful
                    else ""
                ),
                "avg_latency_sec": (
                    round(statistics.fmean(latencies), 3) if latencies else ""
                ),
                "median_latency_sec": (
                    round(statistics.median(latencies), 3) if latencies else ""
                ),
                "p90_latency_sec": (
                    round(percentile(latencies, 0.9), 3) if latencies else ""
                ),
                "min_latency_sec": min(latencies) if latencies else "",
                "max_latency_sec": max(latencies) if latencies else "",
                "avg_source_kb": round(
                    statistics.fmean(source_sizes) / 1024,
                    1,
                ),
                "avg_request_kb": round(
                    statistics.fmean(request_sizes) / 1024,
                    1,
                ),
                "request_size_ratio": round(
                    sum(request_sizes) / sum(source_sizes),
                    4,
                ),
                "scene_change_count": sum(
                    result.scene_change_detected for result in successful
                ),
                "retry_used_count": sum(
                    result.retry_used for result in variant_results
                ),
                "streaming": False,
                "max_output_tokens": max_output_tokens,
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    readme = (
        "Gemini image-compression sequence benchmark\n\n"
        f"Model: {results[0].model}\n"
        f"Frames per variant: {image_count}\n"
        f"Variants: {', '.join(grouped)}\n"
        "Streaming: disabled\n"
        f"Maximum output tokens: {max_output_tokens}\n\n"
        "Each compression variant processes the same ordered image sequence "
        "with an isolated rolling summary and recent-comment history.\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"[done] wrote {frame_results_path}")
    print(f"[done] wrote {comments_path}")
    print(f"[done] wrote {comparison_path}")
    print(f"[done] wrote {quality_path}")
    print(f"[done] wrote {summary_path}")


def parse_dimension(value: str) -> int:
    if value.strip().lower() == "original":
        return 0
    dimension = int(value)
    if dimension <= 0:
        raise argparse.ArgumentTypeError(
            "dimension must be positive or 'original'"
        )
    return dimension


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the screenshot sequence through Gemini 3.1 Flash-Lite "
            "using independent image-compression settings."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dimension",
        action="append",
        type=parse_dimension,
        default=[],
        help=(
            "Maximum image dimension or 'original'. Can be repeated. "
            "Defaults to 512, 768, 1024, and original."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=72)
    parser.add_argument(
        "--image-dir",
        default="logs/latency_benchmarks/for_benchmark",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum ordered images per variant. Use 0 for all.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        default="logs/latency_benchmarks",
    )
    return parser


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    args = build_parser().parse_args()
    dimensions = args.dimension or DEFAULT_DIMENSIONS
    dimensions = list(dict.fromkeys(dimensions))
    images = collect_images(Path(args.image_dir), args.limit)
    run_id = datetime.now().strftime("compression_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[CompressionResult] = []

    print(
        f"[start] model={args.model} images={len(images)} "
        f"variants={','.join(variant_name(item) for item in dimensions)}"
    )
    print(
        f"[start] streaming=off jpeg_quality={args.jpeg_quality} "
        f"max_output_tokens={args.max_output_tokens}"
    )

    for max_dimension in dimensions:
        variant = variant_name(max_dimension)
        client = build_client(
            args.model,
            max_dimension,
            args.jpeg_quality,
            args.max_output_tokens,
        )
        if client is None:
            return 1

        context = ModelContext()
        variant_dir = output_dir / "variants" / variant
        print(f"[variant] starting {variant}")

        for frame_index, image_path in enumerate(images, start=1):
            print(
                f"[run] {variant} frame {frame_index}/{len(images)} "
                f"{image_path.name}"
            )
            result = generate_frame(
                client,
                args.model,
                max_dimension,
                args.jpeg_quality,
                image_path,
                frame_index,
                context,
            )
            results.append(result)
            write_variant_log(
                result,
                variant_dir,
                args.max_output_tokens,
            )
            status = "ok" if result.ok else "error"
            print(
                f"[result] {status} {result.latency_sec}s "
                f"request={round(result.request_bytes / 1024, 1)}KB "
                f"comments={len(result.comments) + len(result.long_comments)}"
            )

        print(f"[variant] finished {variant}")

    write_outputs(
        results,
        output_dir,
        len(images),
        args.max_output_tokens,
    )
    print(f"[done] compression benchmark: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from danmaku.models import CaptureFrame, CommentBatch


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_DIMENSIONS = [0, 768]
DEFAULT_STREAMING_MODES = [False, True]


@dataclass(slots=True)
class StreamingResult:
    logged_at: str
    variant: str
    image_max_dimension: int
    jpeg_quality: int
    streaming: bool
    frame_index: int
    capture_timestamp: float
    image_path: str
    source_bytes: int
    model: str
    latency_sec: float
    image_preparation_sec: float
    api_duration_sec: float
    response_parsing_sec: float
    end_to_end_sec: float
    first_visible_comment_sec: float | None
    streamed_comment_count: int
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


def dimension_name(max_dimension: int) -> str:
    return "original" if max_dimension <= 0 else str(max_dimension)


def variant_name(max_dimension: int, streaming: bool) -> str:
    mode = "stream_on" if streaming else "stream_off"
    return f"{dimension_name(max_dimension)}_{mode}"


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


def add_metrics(
    totals: dict[str, float],
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + float(value)


def call_model(
    client: LLMClient,
    frame: CaptureFrame,
    context_sent: str,
    recent_comments_sent: list[str],
    streaming: bool,
    overall_started: float,
    first_visible_holder: list[float | None],
    streamed_comments: list[str],
) -> CommentBatch:
    def on_comment(comment: str) -> None:
        if first_visible_holder[0] is None:
            first_visible_holder[0] = round(
                time.perf_counter() - overall_started,
                6,
            )
        streamed_comments.append(comment)

    return client.generate_comments(
        frame=frame,
        previous_summary=context_sent,
        previous_comments=recent_comments_sent,
        use_streaming=streaming,
        on_comment=on_comment if streaming else None,
    )


def generate_frame(
    client: LLMClient,
    model: str,
    max_dimension: int,
    jpeg_quality: int,
    streaming: bool,
    image_path: Path,
    frame_index: int,
    context: ModelContext,
    retry_count: int,
) -> StreamingResult:
    frame = CaptureFrame(
        image_path=image_path,
        timestamp=capture_timestamp(image_path, frame_index),
        ocr_text="",
    )
    context_sent = context.build_context_summary()
    recent_comments_sent = context.recent_comment_history[-12:]

    overall_started = time.perf_counter()
    first_visible_holder: list[float | None] = [None]
    streamed_comments: list[str] = []
    timing_totals: dict[str, float] = {}

    first_started = time.perf_counter()
    batch = call_model(
        client=client,
        frame=frame,
        context_sent=context_sent,
        recent_comments_sent=recent_comments_sent,
        streaming=streaming,
        overall_started=overall_started,
        first_visible_holder=first_visible_holder,
        streamed_comments=streamed_comments,
    )
    first_attempt_sec = round(time.perf_counter() - first_started, 3)
    add_metrics(timing_totals, client.last_call_metrics)

    retry_used = False
    retry_duration_sec: float | None = None

    retry_started_total: float | None = None
    for _ in range(retry_count):
        if not batch.is_error:
            break

        retry_used = True
        if retry_started_total is None:
            retry_started_total = time.perf_counter()

        batch = call_model(
            client=client,
            frame=frame,
            context_sent=context_sent,
            recent_comments_sent=recent_comments_sent,
            streaming=streaming,
            overall_started=overall_started,
            first_visible_holder=first_visible_holder,
            streamed_comments=streamed_comments,
        )
        add_metrics(timing_totals, client.last_call_metrics)

    if retry_started_total is not None:
        retry_duration_sec = round(
            time.perf_counter() - retry_started_total,
            3,
        )

    latency_sec = round(time.perf_counter() - overall_started, 3)
    scene_changed = context.apply_batch(batch)

    return StreamingResult(
        logged_at=datetime.now().isoformat(timespec="seconds"),
        variant=variant_name(max_dimension, streaming),
        image_max_dimension=max_dimension,
        jpeg_quality=jpeg_quality,
        streaming=streaming,
        frame_index=frame_index,
        capture_timestamp=frame.timestamp,
        image_path=str(image_path),
        source_bytes=image_path.stat().st_size,
        model=model,
        latency_sec=latency_sec,
        image_preparation_sec=round(
            timing_totals.get("image_preparation_sec", 0.0),
            6,
        ),
        api_duration_sec=round(
            timing_totals.get("api_duration_sec", 0.0),
            6,
        ),
        response_parsing_sec=round(
            timing_totals.get("response_parsing_sec", 0.0),
            6,
        ),
        end_to_end_sec=round(
            timing_totals.get("end_to_end_sec", latency_sec),
            6,
        ),
        first_visible_comment_sec=first_visible_holder[0],
        streamed_comment_count=len(streamed_comments),
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
    result: StreamingResult,
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
            "use_streaming_api": result.streaming,
            "timing": {
                "image_preparation_sec": result.image_preparation_sec,
                "api_duration_sec": result.api_duration_sec,
                "response_parsing_sec": result.response_parsing_sec,
                "end_to_end_sec": result.end_to_end_sec,
                "first_visible_comment_sec": result.first_visible_comment_sec,
                "streamed_comment_count": result.streamed_comment_count,
                "first_attempt_sec": result.first_attempt_sec,
                "retry_used": result.retry_used,
                "retry_duration_sec": result.retry_duration_sec,
            },
        }
    )
    append_jsonl(variant_dir / "comments.jsonl", record)


def format_streaming_comparison_cell(
    result: StreamingResult | None,
) -> str:
    if result is None:
        return "NO RESULT"
    if not result.ok:
        return f"ERROR\n{result.error_message or 'Unknown error'}"

    first_visible = (
        f"{result.first_visible_comment_sec}s"
        if result.first_visible_comment_sec is not None
        else "-"
    )
    sections = [
        "MODE\n"
        f"{'Streaming ON' if result.streaming else 'Streaming OFF'}\n"
        f"max dimension: "
        f"{'original' if result.image_max_dimension <= 0 else result.image_max_dimension}\n"
        f"JPEG quality: {result.jpeg_quality}",
        "TIMING\n"
        f"first visible comment: {first_visible}\n"
        f"full response: {result.latency_sec}s\n"
        f"image preparation: {result.image_preparation_sec}s\n"
        f"API duration: {result.api_duration_sec}s\n"
        f"end-to-end: {result.end_to_end_sec}s\n"
        f"streamed comments: {result.streamed_comment_count}",
    ]
    if result.comments:
        sections.append(
            "COMMENTS\n"
            + "\n".join(f"• {comment}" for comment in result.comments)
        )
    if result.long_comments:
        label = (
            "LONG COMMENT"
            if len(result.long_comments) == 1
            else "LONG COMMENTS"
        )
        sections.append(
            label
            + "\n"
            + "\n".join(
                f"• {comment}" for comment in result.long_comments
            )
        )
    sections.append(f"SUMMARY\n{result.summary or '(empty)'}")
    return "\n\n".join(sections)


def write_streaming_quality_workbook(
    results: list[StreamingResult],
    output_dir: Path,
) -> Path:
    try:
        import xlsxwriter
    except ImportError as exc:
        raise RuntimeError(
            "Excel export requires XlsxWriter. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    workbook_path = output_dir / "streaming_quality_comparison.xlsx"
    variants = list(dict.fromkeys(result.variant for result in results))
    frame_indices = sorted({result.frame_index for result in results})
    result_by_key = {
        (result.frame_index, result.variant): result
        for result in results
    }
    image_by_frame = {
        result.frame_index: result.image_path for result in results
    }

    workbook = xlsxwriter.Workbook(
        workbook_path,
        {"constant_memory": True},
    )
    worksheet = workbook.add_worksheet("Streaming Comparison")
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(1, 1)

    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#243B53",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )
    screenshot_header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#147D8F",
            "align": "center",
            "valign": "vcenter",
        }
    )
    screenshot_formats = [
        workbook.add_format({"bg_color": color})
        for color in ("#D9EEF2", "#CBE7EC")
    ]
    response_formats = [
        workbook.add_format(
            {
                "font_color": "#1F2933",
                "bg_color": color,
                "align": "left",
                "valign": "top",
                "text_wrap": True,
            }
        )
        for color in ("#F4F8FB", "#EAF1F5")
    ]
    error_formats = [
        workbook.add_format(
            {
                "font_color": "#991B1B",
                "bg_color": color,
                "align": "left",
                "valign": "top",
                "text_wrap": True,
            }
        )
        for color in ("#FEE2E2", "#FECACA")
    ]

    worksheet.write(0, 0, "Screenshot", screenshot_header_format)
    for column, variant in enumerate(variants, start=1):
        sample = next(
            result for result in results if result.variant == variant
        )
        dimension = (
            "Original"
            if sample.image_max_dimension <= 0
            else f"{sample.image_max_dimension}px"
        )
        mode = "Streaming ON" if sample.streaming else "Streaming OFF"
        worksheet.write(
            0,
            column,
            f"{dimension}\n{mode}\nJPEG {sample.jpeg_quality}",
            header_format,
        )

    for row, frame_index in enumerate(frame_indices, start=1):
        stripe = (row - 1) % 2
        path = Path(image_by_frame[frame_index])
        worksheet.write_blank(
            row,
            0,
            None,
            screenshot_formats[stripe],
        )
        if path.is_file():
            from PIL import Image

            with Image.open(path) as source:
                image_width, image_height = source.size
            scale = min(
                240 / max(1, image_width),
                150 / max(1, image_height),
            )
            worksheet.insert_image(
                row,
                0,
                str(path),
                {
                    "x_scale": scale,
                    "y_scale": scale,
                    "x_offset": 5,
                    "y_offset": 5,
                    "url": path.resolve().as_uri(),
                    "description": f"Frame {frame_index}: {path.name}",
                    "object_position": 1,
                },
            )
        else:
            worksheet.write(
                row,
                0,
                f"Frame {frame_index}\n{path.name}",
                screenshot_formats[stripe],
            )

        max_lines = 12
        for column, variant in enumerate(variants, start=1):
            result = result_by_key.get((frame_index, variant))
            text = format_streaming_comparison_cell(result)
            cell_format = (
                error_formats[stripe]
                if result is None or not result.ok
                else response_formats[stripe]
            )
            worksheet.write(row, column, text, cell_format)
            max_lines = max(max_lines, text.count("\n") + 1)

        worksheet.set_row(row, min(360, max(150, max_lines * 13)))

    worksheet.set_row(0, 58)
    worksheet.set_column(0, 0, 36)
    worksheet.set_column(1, len(variants), 52)
    worksheet.autofilter(
        0,
        0,
        len(frame_indices),
        len(variants),
    )
    workbook.close()
    return workbook_path


def write_outputs(
    results: list[StreamingResult],
    output_dir: Path,
    image_count: int,
    max_output_tokens: int,
) -> None:
    frame_results_path = output_dir / "frame_results.csv"
    comparison_path = output_dir / "streaming_comparison.csv"
    summary_path = output_dir / "run_summary.csv"
    quality_workbook_path = (
        output_dir / "streaming_quality_comparison.xlsx"
    )

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

    variants = list(dict.fromkeys(result.variant for result in results))
    by_frame: dict[int, dict[str, StreamingResult]] = {}
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
                    first_visible = (
                        f"{result.first_visible_comment_sec}s"
                        if result.first_visible_comment_sec is not None
                        else "-"
                    )
                    sections = [
                        "TIMING\n"
                        f"first visible comment: {first_visible}\n"
                        f"full response: {result.latency_sec}s\n"
                        f"image preparation: {result.image_preparation_sec}s\n"
                        f"API duration: {result.api_duration_sec}s\n"
                        f"end-to-end: {result.end_to_end_sec}s\n"
                        f"streamed comments: {result.streamed_comment_count}"
                    ]
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

    grouped: dict[str, list[StreamingResult]] = {}
    for result in results:
        grouped.setdefault(result.variant, []).append(result)

    summary_rows: list[dict[str, object]] = []
    for variant, variant_results in grouped.items():
        successful = [result for result in variant_results if result.ok]
        latencies = sorted(result.latency_sec for result in successful)
        first_visible_values = sorted(
            result.first_visible_comment_sec
            for result in successful
            if result.first_visible_comment_sec is not None
        )
        comments_generated = sum(
            len(result.comments) + len(result.long_comments)
            for result in successful
        )

        summary_rows.append(
            {
                "variant": variant,
                "image_max_dimension": (
                    variant_results[0].image_max_dimension
                ),
                "jpeg_quality": variant_results[0].jpeg_quality,
                "streaming": variant_results[0].streaming,
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
                "avg_full_response_sec": (
                    round(statistics.fmean(latencies), 3)
                    if latencies
                    else ""
                ),
                "median_full_response_sec": (
                    round(statistics.median(latencies), 3)
                    if latencies
                    else ""
                ),
                "p90_full_response_sec": (
                    round(percentile(latencies, 0.9), 3)
                    if latencies
                    else ""
                ),
                "min_full_response_sec": min(latencies) if latencies else "",
                "max_full_response_sec": max(latencies) if latencies else "",
                "avg_first_visible_comment_sec": (
                    round(statistics.fmean(first_visible_values), 3)
                    if first_visible_values
                    else ""
                ),
                "median_first_visible_comment_sec": (
                    round(statistics.median(first_visible_values), 3)
                    if first_visible_values
                    else ""
                ),
                "p90_first_visible_comment_sec": (
                    round(percentile(first_visible_values, 0.9), 3)
                    if first_visible_values
                    else ""
                ),
                "avg_image_preparation_sec": (
                    round(
                        statistics.fmean(
                            result.image_preparation_sec
                            for result in successful
                        ),
                        6,
                    )
                    if successful
                    else ""
                ),
                "avg_api_duration_sec": (
                    round(
                        statistics.fmean(
                            result.api_duration_sec for result in successful
                        ),
                        3,
                    )
                    if successful
                    else ""
                ),
                "avg_end_to_end_sec": (
                    round(
                        statistics.fmean(
                            result.end_to_end_sec for result in successful
                        ),
                        3,
                    )
                    if successful
                    else ""
                ),
                "avg_streamed_comment_count": (
                    round(
                        statistics.fmean(
                            result.streamed_comment_count
                            for result in successful
                        ),
                        3,
                    )
                    if successful
                    else ""
                ),
                "retry_used_count": sum(
                    result.retry_used for result in variant_results
                ),
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
        "Gemini streaming sequence benchmark\n\n"
        f"Model: {results[0].model}\n"
        f"Frames per variant: {image_count}\n"
        f"Variants: {', '.join(grouped)}\n"
        f"Maximum output tokens: {max_output_tokens}\n\n"
        "Each variant processes the same ordered screenshot sequence with "
        "an isolated rolling summary and recent-comment history. Streaming "
        "is compared with non-streaming at each selected image dimension.\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    write_streaming_quality_workbook(results, output_dir)

    print(f"[done] wrote {frame_results_path}")
    print(f"[done] wrote {comparison_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {quality_workbook_path}")


def parse_dimension(value: str) -> int:
    if value.strip().lower() == "original":
        return 0
    dimension = int(value)
    if dimension <= 0:
        raise argparse.ArgumentTypeError(
            "dimension must be positive or 'original'"
        )
    return dimension


def parse_streaming_mode(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        "streaming mode must be 'on' or 'off'"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an ordered screenshot sequence through Gemini and compare "
            "streaming versus non-streaming response latency."
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
            "Defaults to original and 768."
        ),
    )
    parser.add_argument(
        "--streaming",
        action="append",
        type=parse_streaming_mode,
        default=[],
        help=(
            "Streaming mode: on or off. Can be repeated. "
            "Defaults to both off and on."
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
        "--retry-count",
        type=int,
        default=1,
        help="Retries after an error, using the same streaming mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/latency_benchmarks",
    )
    return parser


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    args = build_parser().parse_args()
    dimensions = list(dict.fromkeys(args.dimension or DEFAULT_DIMENSIONS))
    streaming_modes = list(
        dict.fromkeys(args.streaming or DEFAULT_STREAMING_MODES)
    )
    images = collect_images(Path(args.image_dir), args.limit)

    run_id = datetime.now().strftime("streaming_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[StreamingResult] = []

    variants = [
        (dimension, streaming)
        for dimension in dimensions
        for streaming in streaming_modes
    ]

    print(
        f"[start] model={args.model} images={len(images)} "
        f"variants={','.join(variant_name(*item) for item in variants)}"
    )
    print(
        f"[start] jpeg_quality={args.jpeg_quality} "
        f"max_output_tokens={args.max_output_tokens} "
        f"retry_count={args.retry_count}"
    )

    for variant_index, (max_dimension, streaming) in enumerate(
        variants,
        start=1,
    ):
        variant = variant_name(max_dimension, streaming)
        client = build_client(
            model=args.model,
            max_dimension=max_dimension,
            jpeg_quality=args.jpeg_quality,
            max_output_tokens=args.max_output_tokens,
        )
        if client is None:
            return 1

        context = ModelContext()
        variant_dir = output_dir / "variants" / variant
        print(
            f"[variant] starting {variant_index}/{len(variants)} "
            f"{variant}"
        )

        for frame_index, image_path in enumerate(images, start=1):
            print(
                f"[run] variant {variant_index}/{len(variants)} "
                f"{variant} frame {frame_index}/{len(images)} "
                f"{image_path.name}"
            )
            result = generate_frame(
                client=client,
                model=args.model,
                max_dimension=max_dimension,
                jpeg_quality=args.jpeg_quality,
                streaming=streaming,
                image_path=image_path,
                frame_index=frame_index,
                context=context,
                retry_count=max(0, args.retry_count),
            )
            results.append(result)
            write_variant_log(
                result=result,
                variant_dir=variant_dir,
                max_output_tokens=args.max_output_tokens,
            )
            status = "ok" if result.ok else "error"
            first_visible = (
                f"{result.first_visible_comment_sec}s"
                if result.first_visible_comment_sec is not None
                else "-"
            )
            print(
                f"[result] {status} first={first_visible} "
                f"full={result.latency_sec}s "
                f"api={result.api_duration_sec}s "
                f"streamed={result.streamed_comment_count} "
                f"comments={len(result.comments) + len(result.long_comments)}"
            )

        print(
            f"[variant] finished {variant_index}/{len(variants)} "
            f"{variant}"
        )

    write_outputs(
        results=results,
        output_dir=output_dir,
        image_count=len(images),
        max_output_tokens=args.max_output_tokens,
    )
    print(f"[done] streaming benchmark: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

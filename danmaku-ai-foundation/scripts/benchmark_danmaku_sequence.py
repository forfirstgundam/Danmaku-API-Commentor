from __future__ import annotations

import argparse
import csv
import json
import re
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

from benchmark_model_latency import (
    MODEL_PRESETS,
    build_client,
    parse_model,
    percentile,
)
from danmaku.models import CaptureFrame, CommentBatch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CAPTURE_TIME_PATTERN = re.compile(r"_(\d{8})_(\d{6})_(\d{3})$")


@dataclass(slots=True)
class SequenceResult:
    logged_at: str
    frame_index: int
    capture_timestamp: float
    image_path: str
    image_bytes: int
    provider: str
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


class ModelContext:
    def __init__(self) -> None:
        self.previous_summary = ""
        self.summary_history: list[str] = []
        self.recent_comment_history: list[str] = []
        self.consecutive_api_failures = 0

    def build_context_summary(self) -> str:
        parts: list[str] = []
        if self.previous_summary:
            parts.append(f"Overall context so far:\n{self.previous_summary}")

        recent = self.summary_history[-2:]
        if recent:
            recent_text = "\n".join(
                f"{index + 1}. {summary}"
                for index, summary in enumerate(recent)
            )
            parts.append(
                "Recent scene history, oldest to newest:\n"
                f"{recent_text}"
            )

        return "\n\n".join(parts)

    def apply_batch(self, batch: CommentBatch) -> bool:
        if batch.is_error:
            self.consecutive_api_failures += 1
            return False

        self.consecutive_api_failures = 0
        scene_changed = False
        if batch.summary:
            clean_summary = batch.summary.strip()
            scene_changed = clean_summary.startswith("[SCENE_CHANGE]")
            if scene_changed:
                clean_summary = clean_summary.removeprefix(
                    "[SCENE_CHANGE]"
                ).strip()
                self.summary_history.clear()
                self.previous_summary = ""

            self.summary_history.append(clean_summary)
            self.summary_history = self.summary_history[-4:]
            joined = " ".join(self.summary_history[-4:])
            self.previous_summary = (
                joined if len(joined) <= 600 else joined[-600:]
            )

        new_comments = [
            comment.strip()
            for comment in [*batch.comments, *batch.long_comments]
            if comment.strip()
        ]
        self.recent_comment_history.extend(new_comments)
        self.recent_comment_history = self.recent_comment_history[-24:]
        return scene_changed


def capture_timestamp(path: Path, fallback_index: int) -> float:
    match = CAPTURE_TIME_PATTERN.search(path.stem)
    if not match:
        return path.stat().st_mtime + fallback_index

    date_text, time_text, milliseconds = match.groups()
    captured_at = datetime.strptime(
        f"{date_text}{time_text}",
        "%Y%m%d%H%M%S",
    )
    return captured_at.timestamp() + int(milliseconds) / 1000


def collect_images(image_dir: Path, limit: int) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"no benchmark images found in {image_dir}")
    return images


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_frame(
    client: object,
    provider: str,
    model: str,
    image_path: Path,
    frame_index: int,
    context: ModelContext,
) -> SequenceResult:
    frame = CaptureFrame(
        image_path=image_path,
        timestamp=capture_timestamp(image_path, frame_index),
        ocr_text="",
    )
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

    return SequenceResult(
        logged_at=datetime.now().isoformat(timespec="seconds"),
        frame_index=frame_index,
        capture_timestamp=frame.timestamp,
        image_path=str(image_path),
        image_bytes=image_path.stat().st_size,
        provider=provider,
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


def write_model_log(
    result: SequenceResult,
    model_dir: Path,
    max_output_tokens: int,
) -> None:
    record = asdict(result)
    record.update(
        {
            "ocr_text": "",
            "is_error": not result.ok,
            "used_dummy_api": False,
            "fallback_model": "",
            "send_screenshot_to_api": True,
            "api_image_max_dimension": 0,
            "api_image_jpeg_quality": 95,
            "api_max_output_tokens": max_output_tokens,
            "use_streaming_api": False,
            "save_api_images": False,
            "timing": {
                "api_duration_sec": result.latency_sec,
                "comment_after_capture_sec": result.latency_sec,
                "total_worker_duration_sec": result.latency_sec,
                "first_streamed_comment_sec": None,
                "streamed_comment_count": 0,
                "retry_used": result.retry_used,
                "retry_duration_sec": result.retry_duration_sec,
            },
        }
    )
    append_jsonl(model_dir / "comments.jsonl", record)


def write_aggregate_outputs(
    results: list[SequenceResult],
    output_dir: Path,
    max_output_tokens: int,
    image_count: int,
) -> None:
    results_path = output_dir / "frame_results.csv"
    comments_path = output_dir / "all_comments.csv"
    comparison_path = output_dir / "comment_comparison.csv"
    quality_path = output_dir / "quality_review.csv"
    summary_path = output_dir / "run_summary.csv"

    result_fieldnames = list(asdict(results[0]).keys())
    with results_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=result_fieldnames)
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
                    "provider": result.provider,
                    "model": result.model,
                    "frame_index": result.frame_index,
                    "image_path": result.image_path,
                    "comment_type": comment_type,
                    "comment_index": comment_index,
                    "comment": comment,
                    "summary": result.summary,
                    "latency_sec": result.latency_sec,
                }
            )

    comment_fields = [
        "provider",
        "model",
        "frame_index",
        "image_path",
        "comment_type",
        "comment_index",
        "comment",
        "summary",
        "latency_sec",
    ]
    with comments_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=comment_fields)
        writer.writeheader()
        writer.writerows(comment_rows)

    model_keys = list(group_results_by_model(results))
    frame_results: dict[int, dict[tuple[str, str], SequenceResult]] = {}
    for result in results:
        frame_results.setdefault(result.frame_index, {})[
            (result.provider, result.model)
        ] = result

    comparison_fields = [
        "frame_index",
        "image_path",
        *[f"{provider}:{model}" for provider, model in model_keys],
    ]
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=comparison_fields)
        writer.writeheader()
        for frame_index in sorted(frame_results):
            per_model = frame_results[frame_index]
            first_result = next(iter(per_model.values()))
            row: dict[str, object] = {
                "frame_index": frame_index,
                "image_path": first_result.image_path,
            }
            for provider, model in model_keys:
                result = per_model.get((provider, model))
                cell_name = f"{provider}:{model}"
                if result is None:
                    row[cell_name] = "NO RESULT"
                elif not result.ok:
                    row[cell_name] = f"ERROR\n{result.error_message}"
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
                    row[cell_name] = "\n\n".join(sections)
            writer.writerow(row)

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

    grouped: dict[tuple[str, str], list[SequenceResult]] = {}
    for result in results:
        grouped.setdefault((result.provider, result.model), []).append(result)

    summary_rows: list[dict[str, object]] = []
    for (provider, model), model_results in grouped.items():
        successful = [result for result in model_results if result.ok]
        latencies = sorted(result.latency_sec for result in successful)
        comments_generated = sum(
            len(result.comments) + len(result.long_comments)
            for result in successful
        )
        summary_rows.append(
            {
                "provider": provider,
                "model": model,
                "frames_expected": image_count,
                "frames_attempted": len(model_results),
                "successes": len(successful),
                "failures": len(model_results) - len(successful),
                "success_rate": round(
                    len(successful) / len(model_results),
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
                "scene_change_count": sum(
                    result.scene_change_detected for result in successful
                ),
                "retry_used_count": sum(
                    result.retry_used for result in model_results
                ),
                "image_max_dimension": 0,
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
        "Danmaku sequence benchmark\n\n"
        f"Frames: {image_count}\n"
        f"Models attempted: {len(grouped)}\n"
        "Streaming: disabled\n"
        "Image compression/resizing: disabled\n"
        f"Maximum output tokens: {max_output_tokens}\n\n"
        "Each model processes the same ordered image sequence with isolated "
        "rolling summary and recent-comment context. quality_review.csv is "
        "ready for manual 1-5 scoring.\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"[done] wrote {results_path}")
    print(f"[done] wrote {comments_path}")
    print(f"[done] wrote {comparison_path}")
    print(f"[done] wrote {quality_path}")
    print(f"[done] wrote {summary_path}")


def group_results_by_model(
    results: list[SequenceResult],
) -> dict[tuple[str, str], list[SequenceResult]]:
    grouped: dict[tuple[str, str], list[SequenceResult]] = {}
    for result in results:
        grouped.setdefault((result.provider, result.model), []).append(result)
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an ordered screenshot sequence through each model using "
            "the normal app's rolling context behavior."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        default=[],
        help="Provider/model pair. Can be supplied more than once.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        default="current",
    )
    parser.add_argument(
        "--image-dir",
        default="logs/latency_benchmarks/for_benchmark",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of ordered images. Use 0 for all images.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
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
    images = collect_images(Path(args.image_dir), args.limit)
    model_specs = args.model or [
        parse_model(value) for value in MODEL_PRESETS[args.preset]
    ]
    run_id = datetime.now().strftime("sequence_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SequenceResult] = []

    print(
        f"[start] {len(images)} ordered images, "
        f"{len(model_specs)} configured models"
    )
    print("[start] streaming=off image_compression=off")
    print(f"[start] max_output_tokens={args.max_output_tokens}")

    for provider, model in model_specs:
        client = build_client(provider, model, args.max_output_tokens)
        if client is None:
            continue

        model_context = ModelContext()
        model_dir = output_dir / "models" / (
            f"{safe_name(provider)}__{safe_name(model)}"
        )
        print(f"[model] starting {provider}:{model}")

        for frame_index, image_path in enumerate(images, start=1):
            print(
                f"[run] {provider}:{model} "
                f"frame {frame_index}/{len(images)} {image_path.name}"
            )
            result = generate_frame(
                client,
                provider,
                model,
                image_path,
                frame_index,
                model_context,
            )
            results.append(result)
            write_model_log(
                result,
                model_dir,
                args.max_output_tokens,
            )
            status = "ok" if result.ok else "error"
            print(
                f"[result] {status} {result.latency_sec}s "
                f"comments={len(result.comments) + len(result.long_comments)}"
            )

        print(f"[model] finished {provider}:{model}")

    if not results:
        print("[done] no results; check API keys and image paths")
        return 1

    write_aggregate_outputs(
        results,
        output_dir,
        args.max_output_tokens,
        len(images),
    )
    print(f"[done] sequence benchmark: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

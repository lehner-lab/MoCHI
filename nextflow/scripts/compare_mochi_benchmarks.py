#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def parse_time_log(path: Path) -> tuple[str, int | None]:
    wall_clock = "unknown"
    max_rss_kb: int | None = None

    for line in path.read_text().splitlines():
        if "Elapsed (wall clock) time" in line:
            if ": " in line:
                wall_clock = line.rsplit(": ", 1)[1].strip()
        elif "Maximum resident set size" in line:
            match = re.search(r"(\d+)$", line)
            if match:
                max_rss_kb = int(match.group(1))

    return wall_clock, max_rss_kb


def parse_peak_gpu_memory_mb(path: Path) -> int | None:
    peak_memory: int | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("timestamp=") or line.endswith(":"):
            continue
        if not re.match(r"^\d{4}/\d{2}/\d{2}", line):
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue

        try:
            memory_used = int(parts[4])
        except ValueError:
            continue

        if peak_memory is None or memory_used > peak_memory:
            peak_memory = memory_used

    return peak_memory


def format_gb_from_kb(value_kb: int | None) -> str:
    if value_kb is None:
        return "unknown"
    return f"{value_kb / 1024 / 1024:.2f}"


def format_gb_from_mb(value_mb: int | None) -> str:
    if value_mb is None:
        return "unknown"
    return f"{value_mb / 1024:.2f}"


def build_rows(manifest_paths: list[Path]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for manifest_path in manifest_paths:
        manifest = parse_env_file(manifest_path)
        batch_size = int(manifest["BATCH_SIZE"])
        status = parse_env_file(Path(manifest["STATUS_FILE"]))
        wall_clock, max_rss_kb = parse_time_log(Path(manifest["TIME_LOG"]))
        peak_gpu_mb = parse_peak_gpu_memory_mb(Path(manifest["RESOURCE_LOG"]))

        rows.append(
            {
                "batch_size": batch_size,
                "output_dir": manifest["OUTPUT_DIR"],
                "exit_code": int(manifest["EXIT_CODE"]),
                "elapsed_seconds": int(manifest["ELAPSED_SECONDS"]),
                "wall_clock": wall_clock,
                "max_rss_kb": max_rss_kb if max_rss_kb is not None else -1,
                "peak_gpu_mb": peak_gpu_mb if peak_gpu_mb is not None else -1,
                "status_elapsed_seconds": int(status.get("elapsed_seconds", manifest["ELAPSED_SECONDS"])),
            }
        )

    rows.sort(key=lambda row: int(row["batch_size"]))
    return rows


def write_markdown(rows: list[dict[str, str | int]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("# MoCHI Benchmark Summary\n\nNo runs were provided.\n")
        return

    lines = [
        "# MoCHI Benchmark Summary",
        "",
        "| Batch size | Exit code | Elapsed seconds | Wall clock | Peak GPU memory (GiB) | Max RSS (GiB) | Output dir |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {batch_size} | {exit_code} | {elapsed_seconds} | {wall_clock} | {peak_gpu} | {max_rss} | `{output_dir}` |".format(
                batch_size=row["batch_size"],
                exit_code=row["exit_code"],
                elapsed_seconds=row["elapsed_seconds"],
                wall_clock=row["wall_clock"],
                peak_gpu=format_gb_from_mb(None if int(row["peak_gpu_mb"]) < 0 else int(row["peak_gpu_mb"])),
                max_rss=format_gb_from_kb(None if int(row["max_rss_kb"]) < 0 else int(row["max_rss_kb"])),
                output_dir=row["output_dir"],
            )
        )

    successful_rows = [row for row in rows if int(row["exit_code"]) == 0]
    if len(successful_rows) >= 2:
        fastest = min(successful_rows, key=lambda row: int(row["elapsed_seconds"]))
        slowest = max(successful_rows, key=lambda row: int(row["elapsed_seconds"]))
        delta = int(slowest["elapsed_seconds"]) - int(fastest["elapsed_seconds"])
        speedup = (
            float(slowest["elapsed_seconds"]) / float(fastest["elapsed_seconds"])
            if int(fastest["elapsed_seconds"]) > 0
            else 0.0
        )
        lines.extend(
            [
                "",
                f"Fastest run: `batch_size={fastest['batch_size']}`.",
                f"Elapsed time difference: `{delta}` seconds.",
                f"Relative speedup: `{speedup:.2f}x` versus the slower successful run.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n")


def write_tsv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "batch_size",
                "exit_code",
                "elapsed_seconds",
                "wall_clock",
                "peak_gpu_memory_gib",
                "max_rss_gib",
                "output_dir",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["batch_size"],
                    row["exit_code"],
                    row["elapsed_seconds"],
                    row["wall_clock"],
                    format_gb_from_mb(None if int(row["peak_gpu_mb"]) < 0 else int(row["peak_gpu_mb"])),
                    format_gb_from_kb(None if int(row["max_rss_kb"]) < 0 else int(row["max_rss_kb"])),
                    row["output_dir"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed MoCHI benchmark runs.")
    parser.add_argument("manifests", nargs="+", help="Paths to benchmark manifest env files.")
    parser.add_argument(
        "--markdown-output",
        default="benchmark_summary.md",
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--tsv-output",
        default="benchmark_summary.tsv",
        help="Output TSV file path.",
    )
    args = parser.parse_args()

    manifest_paths = [Path(path).resolve() for path in args.manifests]
    rows = build_rows(manifest_paths)
    write_markdown(rows, Path(args.markdown_output))
    write_tsv(rows, Path(args.tsv_output))


if __name__ == "__main__":
    main()

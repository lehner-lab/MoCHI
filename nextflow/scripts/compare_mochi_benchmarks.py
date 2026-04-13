#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class PhaseResult:
    run_label: str
    run_path: Path
    phase_name: str
    output_dir: str
    exit_code: int | None
    elapsed_seconds: int | None
    wall_clock: str
    max_rss_kb: int | None
    peak_gpu_mb: int | None
    start_time: datetime | None
    end_time: datetime | None


@dataclass
class RunSummary:
    label: str
    input_path: Path
    phases: list[PhaseResult]


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def parse_time_log(path: Path | None) -> tuple[str, int | None, int | None]:
    if path is None or not path.exists():
        return "unknown", None, None

    wall_clock = "unknown"
    max_rss_kb: int | None = None
    exit_code: int | None = None

    for line in path.read_text().splitlines():
        if "Elapsed (wall clock) time" in line and ": " in line:
            wall_clock = line.rsplit(": ", 1)[1].strip()
        elif "Maximum resident set size" in line:
            match = re.search(r"(\d+)$", line)
            if match:
                max_rss_kb = int(match.group(1))
        elif "Exit status" in line:
            match = re.search(r"(\d+)$", line)
            if match:
                exit_code = int(match.group(1))

    return wall_clock, max_rss_kb, exit_code


def parse_peak_gpu_memory_mb(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None

    if path.suffix.lower() == ".csv":
        peak_memory: int | None = None
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_value = row.get("gpu_mem_used_mib") or row.get("tracked_proc_gpu_mem_mib")
                if not raw_value:
                    continue
                try:
                    memory_used = int(float(raw_value))
                except ValueError:
                    continue
                if peak_memory is None or memory_used > peak_memory:
                    peak_memory = memory_used
        return peak_memory

    peak_memory = None
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


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_wall_clock_seconds(value: str) -> int | None:
    if value == "unknown":
        return None
    match = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+)", value)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def format_gb_from_kb(value_kb: int | None) -> str:
    if value_kb is None:
        return "unknown"
    return f"{value_kb / 1024 / 1024:.2f}"


def format_gb_from_mb(value_mb: int | None) -> str:
    if value_mb is None:
        return "unknown"
    return f"{value_mb / 1024:.2f}"


def format_seconds(value: int | None) -> str:
    if value is None:
        return "unknown"
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def resolve_local_file(base_dir: Path, preferred_names: list[str], raw_path: str | None) -> Path | None:
    for name in preferred_names:
        candidate = base_dir / name
        if candidate.exists():
            return candidate

    if not raw_path:
        return None

    raw = Path(raw_path).expanduser()
    if raw.exists():
        return raw

    candidate = base_dir / raw.name
    if candidate.exists():
        return candidate

    return None


def is_phase_dir(path: Path) -> bool:
    return (path / "benchmark_manifest.env").exists() or (path / "run_info.env").exists()


def phase_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name == "grid_search":
        return (0, name)
    if name.startswith("fold_"):
        fold_number = parse_int(name.split("_", 1)[1])
        return (1 if fold_number is not None else 2, f"{fold_number:03d}" if fold_number is not None else name)
    if name == "merge":
        return (3, name)
    return (4, name)


def collect_phase_dirs(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name not in {"benchmark_manifest.env", "run_info.env"}:
            raise ValueError(f"Unsupported file input: {resolved}")
        return [resolved.parent]

    if is_phase_dir(resolved):
        return [resolved]

    direct_children = sorted(
        [child for child in resolved.iterdir() if child.is_dir() and is_phase_dir(child)],
        key=phase_sort_key,
    )
    if direct_children:
        return direct_children

    recursive_dirs = sorted(
        {
            candidate.parent
            for candidate in resolved.glob("**/benchmark_manifest.env")
            if "work" not in candidate.parts and "uv-cache" not in candidate.parts
        }
        | {
            candidate.parent
            for candidate in resolved.glob("**/run_info.env")
            if "work" not in candidate.parts and "uv-cache" not in candidate.parts
        },
        key=phase_sort_key,
    )
    if recursive_dirs:
        return recursive_dirs

    raise ValueError(f"Could not resolve any run phases from {resolved}")


def load_phase_result(run_label: str, run_path: Path, phase_dir: Path) -> PhaseResult:
    benchmark_manifest_path = phase_dir / "benchmark_manifest.env"
    run_info_path = phase_dir / "run_info.env"
    manifest = parse_env_file(benchmark_manifest_path if benchmark_manifest_path.exists() else run_info_path)
    status = parse_env_file(phase_dir / "job_status.txt")

    time_log = resolve_local_file(phase_dir, ["time.log"], manifest.get("TIME_LOG"))
    resource_log = resolve_local_file(
        phase_dir,
        ["resource_snapshots.log", "resource_snapshots.csv"],
        manifest.get("RESOURCE_LOG"),
    )
    wall_clock, max_rss_kb, exit_code_from_time = parse_time_log(time_log)

    exit_code = (
        parse_int(manifest.get("EXIT_CODE"))
        if manifest.get("EXIT_CODE") is not None
        else parse_int(status.get("exit_code"))
    )
    if exit_code is None:
        exit_code = exit_code_from_time

    elapsed_seconds = (
        parse_int(manifest.get("ELAPSED_SECONDS"))
        if manifest.get("ELAPSED_SECONDS") is not None
        else parse_int(status.get("elapsed_seconds"))
    )
    if elapsed_seconds is None:
        elapsed_seconds = parse_wall_clock_seconds(wall_clock)

    start_time = parse_datetime(manifest.get("START_TIME"))
    end_time = parse_datetime(status.get("end_time"))
    if end_time is None and start_time is not None and elapsed_seconds is not None:
        end_time = start_time + timedelta(seconds=elapsed_seconds)

    return PhaseResult(
        run_label=run_label,
        run_path=run_path,
        phase_name=phase_dir.name,
        output_dir=str(phase_dir),
        exit_code=exit_code,
        elapsed_seconds=elapsed_seconds,
        wall_clock=wall_clock,
        max_rss_kb=max_rss_kb,
        peak_gpu_mb=parse_peak_gpu_memory_mb(resource_log),
        start_time=start_time,
        end_time=end_time,
    )


def build_run_summaries(paths: list[Path], labels: list[str] | None) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for index, path in enumerate(paths):
        label = labels[index] if labels is not None else path.name
        phase_dirs = collect_phase_dirs(path)
        phases = [load_phase_result(label, path, phase_dir) for phase_dir in phase_dirs]
        summaries.append(RunSummary(label=label, input_path=path, phases=phases))
    return summaries


def run_makespan_seconds(summary: RunSummary) -> int | None:
    starts = [phase.start_time for phase in summary.phases if phase.start_time is not None]
    ends = [phase.end_time for phase in summary.phases if phase.end_time is not None]
    if starts and ends:
        return int((max(ends) - min(starts)).total_seconds())
    elapsed_values = [phase.elapsed_seconds for phase in summary.phases if phase.elapsed_seconds is not None]
    if elapsed_values:
        return max(elapsed_values)
    return None


def sum_phase_seconds(summary: RunSummary) -> int | None:
    elapsed_values = [phase.elapsed_seconds for phase in summary.phases if phase.elapsed_seconds is not None]
    if not elapsed_values:
        return None
    return sum(elapsed_values)


def longest_phase(summary: RunSummary) -> PhaseResult | None:
    known_elapsed = [phase for phase in summary.phases if phase.elapsed_seconds is not None]
    if not known_elapsed:
        return None
    return max(known_elapsed, key=lambda phase: int(phase.elapsed_seconds or 0))


def count_successes(summary: RunSummary) -> int:
    return sum(1 for phase in summary.phases if phase.exit_code == 0)


def write_markdown(summaries: list[RunSummary], output_path: Path) -> None:
    if not summaries:
        output_path.write_text("# MoCHI Benchmark Summary\n\nNo runs were provided.\n")
        return

    lines = [
        "# MoCHI Run Comparison",
        "",
        "| Run | Phases | Successful phases | Makespan | Summed phase time | Longest phase | Peak GPU memory (GiB) | Max RSS (GiB) | Input path |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]

    for summary in summaries:
        longest = longest_phase(summary)
        peak_gpu_mb = max((phase.peak_gpu_mb for phase in summary.phases if phase.peak_gpu_mb is not None), default=None)
        max_rss_kb = max((phase.max_rss_kb for phase in summary.phases if phase.max_rss_kb is not None), default=None)
        longest_text = "unknown"
        if longest is not None:
            longest_text = f"{longest.phase_name} ({format_seconds(longest.elapsed_seconds)})"

        lines.append(
            "| {label} | {phase_count} | {successes}/{phase_count} | {makespan} | {sum_elapsed} | {longest} | {peak_gpu} | {max_rss} | `{input_path}` |".format(
                label=summary.label,
                phase_count=len(summary.phases),
                successes=count_successes(summary),
                makespan=format_seconds(run_makespan_seconds(summary)),
                sum_elapsed=format_seconds(sum_phase_seconds(summary)),
                longest=longest_text,
                peak_gpu=format_gb_from_mb(peak_gpu_mb),
                max_rss=format_gb_from_kb(max_rss_kb),
                input_path=summary.input_path,
            )
        )

    if len(summaries) == 2:
        left, right = summaries
        left_makespan = run_makespan_seconds(left)
        right_makespan = run_makespan_seconds(right)
        left_sum = sum_phase_seconds(left)
        right_sum = sum_phase_seconds(right)
        lines.append("")
        if left_makespan is not None and right_makespan is not None and min(left_makespan, right_makespan) > 0:
            faster, slower = (left, right) if left_makespan <= right_makespan else (right, left)
            speedup = max(left_makespan, right_makespan) / min(left_makespan, right_makespan)
            lines.append(
                f"Wall-clock completion: `{faster.label}` finished in `{format_seconds(min(left_makespan, right_makespan))}`, "
                f"which is `{speedup:.2f}x` faster than `{slower.label}` at `{format_seconds(max(left_makespan, right_makespan))}`."
            )
        if left_sum is not None and right_sum is not None:
            heavier, lighter = (left, right) if left_sum >= right_sum else (right, left)
            ratio = max(left_sum, right_sum) / min(left_sum, right_sum) if min(left_sum, right_sum) > 0 else 0.0
            lines.append(
                f"Aggregate phase time: `{heavier.label}` used `{format_seconds(max(left_sum, right_sum))}` of summed runtime across phases "
                f"versus `{format_seconds(min(left_sum, right_sum))}` for `{lighter.label}` (`{ratio:.2f}x`)."
            )

    for summary in summaries:
        lines.extend(
            [
                "",
                f"## {summary.label}",
                "",
                "| Phase | Exit code | Elapsed seconds | Wall clock | Peak GPU memory (GiB) | Max RSS (GiB) | Output dir |",
                "| --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for phase in summary.phases:
            lines.append(
                "| {phase_name} | {exit_code} | {elapsed_seconds} | {wall_clock} | {peak_gpu} | {max_rss} | `{output_dir}` |".format(
                    phase_name=phase.phase_name,
                    exit_code=phase.exit_code if phase.exit_code is not None else "unknown",
                    elapsed_seconds=phase.elapsed_seconds if phase.elapsed_seconds is not None else "unknown",
                    wall_clock=phase.wall_clock,
                    peak_gpu=format_gb_from_mb(phase.peak_gpu_mb),
                    max_rss=format_gb_from_kb(phase.max_rss_kb),
                    output_dir=phase.output_dir,
                )
            )

    output_path.write_text("\n".join(lines) + "\n")


def write_tsv(summaries: list[RunSummary], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "run_label",
                "run_path",
                "phase_name",
                "exit_code",
                "elapsed_seconds",
                "wall_clock",
                "peak_gpu_memory_gib",
                "max_rss_gib",
                "output_dir",
            ]
        )
        for summary in summaries:
            for phase in summary.phases:
                writer.writerow(
                    [
                        summary.label,
                        summary.input_path,
                        phase.phase_name,
                        phase.exit_code if phase.exit_code is not None else "unknown",
                        phase.elapsed_seconds if phase.elapsed_seconds is not None else "unknown",
                        phase.wall_clock,
                        format_gb_from_mb(phase.peak_gpu_mb),
                        format_gb_from_kb(phase.max_rss_kb),
                        phase.output_dir,
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed MoCHI runs and run phases.")
    parser.add_argument("inputs", nargs="+", help="Run directories or manifest env files.")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for the provided inputs, in the same order.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path).resolve() for path in args.inputs]
    labels = args.labels
    if labels is not None and len(labels) != len(input_paths):
        raise SystemExit("--labels must provide exactly one label for each input path")

    summaries = build_run_summaries(input_paths, labels)
    write_markdown(summaries, Path(args.markdown_output))
    write_tsv(summaries, Path(args.tsv_output))


if __name__ == "__main__":
    main()

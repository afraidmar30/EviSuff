#!/usr/bin/env python3
"""Run RAES rollouts over prebuilt shards with bounded process parallelism."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", default="raes-bench/raes-bench-v12_process/shards_1k")
    parser.add_argument("--shard-glob", default="public_questions_1k_shard*.jsonl")
    parser.add_argument("--config", default="experiments/configs/searchclaw_raes_light_gate.yaml")
    parser.add_argument("--prefix", default="qwen_plus_searchclaw_raes_light_1k")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--temperatures", default="0,0.2,0.4,0.6")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--training-dir", default="outputs/training")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=60, help="Seconds between shard progress updates.")
    parser.add_argument(
        "--completion-only",
        action="store_true",
        help="Only print one compact progress line when each shard job finishes; suppress START lines and child output.",
    )
    parser.add_argument(
        "--suppress-child-output",
        action="store_true",
        help="Suppress child run output while keeping parent START/DONE/progress lines visible.",
    )
    parser.add_argument(
        "--sample-progress",
        action="store_true",
        help="Print one global progress line whenever one sample is appended to a child results.jsonl.",
    )
    parser.add_argument(
        "--sample-progress-interval",
        type=float,
        default=5.0,
        help="Seconds between results.jsonl polls when --sample-progress is enabled.",
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--keep-forced-gate", action="store_true", default=True)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue launching later shard jobs after a shard exits non-zero.",
    )
    parser.add_argument("--dry-print", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def parse_temperatures(raw: str, rollouts: int) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        values = ["0"]
    while len(values) < rollouts:
        values.append(values[-1])
    return values[:rollouts]


def temp_label(raw: str) -> str:
    return raw.replace("-", "m").replace(".", "p")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_bar(done: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = round(width * min(done, total) / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ShardJob:
    label: str
    cmd: list[str]
    results_path: Path
    sample_total: int


@dataclass
class ProgressState:
    total: int
    started_at: float
    sample_total: int = 0
    sample_completed: int = 0
    completed: int = 0
    failed: int = 0
    running: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.running is None:
            self.running = {}


def print_progress(state: ProgressState, *, final: bool = False) -> None:
    elapsed = time.time() - state.started_at
    done = state.completed
    remaining = max(state.total - done, 0)
    avg = elapsed / done if done else 0.0
    eta = avg * remaining if done else 0.0
    active = []
    for label, started_at in sorted((state.running or {}).items())[:6]:
        active.append(f"{label}:{format_duration(time.time() - started_at)}")
    active_text = ", ".join(active) if active else "-"
    pct = (done / max(state.total, 1)) * 100
    status = "FINAL" if final else "PROGRESS"
    print(
        f"[{timestamp()}] {status} {progress_bar(done, state.total)} "
        f"shards={done}/{state.total} ({pct:.1f}%) "
        f"running={len(state.running or {})} failed={state.failed} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"active=[{active_text}]",
        flush=True,
    )


def print_completion(state: ProgressState, label: str, code: int, duration: float) -> None:
    elapsed = time.time() - state.started_at
    done = state.completed
    remaining = max(state.total - done, 0)
    avg = elapsed / done if done else 0.0
    eta = avg * remaining if done else 0.0
    pct = (done / max(state.total, 1)) * 100
    print(
        f"[{timestamp()}] {progress_bar(done, state.total)} "
        f"{done}/{state.total} ({pct:.1f}%) done={label} exit={code} "
        f"time={format_duration(duration)} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} failed={state.failed}",
        flush=True,
    )


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def print_sample_progress(state: ProgressState, label: str, sample_index: int, sample_total: int) -> None:
    elapsed = time.time() - state.started_at
    done = state.sample_completed
    remaining = max(state.sample_total - done, 0)
    avg = elapsed / done if done else 0.0
    eta = avg * remaining if done else 0.0
    pct = (done / max(state.sample_total, 1)) * 100
    print(
        f"[{timestamp()}] {progress_bar(done, state.sample_total)} "
        f"samples={done}/{state.sample_total} ({pct:.1f}%) "
        f"last={label}:{sample_index}/{sample_total} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"failed_shards={state.failed}",
        flush=True,
    )


async def sample_progress_monitor(
    job: ShardJob,
    state: ProgressState,
    *,
    stop_event: asyncio.Event,
    poll_interval: float,
) -> None:
    seen = min(count_lines(job.results_path), job.sample_total)
    try:
        while not stop_event.is_set():
            await asyncio.sleep(poll_interval)
            current = min(count_lines(job.results_path), job.sample_total)
            while seen < current:
                seen += 1
                state.sample_completed += 1
                print_sample_progress(state, job.label, seen, job.sample_total)
    except asyncio.CancelledError:
        pass
    finally:
        current = min(count_lines(job.results_path), job.sample_total)
        while seen < current:
            seen += 1
            state.sample_completed += 1
            print_sample_progress(state, job.label, seen, job.sample_total)


async def progress_ticker(state: ProgressState, interval: int) -> None:
    try:
        while state.completed < state.total:
            await asyncio.sleep(interval)
            print_progress(state)
    except asyncio.CancelledError:
        return


async def run_one(
    job: ShardJob,
    sem: asyncio.Semaphore,
    state: ProgressState,
    *,
    completion_only: bool = False,
    suppress_child_output: bool = False,
    sample_progress: bool = False,
    sample_progress_interval: float = 5.0,
    stop_event: asyncio.Event | None = None,
    stop_on_failure: bool = True,
) -> int | None:
    async with sem:
        if stop_on_failure and stop_event is not None and stop_event.is_set():
            return None
        start = time.time()
        assert state.running is not None
        state.running[job.label] = start
        if not completion_only:
            print(f"[{timestamp()}] START {job.label} cmd={' '.join(job.cmd)}", flush=True)
        stop_monitor = asyncio.Event()
        monitor_task = None
        if sample_progress:
            monitor_task = asyncio.create_task(
                sample_progress_monitor(
                    job,
                    state,
                    stop_event=stop_monitor,
                    poll_interval=max(1.0, sample_progress_interval),
                )
            )
        proc_kwargs = {"cwd": ROOT}
        if completion_only or suppress_child_output:
            proc_kwargs["stdout"] = asyncio.subprocess.DEVNULL
            proc_kwargs["stderr"] = asyncio.subprocess.DEVNULL
        proc = await asyncio.create_subprocess_exec(*job.cmd, **proc_kwargs)
        code = await proc.wait()
        if monitor_task:
            stop_monitor.set()
            await monitor_task
        state.running.pop(job.label, None)
        state.completed += 1
        if code != 0:
            state.failed += 1
            if stop_on_failure and stop_event is not None:
                stop_event.set()
        duration = time.time() - start
        if completion_only:
            print_completion(state, job.label, code, duration)
        else:
            print(f"[{timestamp()}] DONE {job.label} exit={code} time={format_duration(duration)}", flush=True)
            print_progress(state)
        return code


async def main_async() -> None:
    args = parse_args()
    shards = sorted(Path(args.shard_dir).glob(args.shard_glob))
    if not shards:
        raise SystemExit(f"No shards matched {args.shard_dir}/{args.shard_glob}")
    temperatures = parse_temperatures(args.temperatures, args.rollouts)
    jobs: list[ShardJob] = []
    for rollout_idx, temp in enumerate(temperatures):
        for shard_idx, shard in enumerate(shards):
            shard_prefix = f"{args.prefix}_shard{shard_idx:02d}"
            label = f"rollout{rollout_idx:02d}_t{temp_label(temp)}_shard{shard_idx:02d}"
            inner_run_id = f"{shard_prefix}_rollout00_t{temp_label(temp)}"
            results_path = Path(args.output_dir) / inner_run_id / "results.jsonl"
            sample_total = count_lines(shard)
            cmd = [
                sys.executable,
                "experiments/run_raes_seed2k.py",
                "--dataset",
                str(shard),
                "--config",
                args.config,
                "--prefix",
                shard_prefix,
                "--rollouts",
                "1",
                "--temperatures",
                temp,
                "--output-dir",
                args.output_dir,
                "--training-dir",
                args.training_dir,
                "--seed",
                str(args.seed + rollout_idx * 1000 + shard_idx),
            ]
            if args.no_progress:
                cmd.append("--no-progress")
            if args.skip_export:
                cmd.append("--skip-export")
            if args.keep_forced_gate:
                cmd.append("--keep-forced-gate")
            jobs.append(ShardJob(label=label, cmd=cmd, results_path=results_path, sample_total=sample_total))

    if args.dry_print:
        for job in jobs:
            print(" ".join(job.cmd))
        return

    existing_samples = sum(min(count_lines(job.results_path), job.sample_total) for job in jobs)
    state = ProgressState(
        total=len(jobs),
        started_at=time.time(),
        sample_total=sum(job.sample_total for job in jobs),
        sample_completed=existing_samples,
    )
    if not args.completion_only:
        print_progress(state)
    ticker = None
    if args.progress_interval > 0 and not args.completion_only:
        ticker = asyncio.create_task(progress_ticker(state, args.progress_interval))
    sem = asyncio.Semaphore(max(1, args.parallel))
    stop_event = asyncio.Event()
    results = await asyncio.gather(*(
        run_one(
            job,
            sem,
            state,
            completion_only=args.completion_only,
            suppress_child_output=args.suppress_child_output,
            sample_progress=args.sample_progress,
            sample_progress_interval=args.sample_progress_interval,
            stop_event=stop_event,
            stop_on_failure=not args.continue_on_failure,
        )
        for job in jobs
    ))
    if ticker:
        ticker.cancel()
        try:
            await ticker
        except asyncio.CancelledError:
            pass
    if not args.completion_only:
        print_progress(state, final=True)
    failures = [code for code in results if code not in {None, 0}]
    skipped = sum(code is None for code in results)
    if failures:
        suffix = f"; skipped_later_jobs={skipped}" if skipped else ""
        raise SystemExit(f"{len(failures)} shard commands failed: {failures[:10]}{suffix}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

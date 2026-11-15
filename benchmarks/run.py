#!/usr/bin/env python3
"""Check the frozen sampler receipts and rerun the cache-policy benchmarks."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import random
import statistics
import sys
import tempfile

from receipt import groups, load_receipt, simulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from runtime import DEFAULT_CAP, DecodeError, default_binary, invoke

HERE = Path(__file__).resolve().parent
FIXTURES = json.loads((HERE / "fixtures.json").read_text())
MIB = 1024 * 1024
COMPARISON_CAP_MIB = 3
BYTES_PER_PROMPT = 24
TIMING_POLICIES = ("private", "lru8", "lru16", "last", "lease", "ticket")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(trace, requests):
    lanes = defaultdict(list)
    for row in trace.splitlines(keepends=True):
        kind, lane, _ = row.split(b",", 2)
        require(kind in (b"P", b"T", b"K"), "unknown receipt record")
        lanes[int(lane)].append(row)
    require(set(lanes) == set(range(requests)), "receipt lanes do not match inputs")
    return b"".join(row for i in range(requests) for row in lanes[i])


def prompt_file(name, count, folder):
    case = FIXTURES["cases"][name]
    p = HERE / "prompts" / f"{case['prompt']}.u32"
    if count == case["requests"]:
        return p
    partial = folder / f"{name}-{count}.u32"
    data = p.read_bytes()
    require(len(data) == BYTES_PER_PROMPT * case["requests"], f"bad prompt file: {p}")
    require(1 <= count < case["requests"], f"bad prompt prefix: {count}")
    # A case/prefix pair could be shared by jobs. Prepare it before launching them.
    partial.write_bytes(data[: BYTES_PER_PROMPT * count])
    return partial


def run(binary, job, prompts, *, cap, allow_oom=False, trace=True):
    case = FIXTURES["cases"][job["case"]]
    policy = job["policy"]
    params = dict(
        tokens=FIXTURES["horizon"],
        budget=case["budget"],
        seed=case["seed"],
        mask=case["mask"],
        width=job["width"],
        policy=policy,
        io=job.get("io", "buffered"),
        domain=job.get("domain", "shared"),
        trace=trace,
    )
    try:
        raw, stats = invoke(binary, prompts, cap=cap, **params)
    except DecodeError as exc:
        if allow_oom and exc.alloc_failure:
            return {**job, "data_limit_bytes": cap, "allocated_at_2_mib": False}
        hint = (
            " Increase --data-limit-mib for all arms, or use capacity to measure fit."
            if exc.alloc_failure
            else ""
        )
        raise DecodeError(
            f"{job['case']} / {policy} / {params['io']} at {cap} B: {exc}.{hint}"
        ) from exc

    requests = job.get("requests", case["requests"])
    require(stats["data_limit_bytes"] == cap, f"{job}: process data limit differs")
    require(
        stats["requests"] == requests and stats["tokens"] == requests * params["tokens"],
        f"{job}: completion counts differ",
    )
    require(
        stats["logical"]
        == stats["backing"] + sum(stats[f"hit_{k}"] for k in ("lru", "last", "idle", "pinned")),
        f"{job}: I/O ledger mismatch",
    )
    require(
        stats["all_preads"] == stats["decode_preads"] + stats["prefill"],
        f"{job}: syscall ledger mismatch",
    )

    if "backing" in job:
        require(stats["backing"] == job["backing"], f"{job}: backing reads were {stats['backing']}")
    digest = None
    if trace:
        canon = canonical(raw, requests)
        digest = sha256(canon).hexdigest()
        expected = job.get("receipt_sha256", case["receipt_sha256"])
        require(digest == expected, f"{job}: differs from the frozen proposal/token/tick receipt")
        streams = load_receipt(canon)
        predicted = simulate(
            groups(streams, job["width"]), job["width"], policy, params["domain"] == "isolated"
        )
        for key in (
            "logical",
            "backing",
            "hit_lru",
            "hit_last",
            "hit_idle",
            "hit_pinned",
            "copied_bytes",
            "page_bytes",
            "peak_valid_pages",
            "max_pins_on_one_page",
            "overrides",
            "announced_reads",
            "announced_hits",
        ):
            require(predicted[key] == stats[key], f"{job}: independent replay disagrees on {key}")
    else:
        require(not raw, f"{job}: unexpected output in quiet timing mode")

    return {
        **job,
        **stats,
        "receipt_sha256": digest,
        **({"allocated_at_2_mib": True} if allow_oom else {}),
    }


def deterministic(args, binary, folder):
    all_jobs = FIXTURES[args.suite]
    jobs = [j for j in all_jobs if not (args.buffered_only and j.get("io") == "direct")]
    capacity = args.suite == "capacity"
    paths = {}
    for j in jobs:
        name = j["case"]
        key = name, j.get("requests", FIXTURES["cases"][name]["requests"])
        if key not in paths:
            paths[key] = prompt_file(*key, folder)

    def worker(job):
        key = job["case"], job.get("requests", FIXTURES["cases"][job["case"]]["requests"])
        return run(binary, job, paths[key], cap=args.cap, allow_oom=capacity)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(worker, jobs))
    if capacity:
        matched = sum(r["allocated_at_2_mib"] == r["archived_fit"] for r in rows)
        fit = sum(r["allocated_at_2_mib"] for r in rows)
        print(
            f"Capacity: {fit}/{len(rows)} trials fit the 2 MiB data limit; "
            f"{matched} allocation outcomes match the archive."
        )
    else:
        skipped = len(all_jobs) - len(jobs)
        print(
            f"Validated {len(rows)} configurations at {args.cap // MIB} MiB against frozen receipts and the independent cache replay."
            + (f" Skipped {skipped} direct-I/O configurations by request." if skipped else "")
        )
        if args.suite == "full":
            for family in ("primary", "challenge"):
                baseline = sum(
                    r["backing"] for r in rows if r["family"] == family and r["policy"] == "private"
                )
                ticket = sum(
                    r["backing"] for r in rows if r["family"] == family and r["policy"] == "ticket"
                )
                print(
                    f"{family}: {baseline:,} private vs {ticket:,} DrawPin backing reads "
                    f"({100 * (baseline - ticket) / baseline:.2f}% fewer)."
                )
    return {"suite": args.suite, "data_limit_bytes": args.cap, "rows": rows}


def percentile(sorted_values, p):
    i = (len(sorted_values) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_values[lo] + (i - lo) * (sorted_values[hi] - sorted_values[lo])


def timing(args, binary, folder):
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    records = []
    summaries = []
    rng = random.Random(391793)
    cap = args.cap
    for case in ("primary_heldout_0", "primary_shift_0"):
        p = prompt_file(case, 32, folder)
        for io in ("buffered",) if args.buffered_only else ("buffered", "direct"):
            per_arm = defaultdict(list)
            logical = None
            for round_index in range(args.rounds + 1):
                k = (round_index - 1) % len(TIMING_POLICIES)
                order = TIMING_POLICIES[k:] + TIMING_POLICIES[:k]
                if round_index > len(order):
                    order = order[::-1]
                for policy in order:
                    job = dict(case=case, policy=policy, io=io, width=16, requests=32)
                    result = run(binary, job, p, trace=False, cap=cap)
                    if logical is None:
                        logical = result["logical"]
                    require(
                        result["logical"] == logical,
                        "timing policies charged different logical reads",
                    )
                    records.append({**result, "round": round_index, "warmup": round_index == 0})
                    if round_index:
                        per_arm[policy].append(result["elapsed_ns"])
            ns_per_tok = {k: statistics.median(v) / 1024 for k, v in per_arm.items()}
            pairs = [a / b for a, b in zip(per_arm["ticket"], per_arm["private"])]
            resampled = sorted(
                statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(20_000)
            )
            ratio = statistics.median(pairs)
            interval = [percentile(resampled, p) for p in (0.025, 0.975)]
            summaries.append(
                dict(
                    case=case,
                    io=io,
                    data_limit_bytes=cap,
                    median_ns_per_token=ns_per_tok,
                    paired_ratio=ratio,
                    repeatability_interval=interval,
                )
            )
            print(
                f"{case}/{io} at {cap // MIB} MiB: DrawPin / private {ratio:.3f} "
                f"[{interval[0]:.3f}, {interval[1]:.3f}] (wall time).",
                flush=True,
            )
    return dict(
        suite="timing",
        pinned_cpu=cpu,
        kernel=os.uname().release,
        data_limit_bytes=cap,
        rounds=args.rounds,
        summaries=summaries,
        rows=records,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("quick", "full", "capacity", "timing"))
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, help="write fresh measurements as JSON")
    parser.add_argument(
        "--jobs", type=int, default=2, help="keep one process per job; timing ignores this"
    )
    parser.add_argument("--rounds", type=int, default=9, help="paired timing rounds after warmup")
    parser.add_argument(
        "--buffered-only", action="store_true", help="explicitly omit direct I/O cases"
    )
    parser.add_argument(
        "--data-limit-mib",
        type=int,
        help=f"common comparison cap (default: {COMPARISON_CAP_MIB}; capacity is fixed at 2)",
    )
    args = parser.parse_args()
    if not __debug__:
        parser.error("run without -O; the independent replay uses assertions")
    if args.jobs < 1 or args.rounds < 1:
        parser.error("--jobs and --rounds must be positive")
    if args.data_limit_mib is not None and not 1 <= args.data_limit_mib <= 1024:
        parser.error("--data-limit-mib must be in [1, 1024]")
    if args.suite == "capacity" and args.data_limit_mib is not None:
        parser.error("capacity measures allocation at a fixed 2 MiB cap")
    args.cap = (
        DEFAULT_CAP
        if args.suite == "capacity"
        else (args.data_limit_mib or COMPARISON_CAP_MIB) * MIB
    )
    try:
        binary = args.binary if args.binary is not None else default_binary()
        with tempfile.TemporaryDirectory(prefix="drawpin-bench-") as d:
            fn = timing if args.suite == "timing" else deterministic
            result = fn(args, binary, Path(d))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (DecodeError, OSError, ValueError, AssertionError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

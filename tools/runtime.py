"""File layout, native invocation and tokenization for DrawPin."""

from pathlib import Path
import json
import os
import re
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
PAGE = 4096
CONTEXT = 6
VOCABULARY_SIZE = 16384
DEFAULT_BUDGET = (2, 8, 4, 8)
DEFAULT_CAP = 2 * 1024 * 1024
POLICIES = {name: (name, 0) for name in ("private", "last", "lease", "alias", "ticket")}
POLICIES.update({f"lru{n}": ("lru", n) for n in (4, 8, 16, 32)})
CHUNK = re.compile(rb" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]|\s+")


class DecodeError(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.alloc_failure = code == 2 and "bad_alloc" in message


def default_binary():
    cpuinfo = Path("/proc/cpuinfo")
    avx2 = (
        os.uname().machine == "x86_64"
        and cpuinfo.exists()
        and "avx2" in cpuinfo.read_text().split()
    )
    variants = ("avx2", "scalar") if avx2 else ("scalar",)
    for variant in variants:
        candidate = ROOT / "build" / variant / "drawpin"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise DecodeError("DrawPin is not built. Run make or set --binary.")


def vocab():
    values = [bytes.fromhex(token) for token in json.loads((MODEL / "vocabulary.json").read_text())]
    if (
        len(values) != VOCABULARY_SIZE
        or len(set(values)) != len(values)
        or values[:256] != [bytes([x]) for x in range(256)]
    ):
        raise DecodeError("Invalid frozen vocabulary. Run make verify.")
    return values


def tokenize(text, lookup):
    tokens = []
    for chunk in CHUNK.findall(text.encode("utf-8")):
        token = lookup.get(chunk)
        if token is None:
            tokens.extend(chunk)
        else:
            tokens.append(token)
    return tokens


def write_prompts(path, prompts):
    with path.open("wb") as file:
        for ids in prompts:
            file.write(struct.pack("<6I", *ids))


def invoke(
    binary,
    prompts,
    *,
    tokens=32,
    budget=DEFAULT_BUDGET,
    seed=184467,
    mask="masked",
    width=16,
    policy="ticket",
    io="buffered",
    domain="shared",
    cap=DEFAULT_CAP,
    trace=True,
    timeout=300,
):
    strategy, spare = POLICIES[policy]
    env = {key: value for key, value in os.environ.items() if not key.startswith("QB_")}
    env["QB_DATA_LIMIT_BYTES"] = str(cap)
    if not trace:
        env["QB_SPOOL_QUIET"] = "1"
    model = MODEL.relative_to(ROOT)
    args = [
        str(binary.resolve()),
        str(model / "head"),
        str(model / "refinement"),
        str(model / "body"),
        os.path.relpath(prompts.resolve(), ROOT),
        *map(str, (tokens, *budget, seed, mask, width, strategy, spare, io, domain, 0)),
    ]
    try:
        start = time.perf_counter_ns()
        proc = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
        elapsed = time.perf_counter_ns() - start
    except OSError as exc:
        raise DecodeError(f"Cannot start DrawPin: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DecodeError(f"Decoder exceeded {timeout:g}s.") from exc
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise DecodeError(msg or f"Decoder exited with status {proc.returncode}.", proc.returncode)
    try:
        stats = json.loads(proc.stderr)
    except (ValueError, TypeError) as exc:
        raise DecodeError("Decoder did not produce a valid statistics record.") from exc
    stats["elapsed_ns"] = elapsed
    return proc.stdout, stats

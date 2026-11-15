<h1 align="center">
DrawPin
</h1>

DrawPin is an exact CPU decoder for a frozen recurrent language model with 2,113,664
parameters. A request that pauses for read credit reveals the next immutable
weight sector it will need. DrawPin uses that hint to decide which
unpinned sector to keep. A cache hit still incurs the same logical read
charge, so the sampling decisions and service ticks stay the same.

## Build and run

Requirements: 64-bit, little-endian Linux, a C++20 compiler, GNU Make, GMP
development headers and Python 3. The model and tokenizer are included.
The build does not fetch anything or require Python packages. `make verify`
checks the hashes of the frozen model and benchmark inputs.

On Debian or Ubuntu:

```sh
sudo apt install build-essential libgmp-dev python3
```

From the extracted repository:

```sh
make verify
make
python3 tools/generate.py --tokens 16 "She walked along the river with her sister"
```

Pass multiple quoted prompts to exercise the shared pool:

```sh
python3 tools/generate.py --tokens 16 \
  "She walked along the river with her sister" \
  "He looked up at the books on the shelf"
```

The model uses the last six model tokens of each prompt. By default the
CLI schedules at most 16 live requests at a time and sets a 2 MiB process
data limit. Adjust these with `--width` and `--data-limit-mib`. Use `--seed` for
repeatable sampling, `--allow-repeat` to allow tokens from the current
context, or `--policy private` and `--policy pool-lru` for comparison. Run
`python3 tools/generate.py --help` for the full interface.

Make selects AVX2 on supporting x86-64 hosts and otherwise builds the scalar
decoder. To build and test the scalar path explicitly:

```sh
make CPU=scalar test
```

## Verify and reproduce

For development, install the pinned formatting tools in a virtual environment.
On Debian or Ubuntu, install `python3-venv` first if needed:

```sh
sudo apt install python3-venv
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
make format
make lint
```

`make format` applies clang-format to C++ sources and Ruff to Python sources.
`make lint` checks formatting and Python lint rules without changing files. CI
runs the same checks. These tools are only required for development.

```sh
make test
make benchmark JOBS=4
make capacity
make timing
```

- **test** checks the text CLI, adversarial page eviction schedules, 5,120
  exponential table entries, 32 native near-threshold decisions, and six
  model configurations against the frozen traces.
- **benchmark** runs 135 configurations. It compares the canonical
  proposal, token and service-tick SHA-256 hashes, checks the archived
  backing-read counts, and reconstructs the cache accounting with an
  independent Python replay.
- **capacity** reports the allocation outcome of 34 current-engine
  configurations at the 2 MiB Linux RLIMIT_DATA cap. This cap does not
  bound total RSS. Allocation failures are recorded; successful runs
  must still match the frozen trace. Allocation results can vary with the
  build and allocator.
- **timing** executes fresh processes on one pinned CPU, with a warmup and
  nine paired rounds per condition. It stores wall-clock observations and
  a descriptive paired bootstrap interval. Run it by itself.

Fresh JSON outputs go under `build/results/`. Comparison runs (including
timing) use a common 3 MiB data limit for all policies so allocator failures
at 2 MiB do not interfere with the read-count comparison. The `capacity`
suite measures the 2 MiB limit separately and never retries at a higher cap.
You can set a different common cap with `benchmarks/run.py --data-limit-mib`.
The full and timing suites include direct I/O. To omit it explicitly on
filesystems that do not support it, run:

```sh
python3 benchmarks/run.py full --buffered-only --output build/results/buffered.json
```

The reference read counts at width 16 and equal 64 KiB page payload are:

| Workload | Private | DrawPin | Reduction |
| --- | ---: | ---: | ---: |
| Archived prompts and seeds | 15,168 | 13,120 | 13.50% |
| New-book prompts and seeds | 5,248 | 4,588 | 12.58% |

These count sector fetches from the backing file during decode. Prefill
reads are separate. A backing-file fetch does not imply a physical disk
access, and the measurements do not establish a consistent speedup.

## Repository layout

- `src/`: standalone C++ decoder, exact sampler and sector pool.
- `model/`: the packed frozen weights, refinement tables and vocabulary.
- `tools/`: Python standard-library CLI and native invocation.
- `benchmarks/`: the deterministic prompt inputs, reference trace hashes,
  expected read counts, and replay code.
- `tests/`: native numerical and eviction checks and the CLI regression.

The original DrawPin study archive hash is recorded in
`benchmarks/fixtures.json`. The reference prompts, frozen model and
sampling arithmetic were preserved when extracting the standalone
decoder. No parent study archives are needed.

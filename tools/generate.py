#!/usr/bin/env python3
"""Generate continuations from the frozen CPU language model."""

import argparse
from pathlib import Path
import sys
import tempfile

from runtime import CONTEXT, DecodeError, default_binary, invoke, tokenize, vocab, write_prompts

POLICY_NAMES = {"drawpin": "ticket", "pool-lru": "lease", "private": "private"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="+", help="one quoted prompt per request")
    parser.add_argument(
        "--tokens", type=int, default=32, help="tokens per continuation (default: 32)"
    )
    parser.add_argument("--seed", type=int, default=184467)
    parser.add_argument("--width", type=int, default=16, help="maximum live requests (default: 16)")
    parser.add_argument(
        "--data-limit-mib", type=int, default=2, help="Linux process data limit (default: 2)"
    )
    parser.add_argument("--policy", choices=tuple(POLICY_NAMES), default="drawpin")
    parser.add_argument(
        "--allow-repeat", action="store_true", help="allow tokens from the current context"
    )
    parser.add_argument("--binary", type=Path, help="override the native decoder path")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 4096 or not 0 <= args.seed < 2**64:
        parser.error("tokens must be in [1, 4096] and seed must fit an unsigned 64-bit integer")
    if not 1 <= args.width <= 128 or not 1 <= args.data_limit_mib <= 1024:
        parser.error("width must be in [1, 128] and the data limit in [1, 1024] MiB")

    try:
        binary = args.binary if args.binary is not None else default_binary()
        values = vocab()
        lookup = {word: i for i, word in enumerate(values)}
        requests = [tokenize(text, lookup) for text in args.prompts]
        for i, tokens in enumerate(requests, 1):
            if len(tokens) < CONTEXT:
                parser.error(f"prompt {i} needs at least {CONTEXT} model tokens")
        with tempfile.TemporaryDirectory(prefix="drawpin-") as tmp:
            path = Path(tmp) / "prompts.u32"
            write_prompts(path, [t[-CONTEXT:] for t in requests])
            trace, stats = invoke(
                binary,
                path,
                tokens=args.tokens,
                seed=args.seed,
                width=min(args.width, len(requests)),
                policy=POLICY_NAMES[args.policy],
                mask="all" if args.allow_repeat else "masked",
                cap=args.data_limit_mib * 1024 * 1024,
            )
        outputs = [bytearray() for _ in requests]
        counts = [0] * len(requests)
        for line in trace.splitlines():
            if line.startswith(b"T,"):
                _, lane, step, token, _ = line.split(b",", 4)
                i = int(lane)
                if not 0 <= i < len(outputs) or int(step) != counts[i]:
                    raise DecodeError("Decoder emitted tokens out of order.")
                outputs[i].extend(values[int(token)])
                counts[i] += 1
        if counts != [args.tokens] * len(requests):
            raise DecodeError("Decoder returned an incomplete continuation.")
    except (DecodeError, OSError, ValueError, IndexError) as exc:
        print(f"DrawPin: {exc}", file=sys.stderr)
        if isinstance(exc, DecodeError) and exc.alloc_failure:
            print("Try a smaller --width or a larger --data-limit-mib.", file=sys.stderr)
        return 2

    for i, text in enumerate(outputs, 1):
        print(f"{i}. {text.decode('utf-8', errors='replace')}")
    print(
        f"{stats['logical']} logical decode reads; {stats['backing']} backing reads; "
        f"{stats['prefill']} prefill reads; {stats['page_bytes']} bytes of page payload",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

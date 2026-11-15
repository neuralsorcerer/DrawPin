#!/usr/bin/env python3
"""Decimal cross-check of a frozen fixed-point table and selected native thresholds."""

from decimal import Decimal, localcontext
from hashlib import sha256
from pathlib import Path
import argparse
import json
import re
import subprocess

GAPS = (
    1,
    2,
    17,
    511,
    1023,
    1024,
    1025,
    8191,
    8192,
    8193,
    123456,
    1_000_000,
    4_193_279,
    4_194_303,
    4_194_304,
    4_194_305,
)


def make_cases(header, prec):
    q = Decimal

    def ints(name):
        r = re.search(r"\b" + name + r"\s*=\s*\{(.*?)\};", header, re.S)[1]
        return [int(n) for n in re.findall(r"(\d+)ULL", r)]

    B, R = ints("EXP_BLOCK"), ints("EXP_REMAINDER")
    num, den = (
        int(re.search(name + r'\[\]\s*=\s*"(\d+)"', header)[1]) for name in ("BETA_NUM", "BETA_DEN")
    )
    with localcontext() as c:
        c.prec = prec
        beta = q(num) / den
        for xs, step in ((B, 1024), (R, 1)):
            for i, want in enumerate(xs):
                val = (-beta * i * step).exp() * (2**62)
                assert int(val) == want, (step, i, val, want)
        assert (-beta * 1024 * len(B)).exp() * (2**62) < 1
        rows = []
        denom = 2**64
        for gap in GAPS:
            bound = (-beta * gap).exp() * (2**112)
            pos = int(bound)
            assert pos > 2 and pos < 2**112 - 3
            for cell, want in ((pos - 2, 1), (pos + 2, 0)):
                assert (cell + 1) < bound if want else cell > bound
                offset, word = divmod(cell, denom)
                rows.append((gap, offset, word, want))
    return rows, len(B) + len(R)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not __debug__:
        p.error("run without -O; the numerical oracle uses assertions")
    header = a.grid.read_text()
    rows, n = make_cases(header, 95)
    assert (rows, n) == make_cases(header, 145)
    inp = "\n".join(" ".join(map(str, r)) for r in rows) + "\n"
    r = subprocess.run(
        [str(a.exe.resolve())], input=inp, text=True, check=True, capture_output=True
    )
    out = json.loads(r.stdout)
    out.update(
        grid_sha256=sha256(a.grid.read_bytes()).hexdigest(),
        grid_floor_entries=n,
        decimal_digits=[95, 145],
        score_gaps=GAPS,
    )
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

"""Ensure corrupted numerical tables cannot produce a successful validation report."""

from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    args = parser.parse_args()
    header = (ROOT / "src" / "exp_grid.hpp").read_text()
    corrupt = header.replace("4611686018427387904ULL", "0ULL", 1)
    if corrupt == header:
        raise RuntimeError("could not corrupt the first table entry")
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    with tempfile.TemporaryDirectory(prefix="drawpin-oracle-") as tmp:
        grid = Path(tmp) / "corrupt.hpp"
        grid.write_text(corrupt)
        output = Path(tmp) / "report.json"
        for flags in ((), ("-O",), ("-OO",)):
            result = subprocess.run(
                [
                    sys.executable,
                    *flags,
                    str(ROOT / "tests" / "threshold_cases.py"),
                    "--grid",
                    str(grid),
                    "--exe",
                    str(args.exe.resolve()),
                    "--out",
                    str(output),
                ],
                env=env,
                capture_output=True,
            )
            expected = b"numerical oracle uses assertions" if flags else b"AssertionError"
            if result.returncode == 0 or output.exists() or expected not in result.stderr:
                raise RuntimeError(f"corrupted table was not safely rejected with flags {flags}")
    print("Corrupted numerical tables and optimized Python validation were rejected.")


if __name__ == "__main__":
    main()

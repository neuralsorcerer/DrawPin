"""Regression check against the original DrawPin command's fixed-seed sample."""

from argparse import ArgumentParser
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ("She walked along the river with her sister", "He looked up at the books on the shelf")
EXPECTED = b"1.  universal has had I\n2. . children sentiments notice\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "generate.py"),
        "--binary",
        str(args.binary.resolve()),
        "--tokens",
        "4",
    ]
    # Legacy debug flags in the invoking shell must not change the public CLI.
    env = os.environ.copy()
    env["QB_FOLD_LAYOUT"] = "not-a-layout"
    env["QB_SPOOL_QUIET"] = "1"
    with tempfile.TemporaryDirectory() as cwd:
        ok = subprocess.run([*cmd, *PROMPTS], env=env, cwd=cwd, capture_output=True, check=True)
        if ok.stdout != EXPECTED:
            raise RuntimeError(f"unexpected fixed-seed output: {ok.stdout!r}")
        bad = subprocess.run([*cmd, "Hi"], env=env, cwd=cwd, capture_output=True)
        if bad.returncode != 2 or b"at least 6 model tokens" not in bad.stderr:
            raise RuntimeError("short prompt did not get a useful error")
    print("CLI fixed-seed and input checks passed.")


if __name__ == "__main__":
    main()

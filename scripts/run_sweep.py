#!/usr/bin/env python
"""Run a grid of experiments (methods x seeds x one swept parameter) as separate processes.

Examples
--------
# Table VI: 13 methods x 30 seeds on one dataset
python scripts/run_sweep.py --config configs/mofedcyber.yaml --methods all --seeds 30

# Fig. 10: Byzantine fraction sweep
python scripts/run_sweep.py --config configs/byzantine.yaml --methods mofedcyber,flame,bulyan,krum,median,fedavg \
    --sweep attack.byzantine_fraction=0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4 --seeds 30
"""

import argparse
import itertools
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from mofedcyber.fl import METHODS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", default="mofedcyber")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--sweep", default=None, help="key=v1,v2,...")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    methods = METHODS if a.methods == "all" else a.methods.split(",")
    key, values = (None, [None])
    if a.sweep:
        key, vals = a.sweep.split("=", 1)
        values = vals.split(",")
    for m, s, v in itertools.product(methods, range(a.seeds), values):
        tag = f"{m}_seed{s}" + (f"_{key.split('.')[-1]}{v}" if key else "")
        sets = [f"method={m}", f"seed={s}"] + list(a.set)
        if key:
            sets.append(f"{key}={v}")
        name = os.path.splitext(os.path.basename(a.config))[0]
        out = os.path.join(ROOT, "results", name, tag + ".json")
        if os.path.exists(out):
            print("skip", out)
            continue
        cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "run_experiment.py"),
            "--config",
            a.config,
            "--out",
            out,
            "--quiet",
            "--set",
            *sets,
        ]
        print(" ".join(cmd))
        if not a.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Aggregate result JSON files into the paper's tables.

python scripts/summarize_results.py results/mofedcyber      # mean +- std macro-F1, Wilcoxon-Holm
python scripts/summarize_results.py results/mofedcyber --fronts   # HV / IGD / spacing of archives
"""

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mofedcyber.metrics import front_quality, wilcoxon_holm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ours", default="mofedcyber")
    ap.add_argument("--fronts", action="store_true")
    a = ap.parse_args()
    runs = collections.defaultdict(dict)
    fronts = collections.defaultdict(list)
    for f in glob.glob(os.path.join(a.folder, "*.json")):
        r = json.load(open(f))
        m, s = r["config"]["method"], r["config"]["seed"]
        runs[m][s] = r["final_macro_f1"]
        if r.get("archive_objectives"):
            fronts[m].append(np.array(r["archive_objectives"]))
    print(f"{'method':14s} {'n':>3s} {'macro-F1 mean':>14s} {'std':>7s}")
    for m in sorted(runs, key=lambda k: -np.mean(list(runs[k].values()))):
        v = np.array(list(runs[m].values())) * 100
        print(f"{m:14s} {len(v):3d} {v.mean():14.2f} {v.std(ddof=1) if len(v) > 1 else 0:7.2f}")
    if a.ours in runs and len(runs) > 1:
        seeds = sorted(runs[a.ours])
        base = {m: [runs[m].get(s, np.nan) for s in seeds] for m in runs if m != a.ours}
        base = {m: v for m, v in base.items() if not np.isnan(v).any() and len(v) >= 6}
        if base:
            print("\nWilcoxon signed-rank (two-sided), Holm-Bonferroni corrected:")
            for m, p in wilcoxon_holm([runs[a.ours][s] for s in seeds], base).items():
                print(f"  {a.ours} vs {m:12s} p_raw={p['p_raw']:.2e}  p_holm={p['p_holm']:.2e}")
    if a.fronts and fronts:
        q = front_quality({m: np.vstack(v) for m, v in fronts.items()})
        print("\nFront quality (pooled archives, jointly normalized):")
        for m, d in q.items():
            print(
                f"  {m:12s} HV={d['hv']:.3f} IGD={d['igd']:.3f} spacing={d['spacing']:.3f} size={d['size']}"
            )


if __name__ == "__main__":
    main()

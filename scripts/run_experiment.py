#!/usr/bin/env python
"""Run one federated experiment.

Examples
--------
python scripts/run_experiment.py --config configs/mofedcyber.yaml
python scripts/run_experiment.py --config configs/mofedcyber.yaml --set seed=3 privacy.target_epsilon=0.1
python scripts/run_experiment.py --config configs/smoke.yaml          # 1-minute CPU check
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mofedcyber.config import load_config, parse_overrides  # noqa: E402
from mofedcyber.fl import Simulation  # noqa: E402
from mofedcyber.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. seed=1 federation.rounds=50")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    t0 = time.time()
    sim = Simulation(cfg, verbose=not a.quiet)
    res = sim.run()
    res["runtime_s"] = time.time() - t0
    if cfg.membership_inference:
        from mofedcyber.experiments import run_membership_inference

        res["membership_inference_accuracy"] = run_membership_inference(sim)
    out = a.out or os.path.join(cfg.output_dir, cfg.name, f"{cfg.method}_seed{cfg.seed}.json")
    save_json(res, out)
    print(f"final macro-F1 = {res['final_macro_f1']:.4f}  ->  {out}")


if __name__ == "__main__":
    main()

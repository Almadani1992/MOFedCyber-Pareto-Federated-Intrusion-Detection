#!/usr/bin/env python
"""Print the noise multiplier sigma_z needed for each privacy budget (Table VIII and Section IV-D).

python scripts/calibrate_noise.py --rho 0.3 --rounds 200 --delta 1e-5 --eps 10 1 0.1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mofedcyber.fl.privacy import calibrate_noise, compute_epsilon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--eps", type=float, nargs="+", default=[10, 1, 0.1])
    ap.add_argument("--check-sigma", type=float, nargs="*", default=[1.1])
    a = ap.parse_args()
    print(f"Poisson client sampling rho={a.rho}, T={a.rounds}, delta={a.delta}")
    for e in a.eps:
        print(f"  eps={e:<6} -> sigma_z = {calibrate_noise(e, a.rho, a.rounds, a.delta):.2f}")
    for s in a.check_sigma:
        print(f"  sigma_z={s} gives eps = {compute_epsilon(a.rho, s, a.rounds, a.delta):.2f}")


if __name__ == "__main__":
    main()

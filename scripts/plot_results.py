#!/usr/bin/env python
"""Plot experiment results in the style of the paper's figures.

python scripts/plot_results.py convergence results/mofedcyber --out figs/convergence.pdf
python scripts/plot_results.py communication results/mofedcyber --out figs/communication.pdf
python scripts/plot_results.py sweep results/byzantine --key attack.byzantine_fraction --scale 100 \
       --xlabel "Fraction of Byzantine clients (%)" --out figs/byzantine.pdf
python scripts/plot_results.py sweep results/dropout --key federation.dropout_rate --scale 100 --bars \
       --xlabel "Client dropout rate per round (%)" --out figs/dropout.pdf
"""

import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
    }
)
STYLE = {  # colour follows the method in every figure (palette checked for colour-vision deficiency)
    "mofedcyber": ("#D62728", "o", "-", "xx", "MOFedCyber"),
    "flame": ("#1F77B4", "s", "--", "//", "FLAME"),
    "scaffold": ("#E69F00", "^", "-.", "\\\\", "SCAFFOLD"),
    "fedmgda+": ("#2CA02C", "D", ":", "..", "FedMGDA+"),
    "fedavg": ("#9467BD", "v", "--", "++", "FedAvg"),
    "fedprox": ("#17A0B0", "p", "-.", "--", "FedProx"),
    "bulyan": ("#17A0B0", "p", "-.", "--", "Bulyan"),
    "krum": ("#E69F00", "^", ":", "\\\\", "Krum"),
    "median": ("#2CA02C", "D", "-.", "..", "Median"),
    "dp-fedavg": ("#1F77B4", "s", "--", "//", "DP-FedAvg"),
    "ldp-fed": ("#9467BD", "v", "-.", "++", "LDP-Fed"),
    "moo-fl-ec": ("#2CA02C", "D", ":", "..", "MOO-FL-$\\epsilon$C"),
    "fednova": ("#8C8C8C", "h", ":", "oo", "FedNova"),
}


def _get(d, key):
    for k in key.split("."):
        d = d[k]
    return d


def load(folder):
    R = collections.defaultdict(list)
    for f in glob.glob(os.path.join(folder, "*.json")):
        r = json.load(open(f))
        R[r["config"]["method"]].append(r)
    return R


def box(ax):
    for s in ax.spines.values():
        s.set_visible(True)


def convergence(R, xkey, xlabel, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for m, runs in R.items():
        c, mk, ls, _, name = STYLE.get(m, ("k", "o", "-", "", m))
        curves = []
        for r in runs:
            h = [x for x in r["history"] if x["macro_f1"] is not None]
            f1 = [
                x["macro_f1_selected"] if x.get("macro_f1_selected") is not None else x["macro_f1"] for x in h
            ]
            curves.append(
                (np.array([x[xkey] + (1 if xkey == "round" else 0) for x in h]), np.array(f1) * 100)
            )
        n = min(len(y) for _, y in curves)
        x = curves[0][0][:n]
        Y = np.stack([y[:n] for _, y in curves])
        mu, sd = Y.mean(0), Y.std(0)
        ax.plot(
            x,
            mu,
            color=c,
            ls=ls,
            marker=mk,
            ms=4,
            mfc="white" if m != "mofedcyber" else c,
            markevery=max(1, n // 10),
            label=name,
        )
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(loc="lower right")
    box(ax)
    fig.savefig(out)
    print("saved", out)


def sweep(R, key, xlabel, out, scale=1.0, bars=False):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    methods = list(R)
    xs = sorted({_get(r["config"], key) for runs in R.values() for r in runs})
    for j, m in enumerate(methods):
        c, mk, ls, h, name = STYLE.get(m, ("k", "o", "-", "", m))
        mu, sd = [], []
        for x in xs:
            v = [r["final_macro_f1"] * 100 for r in R[m] if _get(r["config"], key) == x]
            mu.append(np.mean(v) if v else np.nan)
            sd.append(np.std(v) if v else np.nan)
        if bars:
            w = 0.8 / len(methods)
            pos = np.arange(len(xs)) - 0.4 + (j + 0.5) * w
            ax.bar(
                pos,
                mu,
                w,
                yerr=sd,
                color=c,
                edgecolor="black",
                lw=0.5,
                hatch=h,
                label=name,
                error_kw=dict(elinewidth=0.6, capsize=1.5),
            )
        else:
            ax.errorbar(
                np.array(xs) * scale,
                mu,
                yerr=sd,
                color=c,
                ls=ls,
                marker=mk,
                ms=4,
                capsize=1.8,
                mfc="white" if m != "mofedcyber" else c,
                label=name,
            )
    if bars:
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x * scale:g}" for x in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(loc="lower left", ncol=2)
    box(ax)
    fig.savefig(out)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["convergence", "communication", "sweep"])
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--xlabel", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--bars", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R = load(a.folder)
    if a.kind == "convergence":
        convergence(R, "round", a.xlabel or "Communication round $t$", a.out)
    elif a.kind == "communication":
        convergence(R, "cum_gb", a.xlabel or "Cumulative data transmitted (GB)", a.out)
    else:
        sweep(R, a.key, a.xlabel or a.key, a.out, a.scale, a.bars)


if __name__ == "__main__":
    main()

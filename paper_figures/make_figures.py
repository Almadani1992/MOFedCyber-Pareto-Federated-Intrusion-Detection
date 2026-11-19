"""Regenerates every results figure of the paper from data.py.
Usage: python make_figures.py <output_dir>"""

import os
import sys

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data import (  # noqa: E402
    ABLATION,
    ALPHA,
    BYZ,
    BYZ_X,
    CONV,
    DATASET_INFO,
    DATASETS,
    DROP,
    DROP_X,
    ENERGY,
    EPS,
    F1,
    HEADS,
    MIA,
    MIA_EPS,
    NO_DP,
    NONIID,
    NSGA_SHARE,
    PARETO,
    PRIV_F1,
    PRIV_NODP,
    SCAL,
    SCAL_N,
    SIGMA_Z,
)

OUT = sys.argv[1] if len(sys.argv) > 1 else "../figures"
os.makedirs(OUT, exist_ok=True)

for f in [
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf",
]:
    if os.path.exists(f):
        fm.fontManager.addfont(f)

INK, INK2, MUTED, GRID, SURF = "#1f2328", "#52514e", "#8a8985", "#e6e4de", "#ffffff"
mpl.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.2,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
    }
)

# Fixed entity colours (validated categorical order) + markers. Colour follows the method everywhere.
C = {
    "MOFedCyber": "#2a78d6",
    "FLAME": "#eb6834",
    "SCAFFOLD": "#1baf7a",
    "FedMGDA+": "#d99400",
    "MOO-FL-εC": "#6250d6",
    "Bulyan": "#e87ba4",
    "Krum": "#008300",
    "Median": "#a0522d",
    "FedAvg": "#6b6a66",
    "FedProx": "#a9a8a2",
    "FedNova": "#c4c2bb",
    "DP-FedAvg": "#9c4f96",
    "LDP-Fed": "#b8b39c",
}
M = {
    "MOFedCyber": "o",
    "FLAME": "s",
    "SCAFFOLD": "^",
    "FedMGDA+": "D",
    "MOO-FL-εC": "P",
    "Bulyan": "v",
    "Krum": "X",
    "Median": "<",
    "FedAvg": "h",
    "FedProx": ">",
    "FedNova": "d",
    "DP-FedAvg": "p",
    "LDP-Fed": "*",
}
COL1, COL2 = 3.45, 7.16


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


def panel(ax, letter, title=None):
    t = f"({letter})" + (f" {title}" if title else "")
    ax.set_title(t, loc="left", fontsize=7.8, color=INK, pad=4, fontweight="bold")


def lineplot(ax, x, series, order, labels_end=True, ms=4.2, err=None):
    for m in order:
        y = series[m]
        hero = m == "MOFedCyber"
        ax.plot(
            x,
            y,
            color=C[m],
            marker=M[m],
            lw=2.0 if hero else 1.2,
            ms=ms + (0.6 if hero else 0),
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5 if hero else 3,
            label=m,
        )
        if err is not None and m in err:
            ax.fill_between(
                x, np.array(y) - err[m], np.array(y) + err[m], color=C[m], alpha=0.15, lw=0, zorder=2
            )


# ---------------------------------------------------------------- Fig: datasets
def fig_datasets():
    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    dom_c = {
        "Enterprise (legacy)": "#a9a8a2",
        "Enterprise": "#2a78d6",
        "Consumer IoT": "#1baf7a",
        "Industrial IoT": "#d99400",
        "Mixed IoT": "#6250d6",
        "5G edge": "#eb6834",
    }
    rows = sorted(DATASET_INFO, key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.scatter(
        [r[1] for r in rows],
        y,
        s=46,
        color=[dom_c[r[4]] for r in rows],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.grid(axis="x", visible=True)
    ax.set_xscale("log")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.grid(axis="y", visible=True, color=GRID, lw=0.5)
    for yi, r in zip(y, rows):
        ax.text(
            r[1] * 1.45,
            yi,
            f"{r[1]/1e6:.2f}M  ·  {r[3]} attack cl.  ·  {r[2]} feat.",
            va="center",
            fontsize=6.2,
            color=INK2,
        )
    ax.set_xlim(8e4, 3e9)
    ax.set_xlabel("Flow records (log scale)")
    hand = [Patch(color=v, label=k) for k, v in dom_c.items()]
    ax.legend(
        handles=hand,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.28),
        fontsize=6.2,
        handlelength=0.9,
        columnspacing=0.8,
    )
    save(fig, "fig3_datasets")


# ---------------------------------------------------------------- Fig: heatmap
def fig_heatmap():
    order = sorted(F1, key=lambda m: F1[m][1])
    mat = np.array([F1[m][0] + [F1[m][1]] for m in order])
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f1ea", "#cfe0f5", "#7fb0ea", "#2a78d6", "#123d73"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=76, vmax=99)
    cols = [
        "NSL-KDD",
        "UNSW-\nNB15",
        "CIC-\nIDS2018",
        "CIC-\nIoT2023",
        "Edge-\nIIoTset",
        "TON_IoT",
        "5G-\nNIDD",
        "Mean",
    ]
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(order)), [f"{m}" for m in order])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    best = mat.max(axis=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            dark = v > 93
            w = "bold" if v == best[j] else "normal"
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight=w,
                color="white" if dark else INK,
            )
    ax.axvline(6.5, color="white", lw=3)
    ax.axhline(len(order) - 1.5, color="white", lw=3)
    for t in ax.get_yticklabels():
        if t.get_text() == "MOFedCyber":
            t.set_fontweight("bold")
    for i, m in enumerate(order):
        ax.text(7.75, i, f"rank {F1[m][2]:.1f}", va="center", fontsize=6.3, color=INK2)
    ax.set_xlim(-0.5, 8.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Macro-F1 (%)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig4_heatmap")


# ---------------------------------------------------------------- Fig: dumbbell per dataset
def fig_dumbbell():
    comp = ["FLAME", "MOO-FL-εC", "SCAFFOLD", "FedAvg"]
    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    y = np.arange(len(DATASETS))[::-1]
    for yi, j in zip(y, range(len(DATASETS))):
        lo = min(F1[m][0][j] for m in comp + ["MOFedCyber"])
        hi = max(F1[m][0][j] for m in comp + ["MOFedCyber"])
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=3.2, solid_capstyle="round", zorder=1)
    for m in comp + ["MOFedCyber"]:
        hero = m == "MOFedCyber"
        ax.scatter(
            [F1[m][0][j] for j in range(7)],
            y,
            s=34 if hero else 20,
            marker=M[m],
            color=C[m],
            edgecolor="white",
            linewidth=0.6,
            zorder=4 if hero else 3,
            label=m,
        )
    for yi, j in zip(y, range(7)):
        gap = F1["MOFedCyber"][0][j] - F1["FLAME"][0][j]
        ax.text(
            99.4,
            yi,
            f"{gap:+.1f}",
            va="center",
            ha="left",
            fontsize=6.3,
            color=C["MOFedCyber"] if gap > 0 else C["FLAME"],
            fontweight="bold",
        )
    ax.text(99.4, len(DATASETS) - 0.35, "vs FLAME", fontsize=6, color=INK2, ha="left")
    ax.set_yticks(y, DATASETS)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(80, 101.6)
    ax.set_xlabel("Macro-F1 (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.2), handletextpad=0.2, columnspacing=0.7)
    save(fig, "fig5_dumbbell")


# ---------------------------------------------------------------- Fig: rank vs mean
def fig_rank():
    fam_c = {
        "Single-objective": "#6b6a66",
        "Multi-objective": "#6250d6",
        "Privacy-preserving": "#9c4f96",
        "Byzantine-robust": "#eb6834",
        "Proposed": "#2a78d6",
    }
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    for m, (_, mean, rank, fam) in F1.items():
        hero = m == "MOFedCyber"
        ax.scatter(
            rank,
            mean,
            s=60 if hero else 26,
            color=fam_c[fam],
            marker=M[m],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        dx, dy, ha = {
            "MOFedCyber": (0.4, 0.0, "left"),
            "FLAME": (0.35, 0.35, "left"),
            "MOO-FL-εC": (-0.35, 0.15, "right"),
            "FedMGDA+": (0.35, -0.35, "left"),
            "SCAFFOLD": (0.35, 0.0, "left"),
            "Bulyan": (-0.35, 0.05, "right"),
            "FedNova": (-0.3, -0.45, "right"),
            "Krum": (0.0, 0.5, "center"),
            "FedProx": (0.0, 0.5, "center"),
            "Median": (0.35, 0.15, "left"),
            "FedAvg": (0.0, -0.75, "center"),
            "DP-FedAvg": (-0.35, 0.0, "right"),
            "LDP-Fed": (-0.35, 0.0, "right"),
        }[m]
        ax.text(
            rank + dx,
            mean + dy,
            m,
            fontsize=6.3,
            color=INK if hero else INK2,
            va="center",
            fontweight="bold" if hero else "normal",
            ha=ha,
        )
    ax.set_xlabel("Average rank over 7 datasets (lower is better)")
    ax.set_ylabel("Mean macro-F1 (%)")
    ax.set_xlim(0, 14.2)
    ax.set_ylim(83.5, 97.8)
    hand = [Line2D([], [], marker="o", ls="", color=v, label=k, ms=4.5) for k, v in fam_c.items()]
    ax.legend(handles=hand, loc="upper right", fontsize=6.2, handletextpad=0.1)
    save(fig, "fig6_rank")


# ---------------------------------------------------------------- Fig: Pareto quality
def fig_pareto():
    ms = ["FedMGDA+", "MOO-FL-εC", "MOFedCyber"]
    fig, axs = plt.subplots(1, 3, figsize=(COL1, 1.75))
    names = ["Hypervolume ↑", "IGD ↓", "Spacing ↓"]
    for k, ax in enumerate(axs):
        vals = [PARETO[m][k] for m in ms]
        ax.bar(range(3), vals, color=[C[m] for m in ms], width=0.66, edgecolor="white", lw=1)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_xticks(range(3), ["Fed-\nMGDA+", "MOO-\nFL-εC", "Ours"], fontsize=6.2)
        chg = ["+24.3%", "−31.0%", "−19.3%"][k]
        ax.text(
            2,
            vals[2] * 0.5,
            chg,
            ha="center",
            va="center",
            fontsize=6.2,
            color="white",
            fontweight="bold",
            rotation=90,
        )
        ax.set_title(names[k], fontsize=7.2, color=INK, pad=3)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig7_pareto_quality")


# ---------------------------------------------------------------- Fig: communication
def fig_comm():
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.85), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    ms = ["FedAvg", "FLAME", "MOFedCyber"]
    r = [CONV[m]["rounds"] for m in ms]
    ax.barh(range(3), r, color=[C[m] for m in ms], height=0.6, edgecolor="white")
    for i, m in enumerate(ms):
        lab = f"{r[i]}" + ("  (97% reached)" if CONV[m]["reaches_target"] else "  (plateau <97%)")
        ax.text(r[i] + 3, i, lab, va="center", fontsize=6.1, color=INK2)
    ax.set_yticks(range(3), ["FedAvg", "FLAME", "Ours"])
    ax.set_xlim(0, 290)
    ax.set_xlabel("Communication rounds")
    ax.grid(axis="y", visible=False)
    panel(ax, "a", "Rounds")
    ax = axs[1]
    v = [CONV["FedAvg"]["gb"], CONV["MOFedCyber"]["gb"]]
    ax.bar([0, 1], v, color=[C["FedAvg"], C["MOFedCyber"]], width=0.6, edgecolor="white")
    for i, x in enumerate(v):
        ax.text(i, x + 4, f"{x:.1f} GB", ha="center", fontsize=6.3, color=INK)
    ax.annotate(
        "",
        xy=(0.95, 74),
        xytext=(0.3, 178),
        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8, connectionstyle="arc3,rad=-0.25"),
    )
    ax.text(0.72, 140, "−68%", fontsize=8, fontweight="bold", color=C["MOFedCyber"])
    ax.set_xticks([0, 1], ["FedAvg", "Ours"])
    ax.set_ylim(0, 235)
    ax.set_ylabel("Cumulative traffic (GB)")
    ax.grid(axis="x", visible=False)
    panel(ax, "b", "Total traffic")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig8_communication")


# ---------------------------------------------------------------- Fig: privacy
def fig_privacy():
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.25), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    order = ["LDP-Fed", "DP-FedAvg", "MOFedCyber"]
    lineplot(ax, EPS, PRIV_F1, order)
    for m in order:
        ax.axhline(PRIV_NODP[m], color=C[m], lw=0.7, ls=(0, (3, 2)), alpha=0.8)
    ax.set_xscale("log")
    ax.set_xticks(EPS, ["0.1", "0.25", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlim(0.085, 16)
    ax.set_xlabel("Privacy budget ε (client-level, δ = $10^{-5}$)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.axvline(1, color=INK2, lw=0.6, ls=":")
    ax.text(0.94, 71.2, "operating\npoint ε = 1", fontsize=6, color=INK2, ha="right")
    h, l = ax.get_legend_handles_labels()
    h = h[::-1] + [Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)))]
    l = l[::-1] + ["same method, no DP"]
    ax.legend(h, l, loc="lower right", fontsize=6.3)
    ax.annotate("", xy=(1, 96.4), xytext=(1, 87.1), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    ax.text(1.1, 91.0, "+9.3 pp", fontsize=6.6, color=INK, ha="left", fontweight="bold")
    ax.set_ylim(70, 100)
    panel(ax, "a", "Utility across privacy budgets")
    ax = axs[1]
    x = np.arange(4)
    w = 0.36
    for k, m in enumerate(["DP-FedAvg", "MOFedCyber"]):
        ax.bar(x + (k - 0.5) * w, MIA[m], width=w, color=C[m], edgecolor="white", lw=0.8, label=m)
        for xi, v in zip(x, MIA[m]):
            off = 4.5 if (k == 1 and xi == 3) else 1.0
            ax.text(xi + (k - 0.5) * w, v + off, f"{v:.1f}", ha="center", fontsize=5.8, color=INK2)
    ax.axhline(50, color=INK, lw=0.8, ls="--")
    ax.set_xticks(x, [f"{e}\nσz={s}" for e, s in zip(MIA_EPS, SIGMA_Z)], fontsize=6.2)
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel("Membership-inference success (%)")
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    ax.legend(
        h + [Line2D([], [], color=INK, lw=0.8, ls="--")], l + ["50% = random guessing"], loc="upper right"
    )
    panel(ax, "b", "Measured leakage")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig9_privacy")


# ---------------------------------------------------------------- Fig: Byzantine
def fig_byz():
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ax.axvspan(20, 41, color="#f6e9e2", lw=0, zorder=0)
    ax.text(20.6, 28, "beyond Bulyan's guarantee\n($|S_t| < 4b+3$)", fontsize=6, color="#a2492a", va="bottom")
    order = ["FedAvg", "Median", "Krum", "Bulyan", "FLAME", "MOFedCyber"]
    lineplot(ax, BYZ_X, BYZ, order, ms=3.6)
    ax.axhline(94, color=C["MOFedCyber"], lw=0.6, ls=":")
    ax.set_xlim(-1, 41)
    ax.set_ylim(25, 101)
    ax.set_xlabel("Byzantine clients per cohort (%), label flipping")
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=0.8, handlelength=1.6)
    save(fig, "fig10_byzantine")


# ---------------------------------------------------------------- Fig: dropout
def fig_dropout():
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    order = ["FedAvg", "SCAFFOLD", "FLAME", "MOFedCyber"]
    for m in order:
        y, s = DROP[m]
        hero = m == "MOFedCyber"
        ax.errorbar(
            DROP_X,
            y,
            yerr=s,
            color=C[m],
            marker=M[m],
            lw=2.0 if hero else 1.2,
            ms=4.6 if hero else 4,
            capsize=2,
            elinewidth=0.7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=m,
            zorder=5 if hero else 3,
        )
    for m in DROP:
        loss = DROP[m][0][4] - DROP[m][0][0]
        ax.text(
            51.5, DROP[m][0][5], f"{loss:+.1f} pp", fontsize=6.1, color=C[m], fontweight="bold", va="center"
        )
    ax.text(51.5, 99.2, "loss at\n40% drop", fontsize=5.8, color=INK2, va="top")
    ax.axvline(40, color=INK2, lw=0.5, ls=":")
    ax.set_xlim(-2, 60)
    ax.set_xticks(DROP_X)
    ax.set_ylim(76, 100)
    ax.set_xlabel("Per-round client dropout (%)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=0.8, handlelength=1.6)
    save(fig, "fig11_dropout")


# ---------------------------------------------------------------- Fig: scalability
def fig_scal():
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.1), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    lineplot(ax, SCAL_N, SCAL, ["FedAvg", "FLAME", "MOFedCyber"])
    for m, dy in [("MOFedCyber", 1.3), ("FLAME", -1.3), ("FedAvg", 0.0)]:
        b, a = np.polyfit(SCAL_N, SCAL[m], 1)
        ax.text(
            512,
            SCAL[m][-1] + dy,
            f"{m}: {b*100:.2f} s per 100 clients",
            fontsize=6.0,
            color=C[m],
            va="center",
        )
    ax.text(512, 6.0, "linear fits, $R^2 > 0.99$", fontsize=6.0, color=INK2)
    ax.set_xlim(0, 700)
    ax.set_xticks([10, 100, 200, 300, 400, 500])
    ax.set_xlabel("Federation size N (clients)")
    ax.set_ylabel("Wall-clock time per round (s)")
    panel(ax, "a", "Round time grows linearly")
    ax = axs[1]
    x = np.arange(len(SCAL_N))
    ax.bar(x, NSGA_SHARE, color=C["MOO-FL-εC"], width=0.62, edgecolor="white")
    for xi, v in zip(x, NSGA_SHARE):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(x, SCAL_N, fontsize=6.2)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Federation size N (clients)")
    ax.set_ylabel("NSGA-III share of round time (%)")
    ax.grid(axis="x", visible=False)
    panel(ax, "b", "Multi-objective overhead")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig12_scalability")


# ---------------------------------------------------------------- Fig: energy
def fig_energy():
    fig, ax = plt.subplots(figsize=(COL1, 1.95))
    ms = list(ENERGY)[::-1]
    comps = ["Computation", "Radio (communication)", "Idle / straggler wait"]
    cc = ["#2a78d6", "#eb6834", "#c4c2bb"]
    y = np.arange(len(ms))
    left = np.zeros(len(ms))
    for k in range(3):
        v = np.array([ENERGY[m][k] for m in ms])
        ax.barh(y, v, left=left, color=cc[k], height=0.62, edgecolor="white", lw=1.2, label=comps[k])
        for yi, l, vv in zip(y, left, v):
            if vv >= 1.3:
                ax.text(l + vv / 2, yi, f"{vv:.1f}", ha="center", va="center", fontsize=6, color="white")
        left += v
    for yi, t, m in zip(y, left, ms):
        ax.text(
            t + 0.2,
            yi,
            f"{t:.1f} kJ",
            va="center",
            fontsize=6.3,
            color=INK,
            fontweight="bold" if m == "MOFedCyber" else "normal",
        )
    ax.set_yticks(y, [("Ours" if m == "MOFedCyber" else m) for m in ms])
    ax.set_xlim(0, 15.8)
    ax.set_xlabel("Energy per round per client (kJ), Jetson Nano profile")
    ax.grid(axis="y", visible=False)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.27),
        fontsize=6.1,
        handlelength=0.9,
        columnspacing=0.8,
    )
    save(fig, "fig13_energy")


# ---------------------------------------------------------------- Fig: non-IID
def fig_noniid():
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    order = ["FedAvg", "SCAFFOLD", "FedMGDA+", "FLAME", "MOFedCyber"]
    lineplot(ax, ALPHA, NONIID, order)
    ax.set_xscale("log")
    ax.set_xticks(ALPHA, ["0.05", "0.1", "0.3", "0.5", "1", "5"])
    ax.minorticks_off()
    ax.set_xlabel("Dirichlet concentration α (left = more non-IID)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.annotate(
        "", xy=(0.05, 94.9), xytext=(0.05, 76.8), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7)
    )
    ax.text(0.33, 79.5, "gap to FedAvg at α = 0.05:\n+18.1 pp (94.9 vs 76.8)", fontsize=6.2, color=INK)
    ax.set_ylim(75, 100)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=0.8, handlelength=1.6)
    save(fig, "fig14_noniid")


# ---------------------------------------------------------------- Fig: heads
def fig_heads():
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    s = np.array([h[2] for h in HEADS])
    d = np.array([h[3] for h in HEADS])
    b, a = np.polyfit(s, d, 1)
    r = np.corrcoef(s, d)[0, 1]
    xx = np.linspace(0.67, 0.88, 10)
    ax.plot(xx, a + b * xx, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.scatter(s, d, s=34, color=C["MOFedCyber"], edgecolor="white", linewidth=0.7, zorder=3)
    offs = {
        "H7": (-0.004, 0.08, "right"),
        "H5": (0.004, -0.12, "left"),
        "H1": (-0.004, 0.06, "right"),
        "H3": (0.004, -0.08, "left"),
        "H8": (0.004, -0.08, "left"),
        "H2": (-0.004, 0.06, "right"),
        "H6": (0.007, -0.1, "left"),
        "H4": (0.008, -0.05, "left"),
    }
    for h, sp, sc, dr in HEADS:
        ox, oy, ha = offs[h]
        ax.text(sc + ox, dr + oy, f"{h} {sp}", fontsize=5.9, color=INK2, ha=ha, va="center")
    ax.text(0.632, 3.62, f"Pearson r = {r:.3f}", fontsize=6.4, color=INK)
    ax.set_xlim(0.62, 0.93)
    ax.set_ylim(1.9, 3.95)
    ax.set_xlabel("Head specialization score")
    ax.set_ylabel("Macro-F1 drop when head is removed (pp)")
    save(fig, "fig15_heads")
    return r


# ---------------------------------------------------------------- Fig: ablation
def fig_ablation():
    fig, ax = plt.subplots(figsize=(COL1, 2.05))
    rows = [(n, f, d) for n, f, d in ABLATION if d != 0][::-1] + [
        ("No DP (upper bound, not an ablation)", NO_DP, round(NO_DP - 96.6, 1))
    ]
    rows = rows[::-1]
    y = np.arange(len(rows))[::-1]
    for yi, (n, f, d) in zip(y, rows):
        if n.startswith("No DP"):
            ax.barh(yi, d, color="white", edgecolor=MUTED, hatch="////", height=0.6, lw=0.6)
            ax.text(d + 0.15, yi, f"+{d:.1f} pp  ({f:.1f}%)", va="center", fontsize=6.2, color=INK2)
        else:
            ax.barh(yi, d, color="#a9a8a2", height=0.6, edgecolor="white")
            ax.text(
                d - 0.15,
                yi,
                f"{d:+.1f} pp  ({f:.1f}%)".replace("-", "\u2212"),
                va="center",
                ha="right",
                fontsize=6.2,
                color=INK2,
            )
    ax.axvline(0, color=INK, lw=0.9)
    ax.text(
        0.15, y[-1] - 0.75, "full MOFedCyber = 96.6%", fontsize=6.2, color=C["MOFedCyber"], fontweight="bold"
    )
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(-13.5, 3.2)
    ax.set_ylim(y[-1] - 1.1, y[0] + 0.5)
    ax.set_xlabel("Change in mean macro-F1 vs. full model (pp), 10% Byzantine")
    ax.grid(axis="y", visible=False)
    save(fig, "fig16_ablation")


if __name__ == "__main__":
    fig_datasets()
    fig_heatmap()
    fig_dumbbell()
    fig_rank()
    fig_pareto()
    fig_comm()
    fig_privacy()
    fig_byz()
    fig_dropout()
    fig_scal()
    fig_energy()
    fig_noniid()
    r = fig_heads()
    fig_ablation()
    print("heads r =", round(r, 3))
    print("done")

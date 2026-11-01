"""Figure conventions shared by make_figures.py and make_visuals.py.

Color carries the identity of only three methods, the first three slots of the
validated categorical palette (they pass all-pairs CVD and normal-vision
checks on white): MR-ScaledGD blue, ScaledGD orange, APGD aqua. Every other
method is a neutral gray told apart by its dash pattern, marker and a legend
entry, so no figure relies on color alone.
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fontTools warns about the timestamps of the bundled fonts it subsets into each PDF
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"

STYLE = {
    "mr": dict(color=BLUE, ls="-", marker="o", label="MR-ScaledGD (ours)"),
    "scaledgd": dict(color=ORANGE, ls="-", marker="s", label="ScaledGD"),
    "apgd": dict(color=AQUA, ls="-", marker="^", label="APGD"),
    "gd": dict(color=INK3, ls=":", marker="", label="GD"),
    "tnn": dict(color=INK2, ls="--", marker="x", label="TNN"),
    "tctf": dict(color=INK3, ls="-.", marker="d", label="TCTF"),
    "svp": dict(color=INK2, ls=":", marker="v", label="Budgeted SVP"),
    "mr-oracle": dict(color=INK2, ls="--", marker="", label="MR, no reallocation"),
    "scaledgd-damp": dict(color=INK2, ls="-.", marker="", label="ScaledGD, damped"),
    "tubal": dict(color=ORANGE, ls="-", marker="s", label="tubal rank (uniform)"),
    "multi": dict(color=BLUE, ls="-", marker="o", label="multi-rank (budgeted)"),
}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def line(ax, x, y, key, label=None, markevery=None, **kw):
    st = dict(STYLE[key])
    default_label = st.pop("label")
    st.update(kw)
    return ax.plot(x, y, label=default_label if label is None else label, markevery=markevery,
                   markerfacecolor="white", markeredgewidth=0.9, **st)


def finish(ax, grid_y=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_y:
        ax.grid(True, axis="y")
    ax.set_axisbelow(True)

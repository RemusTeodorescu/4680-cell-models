"""Paper 1.2 / S32K5 project: reference 23s9p module of 207 BYD FC4680 cells.

    python fig_module_23s9p.py  ->  figures/fig_module_23s9p.{pdf,png}

A reference layout, not a specific product. Chosen to match one of the four
modules of the Tesla Model Y 4680 structural pack (828 cells, 92s9p, i.e.
4 x 23s9p = 207 cells per module), with the sensing assumed for this work:
  * 2 NTCs per module, at the coolant inlet and outlet (Tesla Model S practice:
    2 per module at the coolant inlet and outlet sides);
  * one voltage per series group (23 groups) and the module current.

(a) Top view to scale: 9 rows x 23 columns, hexagonal (staggered) packing,
    one parallel group per column, serpentine cooling ribbons between rows,
    busbars per column, coolant inlet/outlet at one end (contra-flow).
(b) Electrical and sensing view: 23 groups of 9 cells in parallel in series,
    group voltage taps to the cell monitor, current sensor, NTCs, S32K5.
    One cell with a soft short (R_isc) in group 12, fed by its 8 neighbours.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon

D = 46.3                  # cell diameter, mm
NS, NP = 23, 9            # series groups (columns), cells per group (rows)
GAP_X = 1.5               # gap between cells along a row, mm
RIB = 2.5                 # cooling ribbon thickness, mm
PX = D + GAP_X            # column pitch along x
PY = D * np.sqrt(3) / 2 + RIB * 0.9   # row pitch (staggered), mm

INK, MUTE = "#1f2328", "#5d6470"
CELL, CELL_E = "#c9ccd1", "#7f8186"
GRP_A, GRP_B = "#dfe8f6", "#f6eadf"
RIBBON = "#3f8fd2"
BUS_A, BUS_B = "#c47f4f", "#9aa1a9"
NTC_C = "#d0632f"
SHORT_C = "#c0392b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def cell_xy(i_col, j_row):
    x = i_col * PX + (PX / 2 if j_row % 2 else 0.0)
    y = j_row * PY
    return x, y


def panel_a(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    W = (NS - 1) * PX + PX / 2 + D
    H = (NP - 1) * PY + D
    # housing
    ax.add_patch(FancyBboxPatch((-D / 2 - 12, -D / 2 - 12), W + 24, H + 24,
                                boxstyle="round,pad=0,rounding_size=8", fc="#f7f7f5", ec=MUTE, lw=0.8))
    # group shading (columns)
    for i in range(NS):
        xs = [cell_xy(i, j)[0] for j in range(NP)]
        ax.add_patch(Rectangle((min(xs) - D / 2 - 0.6, -D / 2 - 3), max(xs) - min(xs) + D + 1.2, H + 6,
                               fc=GRP_A if i % 2 == 0 else GRP_B, ec="none", zorder=0.5, alpha=0.9))
    # serpentine ribbons between rows j and j+1, following the stagger
    for j in range(NP - 1):
        xs, ys = [], []
        for i in range(NS):
            for jj in (j, j + 1):
                x, y = cell_xy(i, jj)
                xs.append(x)
                ys.append(y + (D / 2 + RIB / 2) * (1 if jj == j else -1))
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.plot(xs, ys, color=RIBBON, lw=1.6, zorder=1, solid_joinstyle="round")
    # cells
    for i in range(NS):
        for j in range(NP):
            x, y = cell_xy(i, j)
            short = (i == 11 and j == 4)
            ax.add_patch(Circle((x, y), D / 2, fc=CELL if not short else "#f2c6c0",
                                ec=CELL_E if not short else SHORT_C, lw=0.5 if not short else 1.4, zorder=2))
            ax.add_patch(Circle((x, y), 8.0, fc="#e9eaec", ec=CELL_E, lw=0.3, zorder=2.1))   # + terminal
    # busbars: one per column on top (+ of group i to - (can rim) of group i+1), drawn as a strip
    for i in range(NS):
        xs = [cell_xy(i, j)[0] for j in range(NP)]
        xb = np.mean(xs) + PX / 4
        ax.add_patch(Rectangle((xb - 2.0, -D / 2 + 4), 4.0, H - 8, fc=BUS_A if i % 2 else BUS_B,
                               ec="none", alpha=0.75, zorder=3))
    # group numbers
    for i in range(NS):
        x = np.mean([cell_xy(i, j)[0] for j in range(NP)])
        ax.text(x, H - D / 2 + 20, f"{i+1}", ha="center", va="bottom", fontsize=6.3, color=MUTE)
    ax.text(-D / 2 - 12, H - D / 2 + 20, "group", ha="right", va="bottom", fontsize=6.3, color=MUTE)
    # coolant manifold at the left end, inlet bottom, outlet top (contra-flow)
    ax.add_patch(Rectangle((-D / 2 - 40, -D / 2 - 6), 22, H + 12, fc="#d7e9f7", ec=RIBBON, lw=0.8))
    ax.annotate("", xy=(-D / 2 - 40, 0), xytext=(-D / 2 - 90, 0),
                arrowprops=dict(arrowstyle="-|>", color=RIBBON, lw=1.0))
    ax.text(-D / 2 - 92, 0, "coolant\nin", ha="right", va="center", fontsize=7, color=RIBBON)
    ax.annotate("", xy=(-D / 2 - 90, H - D), xytext=(-D / 2 - 40, H - D),
                arrowprops=dict(arrowstyle="-|>", color=RIBBON, lw=1.0))
    ax.text(-D / 2 - 92, H - D, "coolant\nout", ha="right", va="center", fontsize=7, color=RIBBON)
    # NTCs at inlet and outlet cells
    for (i, j, lab) in ((0, 0, "NTC 1\n(inlet)"), (0, NP - 1, "NTC 2\n(outlet)")):
        x, y = cell_xy(i, j)
        ax.plot(x - D / 2 + 3, y, marker="o", ms=6, mfc=NTC_C, mec="white", mew=0.8, zorder=6)
        ax.annotate(lab, xy=(x - D / 2 + 3, y), xytext=(-D / 2 - 92, y + (75 if j == 0 else -75)), fontsize=7,
                    color=NTC_C, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=NTC_C, lw=0.7))
    # short-circuit cell
    xs_, ys_ = cell_xy(11, 4)
    ax.annotate("soft short (example)\none cell of group 12", xy=(xs_ + D / 2, ys_), xytext=(xs_ + 330, -D / 2 - 45),
                fontsize=7, color=SHORT_C, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=SHORT_C, lw=0.8))
    # dimensions
    ax.annotate("", xy=(-D / 2, -D / 2 - 26), xytext=(W - D / 2, -D / 2 - 26),
                arrowprops=dict(arrowstyle="<->", color=MUTE, lw=0.6))
    ax.text(W / 2 - D / 2 - 200, -D / 2 - 34, f"≈ {W:.0f} mm  (23 groups, series)", ha="center", va="top", fontsize=7, color=MUTE)
    ax.annotate("", xy=(W - D / 2 + 22, -D / 2), xytext=(W - D / 2 + 22, H - D / 2),
                arrowprops=dict(arrowstyle="<->", color=MUTE, lw=0.6))
    ax.text(W - D / 2 + 28, H / 2 - D / 2, f"≈ {H:.0f} mm\n(9 cells\nin parallel)", ha="left", va="center", fontsize=7, color=MUTE)
    ax.set_xlim(-D / 2 - 175, W - D / 2 + 125)
    ax.set_ylim(-D / 2 - 75, H - D / 2 + 70)
    ax.text(-D / 2 - 175, H - D / 2 + 68, "a", fontsize=11, weight="bold", va="top")
    ax.text(-D / 2 - 150, H - D / 2 + 68, "top view to scale: 207 cells, 23s9p, staggered, cooling ribbons between rows",
            fontsize=8.5, va="top")
    return W, H


def panel_b(ax):
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    groups_drawn = [1, 2, 3, None, 12, None, 22, 23]
    x0, dx = 6, 10.2
    xs = []
    for k, g in enumerate(groups_drawn):
        x = x0 + k * dx
        xs.append(x)
        if g is None:
            ax.text(x + 3, 22, "…", ha="center", va="center", fontsize=12, color=MUTE)
            continue
        # 9 cells in parallel, drawn as 9 small battery symbols stacked
        for p in range(9):
            y = 13 + p * 2.2
            red = (g == 12 and p == 4)
            ax.plot([x, x + 6], [y, y], color=SHORT_C if red else INK, lw=0.6)
            ax.add_patch(Rectangle((x + 2.4, y - 0.7), 0.4, 1.4, fc=INK, ec="none"))
            ax.add_patch(Rectangle((x + 3.3, y - 0.4), 0.4, 0.8, fc=INK, ec="none"))
            if red:
                ax.plot([x + 6.6, x + 6.6], [y - 0.6, y + 0.6], color=SHORT_C, lw=1.2)
                ax.text(x + 7.2, y, "R_isc", fontsize=6, color=SHORT_C, va="center")
        ax.plot([x, x], [13, 13 + 8 * 2.2], color=INK, lw=0.8)
        ax.plot([x + 6, x + 6], [13, 13 + 8 * 2.2], color=INK, lw=0.8)
        ax.text(x + 3, 10.5, f"G{g}", ha="center", fontsize=7)
        # voltage tap
        ax.plot([x + 6, x + 6], [13 + 8 * 2.2, 35], color="#7d5ba6", lw=0.6, ls=(0, (2, 1.5)))
    # series connections
    for k in range(len(xs) - 1):
        ax.plot([xs[k] + 6, xs[k + 1]], [13, 13], color=INK, lw=0.8)
    # cell monitor and S32K5
    ax.add_patch(FancyBboxPatch((4, 35), 82, 3.5, boxstyle="round,pad=0.3", fc="#efe8f7", ec="#7d5ba6", lw=0.8))
    ax.text(45, 36.75, "cell monitor: 23 group voltages (+ group impedance, optional)", ha="center", va="center", fontsize=7, color="#4b3570")
    ax.add_patch(FancyBboxPatch((89, 26), 9, 12.5, boxstyle="round,pad=0.3", fc="#e6f0e9", ec="#2f9d6e", lw=0.8))
    ax.text(93.5, 32.3, "NXP\nS32K5\nBMS", ha="center", va="center", fontsize=7, color="#1f6b4a")
    ax.plot([86.3, 88.7], [36.7, 36.7], color="#7d5ba6", lw=0.8)
    # current sensor
    ax.add_patch(Circle((92, 13), 2.0, fc="white", ec=INK, lw=0.7))
    ax.text(92, 13, "I", ha="center", va="center", fontsize=7)
    ax.plot([xs[-1] + 6, 90], [13, 13], color=INK, lw=0.8)
    ax.plot([92, 92], [15, 25.7], color=INK, lw=0.6, ls=(0, (2, 1.5)))
    # NTCs
    ax.plot(4, 6, marker="o", ms=6, mfc=NTC_C, mec="white")
    ax.text(6.5, 6, "2 NTCs (coolant inlet, outlet) → S32K5", va="center", fontsize=7, color=NTC_C)
    ax.text(50, 3, "A short in one cell of a group is fed by the whole group: the group voltage drifts\n"
            "against the other 22 groups at rest, by about 1/9 of what a single cell would show.",
            ha="center", va="center", fontsize=7, color=SHORT_C)
    ax.text(0, 40, "b", fontsize=11, weight="bold", va="top")
    ax.text(3, 40, "electrical and sensing view", fontsize=8.5, va="top")


def main():
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.04)
    W, H = panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    fig.text(0.01, 0.005, "Reference module, not a specific product: 23s9p as one of four modules of the Tesla Model Y 4680 pack (92s9p, 828 cells); "
             "BYD FC4680 cells (∅46.3 × 80.2 mm).\nSensing as assumed in this work. Ribbon, busbar and manifold geometry schematic; cell pitch and packing to scale.",
             fontsize=6.2, color=MUTE, va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.07)
    for ext in ("pdf", "png"):
        fig.savefig(f"../figures/fig_module_23s9p.{ext}", dpi=220)
    print("module", round(W), "x", round(H), "mm")


if __name__ == "__main__":
    main()

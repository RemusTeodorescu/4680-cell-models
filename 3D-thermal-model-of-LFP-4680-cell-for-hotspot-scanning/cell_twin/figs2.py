"""Figures for the four follow-up points of paper 1.2: MT-TDFN coupled runs,
refined patch at the short, response library (reads out/*.json, out/*.npz)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from figs import OUT, FIG, _slice_rz

CASE = {"mt_B_no_short": ("calibrated foils (B), no short", "#1f4e79", "-"),
        "mt_B_short_3ohm_Al-anode_mid": ("calibrated foils (B), 3 Ω Al-anode short", "#c0392b", "-"),
        "mt_A_literal_no_short": ("literal foils (A), no short", "#7f8c8d", "--")}


def fig_mt():
    R = json.loads((OUT / "coupled_mt.json").read_text())["runs"]
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.32)
    axV, axP = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for name, (lab, c, ls) in CASE.items():
        h = R[name]["hist"]
        t = np.array([x["t"] for x in h]) / 60
        axV.plot(t, [x["V"] for x in h], color=c, ls=ls, lw=1.3, label=lab)
        axP.plot(t, [x["max_internal"] for x in h], color=c, ls=ls, lw=1.3, label="hottest inside")
        axP.plot(t, [x["max_side"] for x in h], color=c, ls=":", lw=1.1, label="hottest on can side")
    axV.set_xlabel("time [min]"); axV.set_ylabel("terminal voltage [V]")
    axV.set_title("(a) 2C discharge from 80 % SOC, MT-TDFN columns")
    axV.legend(frameon=False, fontsize=5.8); axV.grid(alpha=0.3)
    axP.set_xlabel("time [min]"); axP.set_ylabel("rise above ambient [K]"); axP.set_yscale("log")
    axP.set_title("(b) solid: hottest inside, dotted: hottest on the can side"); axP.grid(alpha=0.3, which="both")
    for k, name in enumerate(("mt_B_no_short", "mt_A_literal_no_short")):
        d = dict(np.load(OUT / f"coupled_{name}.npz"))
        ax = fig.add_subplot(gs[1, k])
        m = ax.pcolormesh(d["s_f"] / 1000, d["z_f"], -d["j"].T, cmap="viridis", shading="flat",
                          vmin=0, vmax=250)
        ax.set_title(f"({'cd'[k]}) discharge current density, 600 s,\n{CASE[name][0]}")
        ax.set_xlabel("s along the cathode [m]"); ax.set_ylabel("z [mm]")
        fig.colorbar(m, ax=ax, label=r"A/m$^2$", pad=0.02)
    a = dict(np.load(OUT / "coupled_mt_B_no_short.npz"))
    b = dict(np.load(OUT / "coupled_mt_B_short_3ohm_Al-anode_mid.npz"))
    ax = fig.add_subplot(gs[2, 0])
    m = _slice_rz(ax, dict(T=a["T3"], r_f=a["r_f"], z_f=a["zg_f"], th_f=a["th_f"]),
                  "(e) B, no short: r-z section, 600 s")
    ax.set_xlim(0, 23.2); fig.colorbar(m, ax=ax, label="K above ambient", shrink=0.8)
    ax = fig.add_subplot(gs[2, 1])
    m = _slice_rz(ax, dict(T=b["T3"] - a["T3"], r_f=a["r_f"], z_f=a["zg_f"], th_f=a["th_f"]),
                  "(f) extra rise from the 3 Ω short, 600 s\n(2 mm grid; peak set by the mesh)")
    ax.set_xlim(0, 23.2); fig.colorbar(m, ax=ax, label="K", shrink=0.8)
    fig.savefig(FIG / "fig_mt_coupled.pdf"); fig.savefig(FIG / "fig_mt_coupled.png", dpi=200)
    plt.close(fig)


def fig_patch():
    L = [json.loads(l) for l in (OUT / "patch_study.log").read_text().splitlines() if l.startswith("{")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    conv = sorted([x for x in L if x["w_mm"] == 1.0], key=lambda x: -x["h_mm"])
    ax = axs[0]
    for key, lab, c in (("peak_1s", "1 s", "#7fb3d5"), ("peak_10s", "10 s", "#2e86c1"),
                        ("peak_60s", "60 s", "#1f4e79"), ("peak_steady", "steady", "#c0392b")):
        ax.plot([x["h_mm"] for x in conv], [x[key] for x in conv], "o-", color=c, label=lab)
    hs = [x["h_mm"] for x in conv]
    ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([f"{v:g}" for v in hs])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.invert_xaxis(); ax.set_ylim(0, 40)
    ax.set_xlabel("in-plane cell size at the short [mm]"); ax.set_ylabel("hottest point [K/W]")
    ax.set_title("(a) convergence, 1 × 1 mm short at r = 13 mm"); ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.3, which="both")
    best = {}
    for x in L:
        if x["w_mm"] not in best or x["h_mm"] < best[x["w_mm"]]["h_mm"]:
            best[x["w_mm"]] = x
    w = sorted(best)
    ax = axs[1]
    for key, lab, c in (("peak_1s", "1 s", "#7fb3d5"), ("peak_60s", "60 s", "#1f4e79"),
                        ("peak_steady", "steady", "#c0392b")):
        ax.plot(w, [best[v][key] for v in w], "o-", color=c, label=lab)
    ax.plot(w, [best[v]["side_steady"] for v in w], "^:", color="k", label="can side, steady")
    ax.set_xscale("log"); ax.set_xticks(w); ax.set_xticklabels([f"{v:g}" for v in w])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("footprint of the short, w × w [mm]"); ax.set_ylabel("K/W")
    ax.set_title("(b) hottest point vs footprint, finest grid"); ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig_patch_study.pdf"); fig.savefig(FIG / "fig_patch_study.png", dpi=200)
    plt.close(fig)


def fig_library():
    L = json.loads((OUT / "response_library.json").read_text())
    t = np.array(L["t"])
    hc = {"base_end": "#1f4e79", "lower": "#2e86c1", "mid": "#7f8c8d", "upper": "#e67e22", "cap_end": "#c0392b"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for hgt, c in hc.items():
        E = sorted([e for e in L["entries"] if e["height"] == hgt and e["angle"] == "away_from_tab"],
                   key=lambda e: e["r"])
        r = [e["r"] for e in E]
        axs[0].plot(r, [np.interp(60, t, e["peak"]) for e in E], "o-", ms=3, color=c, label=hgt.replace("_", " "))
        axs[1].plot(r, [np.interp(600, t, e["side_max"]) for e in E], "o-", ms=3, color=c)
        axs[2].plot(r, [np.interp(3600, t, e["side_max"]) for e in E], "o-", ms=3, color=c)
    for hgt, mk in (("cap_end", "x"),):
        E = sorted([e for e in L["entries"] if e["height"] == hgt and e["angle"] == "at_tab_angle"], key=lambda e: e["r"])
        if E:
            axs[0].plot([e["r"] for e in E], [np.interp(60, t, e["peak"]) for e in E], mk + "--", color="#c0392b",
                        ms=4, lw=0.8, label="cap end, at a tab")
    for ax, ttl in zip(axs, ("(a) hottest inside, 60 s", "(b) hottest on the can side, 600 s",
                             "(c) hottest on the can side, 3600 s")):
        ax.set_xlabel("radius of the short [mm]"); ax.set_title(ttl); ax.grid(alpha=0.3)
    axs[0].set_ylabel("K per W (2 mm short, coarse grid)"); axs[0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_response_library.pdf"); fig.savefig(FIG / "fig_response_library.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_mt(); fig_patch(); fig_library()

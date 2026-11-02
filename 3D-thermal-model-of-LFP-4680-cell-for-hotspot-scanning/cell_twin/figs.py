"""Figures for steps 2-4 of paper 1.2 (reads out/*.json, out/*.npz)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "out"
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"core": "#1f4e79", "mid_radius": "#2e86c1", "near_side": "#7fb3d5",
       "near_cap": "#c0392b", "at_cap_tab": "#e67e22"}
LAB = {"core": "core (r = 4 mm)", "mid_radius": "mid (r = 13 mm)", "near_side": "near side (r = 22 mm)",
       "near_cap": "near cap (r = 13 mm, 2 mm below roll top)", "at_cap_tab": "at a cathode tab (r = 16 mm)"}


def _slice_rz(ax, d, title, th_idx=None, vmax=None):
    T = d["T"]; rf, zf, thf = d["r_f"], d["z_f"], d["th_f"]
    if th_idx is None:
        th_idx = np.unravel_index(np.argmax(T), T.shape)[1]
    S = T[:, th_idx, :]
    m = ax.pcolormesh(rf, zf, S.T, cmap="inferno", vmin=0, vmax=vmax or S.max(), shading="flat")
    ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm]"); ax.set_title(title)
    ax.set_aspect("auto")
    return m


def _slice_rt(ax, d, title, vmax=None):
    T = d["T"]; rf, thf = d["r_f"], d["th_f"]
    iz = np.unravel_index(np.argmax(T), T.shape)[2]
    S = T[:, :, iz]
    TH, R = np.meshgrid(thf, rf)
    X, Y = R * np.cos(TH), R * np.sin(TH)
    m = ax.pcolormesh(X, Y, S, cmap="inferno", vmin=0, vmax=vmax or S.max(), shading="flat")
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title(title)
    return m


def fig_short():
    p = OUT / "short_study.json"
    if not p.exists():
        return
    S = json.loads(p.read_text())
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.62)
    # (a) response per W
    ax = fig.add_subplot(gs[0, :2])
    for name, d in S["per_W"].items():
        t = np.array(d["t"])
        ax.plot(t, d["max_internal"], color=COL[name], lw=1.4, label=LAB[name])
        ax.plot(t, d["max_side"], color=COL[name], lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_xlim(0.5, 1800)
    ax.set_xlabel("time after onset [s]"); ax.set_ylabel(r"$\Delta T$ per watt of short [K/W]")
    ax.set_title("(a) rise per watt of short; solid: hottest point inside, dashed: hottest point on the can side")
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none", fontsize=6)
    ax.grid(alpha=0.3, which="both")
    # (b) convergence
    ax = fig.add_subplot(gs[0, 2])
    ns, a60, ast = [], [], []
    for ref, c in sorted(S["convergence"].items(), key=lambda kv: kv[1]["n"]):
        ns.append(c["n"]); a60.append(c["res"]["mid_radius"]["60"]["max_internal"])
        ast.append(c["res"]["mid_radius"]["steady"]["max_internal"])
    ax.plot(ns, a60, "o-", color="#2e86c1", label="peak 60 s, uniform refinement")
    ax.plot(ns, ast, "s-", color="#c0392b", label="peak steady, uniform refinement")
    pw = OUT / "perwrap.json"
    if pw.exists():
        P = json.loads(pw.read_text())[1:]
        ax.plot([p["n"] for p in P], [p["peak_60s"] for p in P], "o--", color="#2e86c1", mfc="w",
                label="peak 60 s, 1-2 cells per wrap")
        ax.plot([p["n"] for p in P], [p["steady_peak"] for p in P], "s--", color="#c0392b", mfc="w",
                label="peak steady, 1-2 cells per wrap")
        ax.plot([p["n"] for p in P], [p["side_60s"] * 10 for p in P], "^:", color="k", mfc="w",
                label="can side 60 s (x10)")
    ax.set_xscale("log")
    ax.set_xlabel("cells"); ax.set_ylabel("K/W"); ax.set_title("(b) mesh convergence,\nmid-radius short")
    ax.legend(frameon=False, fontsize=5.2, loc="lower right"); ax.grid(alpha=0.3); ax.set_ylim(0, 22)
    # (c, d) snapshots
    for k, (tag, col) in enumerate((("60s", 0), ("600s", 1))):
        f = OUT / f"snapshot_mid_{tag}.npz"
        if f.exists():
            d = dict(np.load(f))
            ax = fig.add_subplot(gs[1, col])
            m = _slice_rz(ax, d, f"({'cd'[k]}) r-z section, {tag[:-1]} s")
            ax.set_xlim(0, 23.2); fig.colorbar(m, ax=ax, label="K/W", shrink=0.8)
    f = OUT / "snapshot_mid_60s.npz"
    if f.exists():
        d = dict(np.load(f))
        ax = fig.add_subplot(gs[1, 2])
        m = _slice_rt(ax, d, "(e) r-θ section, 60 s")
        fig.colorbar(m, ax=ax, label="K/W", shrink=0.8)
    fig.savefig(FIG / "fig_short_response.pdf"); fig.savefig(FIG / "fig_short_response.png", dpi=200)
    plt.close(fig)


def detectability_table():
    p = OUT / "short_study.json"
    if not p.exists():
        return None
    S = json.loads(p.read_text())
    rows = []
    P = {"100 ohm (0.11 W)": 0.11, "33 ohm (0.33 W)": 0.33, "10 ohm (1.1 W)": 1.1, "3 ohm (3.6 W)": 3.6}
    for name, d in S["per_W"].items():
        t = np.array(d["t"]); side = np.array(d["max_side"]); core = np.array(d["max_internal"])
        row = dict(location=name, steady_internal_K_per_W=d["steady"]["max_internal"],
                   steady_side_K_per_W=d["steady"]["max_side"],
                   internal_60s_K_per_W=float(np.interp(60, t, core)),
                   side_60s_K_per_W=float(np.interp(60, t, side)),
                   side_600s_K_per_W=float(np.interp(600, t, side)))
        for lab, pw in P.items():
            reach = np.nonzero(pw * side >= 0.1)[0]
            row[f"t_side_0.1K {lab}"] = float(t[reach[0]]) if len(reach) else None
        rows.append(row)
    (OUT / "detectability.json").write_text(json.dumps(rows, indent=1))
    return rows


def fig_uniform():
    p = OUT / "uniform_study.json"
    if not p.exists():
        return
    U = json.loads(p.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, cr in zip(axs, ("1C", "2C")):
        for foam, ls in (("bare", "--"), ("foam", "-")):
            d = U[f"{foam}_{cr}"]["transient"]
            t = np.array(d["t"]) / 60
            for k, c in (("TC_cap", "#c0392b"), ("TC_mid", "#7f8c8d"), ("TC_base", "#1f4e79")):
                ax.plot(t, d[k], color=c, ls=ls, lw=1.2,
                        label={"TC_cap": "T1", "TC_mid": "T2", "TC_base": "T3"}[k] + f" ({foam})")
            ax.plot(t, d["cap_centre"], color="#e67e22", ls=ls, lw=1.2, label=f"T4 cap end face ({foam})")
            ax.plot(t, d["core_max"], color="k", ls=ls, lw=0.8, label=f"core max ({foam})")
        ax.set_title(f"({'ab'[cr == '2C']}) {cr} discharge, uniform heat, {U[f'bare_{cr}']['P_W']:.1f} W")
        ax.set_xlabel("time [min]"); ax.grid(alpha=0.3)
    axs[0].set_ylabel(r"rise above ambient [K]")
    axs[0].legend(frameon=False, ncol=2, fontsize=5.5, loc="upper left")
    fig.suptitle("Foam cap on the positive end: T1 1 mm from the positive end, T3 1 mm from the negative end, dashed bare, solid with cap", fontsize=7.5, y=1.0)
    fig.savefig(FIG / "fig_foam_cap.pdf"); fig.savefig(FIG / "fig_foam_cap.png", dpi=200)
    plt.close(fig)


def fig_coupled():
    p = OUT / "coupled.json"
    if not p.exists():
        return
    R = json.loads(p.read_text())
    a = dict(np.load(OUT / "coupled_2C_no_short.npz"))
    b = dict(np.load(OUT / "coupled_2C_short_3ohm_mid.npz"))
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.3)
    s_f, z_f = a["s_f"] / 1000, a["z_f"]
    ax = fig.add_subplot(gs[0, :])
    m = ax.pcolormesh(s_f, z_f, -a["j"].T, cmap="viridis", shading="flat")
    ax.set_title("(a) discharge current density on the unrolled cathode, 2C, 600 s "
                 "(cathode tabs at the top edge, anode tabs at the bottom edge)")
    ax.set_xlabel("s, position along the cathode [m]"); ax.set_ylabel("z [mm]")
    fig.colorbar(m, ax=ax, label=r"A/m$^2$", pad=0.01)
    from geom3d import Cell
    for t in Cell().tabs():
        y = z_f[-1] if t["kind"] == "cathode" else z_f[0]
        ax.plot(t["s"] / 1000, y, marker="v" if t["kind"] == "cathode" else "^", ms=6,
                color="w" if t["live"] else "r", mec="k", clip_on=False)
    ax = fig.add_subplot(gs[1, :])
    from matplotlib.colors import LogNorm
    m = ax.pcolormesh(s_f, z_f, a["q_sheet"].T, cmap="magma", shading="flat",
                      norm=LogNorm(vmin=max(a["q_sheet"].min(), 1.0), vmax=a["q_sheet"].max()))
    ax.set_title("(b) heat generated per unit sheet area, local + foil ohmic (log scale)")
    ax.set_xlabel("s [m]"); ax.set_ylabel("z [mm]")
    fig.colorbar(m, ax=ax, label=r"W/m$^2$", pad=0.01)
    ax = fig.add_subplot(gs[2, 0])
    m = _slice_rz(ax, dict(T=a["T3"], r_f=a["r_f"], z_f=a["zg_f"], th_f=a["th_f"]),
                  "(c) 2C, no short: r-z section at the hottest angle, 600 s")
    ax.set_xlim(0, 23.2); fig.colorbar(m, ax=ax, label="K above ambient", shrink=0.8)
    ax = fig.add_subplot(gs[2, 1])
    d = b["T3"] - a["T3"]
    m = _slice_rz(ax, dict(T=d, r_f=a["r_f"], z_f=a["zg_f"], th_f=a["th_f"]),
                  "(d) extra rise from a 3 ohm short, 600 s")
    ax.set_xlim(0, 23.2)
    fig.colorbar(m, ax=ax, label="K", shrink=0.8)
    fig.savefig(FIG / "fig_coupled_sheet.pdf"); fig.savefig(FIG / "fig_coupled_sheet.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_short(); print(json.dumps(detectability_table(), indent=1)); fig_uniform(); fig_coupled()

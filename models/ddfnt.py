"""DDFNT: dynamic DFNT -- the 2-D DFN+T slab driven by a current profile.

Two things make DFNT a constant-current model, and this module removes both
without touching its physics:

1. The particle closure.  Gu & Wang's Eq. 14 slaves the particle surface to
   the local current instantly.  `Params(particle_closure="quartic")` replaces
   it by the Subramanian et al. (2005) quartic profile (see gw2d.py), which
   adds one state per particle control volume and reduces to Eq. 14 at
   constant current.

2. The load.  `GuWang2D.run()` marches at the fixed `Params.C_rate`.  Here
   the rate is set step by step from a piecewise-constant profile, every
   change of current is a time-step boundary, the step after it restarts
   small, and delivered charge is integrated, Q = int I dt, instead of being
   computed as I*t.

    profile = [(duration_s, C_rate), ...]     # C_rate > 0 discharge, < 0 charge

Backward Euler evaluates the current at the end of each step, so a step that
ends inside a segment sees that segment's current exactly.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from gw2d import GuWang2D


def expand(profile: Sequence[Tuple[float, float]]):
    """Segment start times and rates."""
    t0, starts, rates = 0.0, [], []
    for dur, c in profile:
        starts.append(t0)
        rates.append(float(c))
        t0 += float(dur)
    return np.array(starts), np.array(rates), t0


def run_profile(mdl: GuWang2D, profile, dt_first=0.05, dt_max=5.0,
                dV_max=0.01, V_min=None, V_max=4.25, verbose=False,
                u0=None) -> Dict[str, np.ndarray]:
    """March `mdl` through `profile`.  Returns a history dict."""
    p = mdl.p
    starts, rates, t_end = expand(profile)
    edges = list(starts[1:]) + [t_end]
    V_min = p.V_cut if V_min is None else V_min

    p.C_rate = rates[0] if rates[0] != 0 else 1e-9
    u = mdl.initial_state() if u0 is None else u0.copy()
    if u0 is None and getattr(mdl, "can_node", False):
        mdl.T_can = p.T0
    wcp = mdl.rho_cp * mdl.m.V
    vol = float(mdl.m.V.sum())
    Q_nom = p.capacity_Ah * 3600.0

    hist = {k: [] for k in ("t", "I", "C_rate", "V", "Tavg", "Tmax", "Tmin",
                            "q_tot", "Q_Ah", "DOD", "Tcan")}
    t, Q, nstep, nfail = 0.0, 0.0, 0, 0
    wall = time.time()

    def record(st, V):
        hist["t"].append(t)
        hist["I"].append(p.I_cell)
        hist["C_rate"].append(p.C_rate)
        hist["V"].append(V)
        hist["Tavg"].append(float((st["T"] * wcp).sum() / wcp.sum()))
        hist["Tmax"].append(float(st["T"].max()))
        hist["Tmin"].append(float(st["T"].min()))
        hist["q_tot"].append(float((st["q"] * mdl.m.V).sum()))     # W
        hist["Q_Ah"].append(Q / 3600.0)
        hist["DOD"].append(100.0 * Q / Q_nom)
        hist["Tcan"].append(float(getattr(mdl, "T_can", np.nan))
                            if getattr(mdl, "can_node", False) else np.nan)

    # state at t = 0 under the first segment's current is the first step
    for k, (t_seg, c, t_stop) in enumerate(zip(starts, rates, edges)):
        p.C_rate = c
        dt = dt_first
        while t < t_stop - 1e-9:
            dtl = min(dt, t_stop - t)
            # The first step after a current change must move every potential
            # from the old operating point to the new one, and the Newton
            # update is step-limited (GuWang2D._limit_step), so a large step
            # from rest (3C on the NMC cell) needs more than the default 25
            # iterations. Later steps keep the default: a converged solve is
            # the same solve whatever the iteration cap.
            first = abs(t - t_seg) < 1e-9
            un, ok, nit = mdl.step(u, dtl, maxit=200 if first else 25)
            V = mdl.cell_voltage(mdl.state(un)) if ok else None
            if ok and nstep > 0 and abs(V - hist["V"][-1]) > dV_max \
                    and dtl > dt_first * 1.01 and not (t == t_seg):
                ok = False                       # too large a voltage change
            if not ok:
                nfail += 1
                dt = 0.4 * dtl
                if dt < 1e-5:
                    raise RuntimeError(f"time step collapsed at t = {t:.2f} s")
                continue
            u = un
            if hasattr(mdl, "accept_step"):
                mdl.accept_step(u, dtl)
            t += dtl
            Q += p.I_cell * dtl
            nstep += 1
            record(mdl.state(u), V)
            if V < V_min or V > V_max:
                if verbose:
                    print(f"  voltage limit reached: V = {V:.4f} at t = {t:.1f} s")
                break
            g = 1.6 if nit <= 4 else (1.15 if nit <= 8 else 0.7)
            dt = min(dt * g, dt_max)
        else:
            if verbose:
                print(f"  segment {k}: {c:+.2f}C to t = {t:7.1f} s  "
                      f"V = {hist['V'][-1]:.4f}  Tavg = {hist['Tavg'][-1]-273.15:.3f} C")
            continue
        break
    out = {k: np.asarray(v) for k, v in hist.items()}
    out["wall_s"] = time.time() - wall
    out["n_steps"] = nstep
    out["n_rejected"] = nfail
    mdl.u_final = u
    return out

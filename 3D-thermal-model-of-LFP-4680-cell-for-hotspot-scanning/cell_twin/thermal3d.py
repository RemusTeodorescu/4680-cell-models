"""Paper 1.2 - step 2: thermal-only 3-D model of the BYD FC4680 on (r, theta, z).

    python thermal3d.py verify        # analytic + energy checks
    python thermal3d.py short         # micro-short resolution / detectability study
    python thermal3d.py uniform       # uniform heat, with and without the foam cap
    python thermal3d.py all

Finite volumes, cell-centred, periodic in theta, backward Euler in time.
The equations are linear (constant k, rho*cp, h), so a short is studied as
the temperature RISE it causes on its own (superposition): every result per
1 W scales to any power, and adds to the operating field.

Boundary: convection h to ambient on the side, the cap and the base. The foam
cap is a low h on one end. No contact resistance inside yet [E].
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geom3d import Cell, Grid, build_grid, MM, MAT_ID

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


# ============================================================ assembly
@dataclass
class BC:
    h_side: float = 20.0      # W/m^2/K, free convection + radiation [E]; hA_total ~0.30 W/K
    h_cap: float = 20.0       # positive end (z = H)
    h_base: float = 20.0      # negative end (z = 0)


def assemble(g: Grid, bc: BC):
    """Returns K (W/K, SPD, n x n), C (J/K, diag), Gb (W/K per cell to ambient)."""
    nr, nt, nz = g.shape
    idx = np.arange(g.n).reshape(g.shape)
    rf, tf, zf = g.r_f * MM, g.th_f, g.z_f * MM
    rc = 0.5 * (rf[1:] + rf[:-1])
    dr, dth, dz = np.diff(rf), np.diff(tf), np.diff(zf)
    kr, kt, kz = g.k
    rows, cols, vals = [], [], []

    def link(i1, i2, G):
        rows.extend([i1, i2]); cols.extend([i2, i1]); vals.extend([-G, -G])

    # radial faces (internal)
    R = rf[1:-1][:, None, None]
    A = R * dth[None, :, None] * dz[None, None, :]
    d1 = (rf[1:-1] - rc[:-1])[:, None, None]
    d2 = (rc[1:] - rf[1:-1])[:, None, None]
    G = A / (d1 / kr[:-1] + d2 / kr[1:])
    link(idx[:-1].ravel(), idx[1:].ravel(), G.ravel())
    # angular faces (periodic)
    A = (dr[:, None, None] * dz[None, None, :]) * np.ones((1, nt, 1))
    half = (rc[:, None, None] * dth[None, :, None] / 2)
    nxt = np.roll(np.arange(nt), -1)
    G = A / (half / kt + half[:, nxt] / kt[:, nxt])
    link(idx.ravel(), idx[:, nxt].ravel(), G.ravel())
    # axial faces
    Aax = ((rf[1:] ** 2 - rf[:-1] ** 2) / 2)[:, None, None] * dth[None, :, None]
    hz = dz[None, None, :] / 2
    G = Aax / (hz[..., :-1] / kz[..., :-1] + hz[..., 1:] / kz[..., 1:])
    link(idx[..., :-1].ravel(), idx[..., 1:].ravel(), G.ravel())

    # boundary conductances to ambient
    Gb = np.zeros(g.shape)
    A_side = rf[-1] * dth[:, None] * dz[None, :]
    inv = lambda h: np.inf if h <= 0 else 1.0 / h
    Gb[-1] += A_side / ((rf[-1] - rc[-1]) / kr[-1] + inv(bc.h_side))
    Aend = Aax[..., 0]
    Gb[..., 0] += Aend / (dz[0] / 2 / kz[..., 0] + inv(bc.h_base))
    Gb[..., -1] += Aend / (dz[-1] / 2 / kz[..., -1] + inv(bc.h_cap))

    K = sp.coo_matrix((np.concatenate([np.atleast_1d(v) for v in vals]),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(g.n, g.n)).tocsr()
    diag = -np.asarray(K.sum(axis=1)).ravel() + Gb.ravel()
    K = K + sp.diags(diag)
    C = (g.rho_cp * g.volumes()).ravel()
    return K.tocsc(), C, Gb.ravel()


class Solver:
    """Backward Euler. Each step is solved by CG with a smoothed-aggregation AMG
    preconditioner (pyamg), built once per time-step size, warm-started from the
    previous step. A direct LU was tried: 87 s and 115 M fill for 100 k cells."""

    def __init__(self, K, C, tol=1e-9):
        import pyamg
        self._pyamg = pyamg
        self.K, self.C, self.tol, self._ml = K.tocsr(), C, tol, {}
        self.iters = 0

    def _amg(self, dt):
        if dt not in self._ml:
            A = self.K if dt is None else (self.K + sp.diags(self.C / dt)).tocsr()
            self._ml[dt] = self._pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")
        return self._ml[dt]

    def _solve(self, dt, b, x0):
        res = []
        x = self._amg(dt).solve(b, x0=x0, tol=self.tol, accel="cg", residuals=res)
        self.iters += len(res)
        return x

    def steady(self, Q):
        return self._solve(None, Q, None)

    def run(self, Q, schedule, T0=None, callback=None):
        """schedule: list of (dt, n_steps). Q: source (W per cell) or callable(t)."""
        T = np.zeros(self.K.shape[0]) if T0 is None else T0.copy()
        t = 0.0
        for dt, n in schedule:
            for _ in range(n):
                q = Q(t + dt) if callable(Q) else Q
                T = self._solve(dt, self.C / dt * T + q, T)
                t += dt
                if callback:
                    callback(t, T)
        return T


def schedule_to(t_end, dts=((0.5, 2.0), (1.0, 10.0), (5.0, 60.0), (15.0, 300.0), (30.0, 600.0),
                            (60.0, 1800.0), (120.0, 7200.0))):
    s, t = [], 0.0
    for dt, tt in dts:
        if t >= t_end - 1e-9:
            break
        n = int(round((min(tt, t_end) - t) / dt))
        if n > 0:
            s.append((dt, n)); t += n * dt
    return s


# ============================================================ sources
def box_source(g: Grid, P, r0, th0, z0, d_r=0.6, w_arc=2.0, h_z=2.0):
    """Power P [W] spread uniformly over a box centred at (r0 mm, th0 rad, z0 mm),
    d_r x w_arc x h_z mm (radial x along-winding x axial). Exact separable overlap."""
    def overlap(f, a, b):
        lo, hi = np.maximum(f[:-1], a), np.minimum(f[1:], b)
        return np.clip(hi - lo, 0, None)
    # radial overlap weighted by r (volume), angular on the unrolled arc at r0
    fr = overlap(g.r_f, r0 - d_r / 2, r0 + d_r / 2) * 0.5 * (g.r_f[1:] + g.r_f[:-1])
    half = w_arc / 2 / r0
    th = g.th_f
    ft = sum(overlap(th, th0 - half + k * 2 * np.pi, th0 + half + k * 2 * np.pi) for k in (-1, 0, 1))
    fz = overlap(g.z_f, z0 - h_z / 2, z0 + h_z / 2)
    w = fr[:, None, None] * ft[None, :, None] * fz[None, None, :]
    if w.sum() == 0:
        raise ValueError("source box outside grid")
    return (P * w / w.sum()).ravel()


def uniform_roll_source(g: Grid, P):
    m = (g.mat == MAT_ID["roll"]).astype(float) * g.volumes()
    return (P * m / m.sum()).ravel()


# ============================================================ probes
def surface_masks(g: Grid):
    nr, nt, nz = g.shape
    side = np.zeros(g.shape, bool); side[-1] = True
    cap = np.zeros(g.shape, bool); cap[..., -1] = True
    base = np.zeros(g.shape, bool); base[..., 0] = True
    return dict(side=side.ravel(), cap=cap.ravel(), base=base.ravel())


def field_stats(g, T, masks):
    roll = (g.mat == MAT_ID["roll"]).ravel()
    return dict(max_internal=float(T[roll].max()),
                max_side=float(T[masks["side"]].max()),
                mean_side=float(T[masks["side"]].mean()),
                max_cap=float(T[masks["cap"]].max()),
                max_base=float(T[masks["base"]].max()),
                mean_cell=float(np.average(T, weights=(g.rho_cp * g.volumes()).ravel())))


# ============================================================ verification
def verify():
    """(1) Steady analytic solution of a uniform, isotropic cylinder with
    adiabatic ends and convection on the side; (2) energy conservation of the
    full model under uniform heat."""
    res = {}
    # (1) homogeneous cylinder
    for refine in (0.5, 1.0, 1.5):
        g = build_grid(refine=refine)
        k0, rc0, q = 5.0, 2.0e6, 1.0
        g.k[:] = k0; g.rho_cp[:] = rc0
        bc = BC(h_side=20.0, h_cap=0.0, h_base=0.0)
        K, C, Gb = assemble(g, bc)
        V = g.volumes().ravel()
        Q = q * V / V.sum()
        T = Solver(K, C).steady(Q)
        R = g.r_f[-1] * MM
        qv = q / V.sum()
        rc = 0.5 * (g.r_f[1:] + g.r_f[:-1]) * MM
        Ta = qv / (4 * k0) * (R ** 2 - rc ** 2) + qv * R / (2 * 20.0)
        Tn = T.reshape(g.shape).mean(axis=(1, 2))
        res[f"analytic_refine_{refine}"] = dict(
            nr=g.shape[0], max_abs_err_K=float(np.abs(Tn - Ta).max()), T_centre_K=float(Ta[0]))
    # (2) energy conservation, real materials
    g = build_grid()
    K, C, Gb = assemble(g, BC())
    Q = uniform_roll_source(g, 1.0)
    sol = Solver(K, C)
    lost, last = [0.0], [0.0]
    def cb(t, T):
        lost[0] += float(Gb @ T) * (t - last[0]); last[0] = t
    T = sol.run(Q, schedule_to(600), callback=cb)
    stored = float(C @ T)
    res["energy_600s"] = dict(input_J=600.0, stored_J=stored, lost_J=lost[0],
                              rel_err=(stored + lost[0] - 600.0) / 600.0)
    Ts = sol.steady(Q)
    res["steady_1W"] = dict(R_total_K_per_W=float(np.average(Ts, weights=C)),
                            hA_W_per_K=float(Gb.sum()),
                            C_total_J_per_K=float(C.sum()),
                            tau_s=float(C.sum() / Gb.sum()))
    return res


# ============================================================ short study
def short_locations(c: Cell):
    L = {n: (a, b) for n, a, b in c.z_layout()}
    zmid = 0.5 * sum(L["roll"])
    z_top = L["roll"][1] - 2.0
    tabs = c.tabs()
    tab = next(t for t in tabs if t["kind"] == "cathode" and abs(t["r"] - 16.0) < 0.1)
    return {
        "core":        (c.R_core + 0.8, 0.0, zmid),
        "mid_radius":  (13.0, 0.0, zmid),
        "near_side":   (c.R_roll - 0.8, 0.0, zmid),
        "near_cap":    (13.0, 0.0, z_top),
        "at_cap_tab":  (tab["r"], tab["theta"], z_top),
    }


T_REPORT = (2, 10, 60, 300, 600, 1800)


def short_response(g, sol, Q, masks, t_end=1800):
    times, hist = [], []
    def cb(t, T):
        times.append(t); hist.append(field_stats(g, T, masks))
    T = sol.run(Q, schedule_to(t_end), callback=cb)
    return np.array(times), hist, T


def short_study(refines=(0.75, 1.0, 1.5)):
    c = Cell()
    out = dict(per_W={}, convergence={}, snapshot=None)
    for refine in refines:
        g = build_grid(c, refine=refine)
        t0 = time.time()
        K, C, Gb = assemble(g, BC())
        sol = Solver(K, C)
        masks = surface_masks(g)
        locs = short_locations(c)
        conv = {}
        for name, (r0, th0, z0) in locs.items():
            Q = box_source(g, 1.0, r0, th0, z0)
            times, hist, T = short_response(g, sol, Q, masks)
            sel = {int(tt): hist[int(np.argmin(np.abs(times - tt)))] for tt in T_REPORT}
            Ts = sol.steady(Q)
            sel["steady"] = field_stats(g, Ts, masks)
            conv[name] = sel
            if refine == 1.0:
                out["per_W"][name] = dict(
                    t=times.tolist(),
                    max_internal=[h["max_internal"] for h in hist],
                    max_side=[h["max_side"] for h in hist],
                    max_cap=[h["max_cap"] for h in hist],
                    max_base=[h["max_base"] for h in hist],
                    steady=sel["steady"])
                if name == "mid_radius":
                    sol2 = sol
                    T60 = sol2.run(Q, schedule_to(60))
                    np.savez_compressed(OUT / "snapshot_mid_60s.npz", T=T60.reshape(g.shape),
                                        r_f=g.r_f, th_f=g.th_f, z_f=g.z_f, mat=g.mat)
                    np.savez_compressed(OUT / "snapshot_mid_600s.npz",
                                        T=sol2.run(Q, schedule_to(600)).reshape(g.shape),
                                        r_f=g.r_f, th_f=g.th_f, z_f=g.z_f, mat=g.mat)
        out["convergence"][f"{refine}"] = dict(shape=g.shape, n=g.n, seconds=time.time() - t0, res=conv)
        print(f"refine {refine}: {g.shape} {g.n} cells, {time.time()-t0:.1f} s", flush=True)
    (OUT / "short_study.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ============================================================ uniform / foam
def uniform_study(R_cell=11e-3, Q_Ah=15.41, h=22.4):
    """Uniform heat I^2 R at 1C and 2C with and without a foam cap on the
    positive end (h_cap 2.5 W/m^2K [E]). Charge-discharge heat as a constant
    average; entropic heat neglected here [E]. R_cell = 11 mOhm is the
    measured 60 s 1C pulse resistance at 51 % SOC (paper 1.1); h = 22.4 W/m^2K
    gives the bench hA = 0.337 W/K of the September test."""
    c = Cell()
    g = build_grid(c)
    masks = surface_masks(g)
    res = {}
    # probe positions for thermocouples on the can side: near cap, middle, near base
    rc, tc, zc = g.centres()
    zs = {"TC_cap": c.H_can - 5.0, "TC_mid": c.H_can / 2, "TC_base": 5.0}
    iz = {k: int(np.argmin(np.abs(zc - v))) for k, v in zs.items()}
    for foam in (False, True):
        bc = BC(h_side=h, h_base=h, h_cap=2.5 if foam else h)
        K, C, Gb = assemble(g, bc)
        sol = Solver(K, C)
        for crate in (1.0, 2.0):
            P = (crate * Q_Ah) ** 2 * R_cell
            Q = uniform_roll_source(g, P)
            rec = dict(t=[], **{k: [] for k in zs}, core_max=[], cap_centre=[], base_centre=[])
            def cb(t, T):
                T3 = T.reshape(g.shape)
                rec["t"].append(t)
                for k, i in iz.items():
                    rec[k].append(float(T3[-1, :, i].mean()))
                rec["core_max"].append(float(T3.max()))
                rec["cap_centre"].append(float(T3[0, :, -1].mean()))
                rec["base_centre"].append(float(T3[0, :, 0].mean()))
            sol.run(Q, schedule_to(3600 / crate), callback=cb)
            Ts = sol.steady(Q).reshape(g.shape)
            st = {k: float(Ts[-1, :, i].mean()) for k, i in iz.items()}
            st.update(core_max=float(Ts.max()), cap_centre=float(Ts[0, :, -1].mean()),
                      base_centre=float(Ts[0, :, 0].mean()))
            res[f"{'foam' if foam else 'bare'}_{crate:g}C"] = dict(P_W=P, transient=rec, steady=st)
            print(f"foam={foam} {crate}C P={P:.2f} W  end: " +
                  ", ".join(f"{k}={rec[k][-1]:.2f}" for k in zs) + f", core={rec['core_max'][-1]:.2f}",
                  flush=True)
    (OUT / "uniform_study.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("verify", "all"):
        r = verify(); print(json.dumps(r, indent=1))
        (OUT / "verify.json").write_text(json.dumps(r, indent=1))
    if what in ("short", "all"):
        short_study()
    if what in ("uniform", "all"):
        uniform_study()

"""Paper 1.2 - step 4: electrode sheet (s, z) coupled to the 3-D thermal grid.

    python sheet.py            # 2C discharge, with and without a soft short

The wound sheet is unrolled: s = 0..L along the cathode, z over the cathode
height. Two foil potentials, phi_Al and phi_Cu, carry the current along the
foils to the real tabs (cathode tabs on the cap edge, anode tabs on the base
edge, one anode tab disconnected). Between the foils, at every sheet node, a
LOCAL MODEL gives the current density from the local foil voltage and the
local temperature.

Local model interface
---------------------
    lm.linearize(T_node)      -> (U_eff [V], r_eff [ohm m^2])  per node
    lm.advance(j, T_node, dt)    update the internal state
    lm.heat(j, T_node)        -> W/m^2 of sheet, beyond the ohmic r_eff j^2 part

`TheveninLocal` below is a placeholder (OCV(SOC) + area-specific resistance with
Arrhenius T dependence). The MT-TDFN column of paper 1.1 plugs in through the
same three calls: linearize = its tangent (OCV-like source and the slope of
its current-voltage response over one step), advance = one step of the
column. This is the multi-scale multi-domain split of Kim et al. (2011).

Coupling to the thermal grid
----------------------------
Each sheet node is sampled at 1 mm along s, mapped by the spiral to (r, theta)
and to the 3-D cell it falls in. This gives a sparse matrix M (3-D cells x
sheet nodes): heat goes q3d = M q_sheet, temperature comes back as the
M-weighted average. Explicit (operator-split) coupling per thermal step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import byd_parts as B
from geom3d import Cell, build_grid, MM, MAT_ID
from thermal3d import BC, assemble, Solver, surface_masks, field_stats

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

SIG_AL = 3.50e7    # S/m
SIG_CU = 5.96e7
SIG_NI = 1.43e7


# ============================================================ local models
class TheveninLocal:
    """Placeholder local model: LFP-like OCV(SOC) and an area-specific
    resistance with Arrhenius temperature dependence. NOT MT-TDFN."""

    def __init__(self, n, q_areal, soc0=0.8, asr=1.3e-3, Ea=30e3, T_ref=298.15):
        self.soc = np.full(n, soc0)
        self.q_areal = q_areal            # C/m^2 of sheet (both sides together)
        self.asr, self.Ea, self.T_ref = asr, Ea, T_ref

    @staticmethod
    def ocv(soc):
        # flat LFP plateau with steep ends [E]
        return 3.30 + 0.06 * (soc - 0.5) + 0.25 * np.exp(-(1 - soc) / 0.03) - 0.4 * np.exp(-soc / 0.03)

    def r(self, T):
        return self.asr * np.exp(self.Ea / 8.314 * (1 / T - 1 / self.T_ref))

    def linearize(self, T):
        return self.ocv(self.soc), self.r(T)

    def advance(self, j, T, dt):
        # j > 0 charges (current from Al to Cu inside the cell)
        self.soc = np.clip(self.soc + j * dt / self.q_areal, 0.0, 1.0)

    def heat(self, j, T):
        return np.zeros_like(j)           # entropic heat neglected in the placeholder


# ============================================================ sheet grid
def _graded(L, focus, fine=1.0, coarse=8.0, band=12.0):
    """1-D faces on [0, L]: `fine` mm within `band` of each focus point."""
    pts = [0.0]
    while pts[-1] < L:
        x = pts[-1]
        d = min((abs(x - f) for f in focus), default=1e9)
        h = fine if d < band else min(coarse, max(fine, (d - band) * 0.5 + fine))
        pts.append(min(L, x + h))
    f = np.array(pts)
    if f[-1] - f[-2] < 0.3 * fine:
        f = np.delete(f, -2)
    return f


class Sheet:
    def __init__(self, cell: Cell, nz=12, short_s=None, tab_phi0=(np.pi / 2, np.pi / 2)):
        self.c = cell
        self.tabs = cell.tabs(*tab_phi0)
        focus = [t["s"] for t in self.tabs] + ([short_s] if short_s is not None else [])
        self.s_f = _graded(cell.L_cath, focus)
        self.z_f = np.linspace(0, cell.H_cath, nz + 1)
        self.ns, self.nz = len(self.s_f) - 1, nz
        self.n = self.ns * self.nz
        self.sc = 0.5 * (self.s_f[1:] + self.s_f[:-1])
        self.zc = 0.5 * (self.z_f[1:] + self.z_f[:-1])
        ds, dz = np.diff(self.s_f) * MM, np.diff(self.z_f) * MM
        self.A = (ds[:, None] * dz[None, :]).ravel()          # m^2 of sheet per node
        self.ds, self.dz = ds, dz
        self.idx = np.arange(self.n).reshape(self.ns, self.nz)

    def foil_laplacian(self, sig_t):
        """Conductance matrix (S) of one foil on the sheet, Neumann edges."""
        ds, dz, idx = self.ds, self.dz, self.idx
        rows, cols, vals = [], [], []
        # along s
        G = sig_t * dz[None, :] / (0.5 * (ds[:-1] + ds[1:]))[:, None]
        a, b = idx[:-1].ravel(), idx[1:].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-G.ravel(), -G.ravel()]
        # along z
        G = sig_t * ds[:, None] / (0.5 * (dz[:-1] + dz[1:]))[None, :]
        a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-G.ravel(), -G.ravel()]
        L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n, self.n)).tocsr()
        a = np.concatenate([idx[:-1].ravel(), idx[:, :-1].ravel()])   # all edges (s and z)
        b = np.concatenate([idx[1:].ravel(), idx[:, 1:].ravel()])
        # KCL form: (L phi)_i = sum_j G_ij (phi_j - phi_i)
        return sp.diags(np.asarray(L.sum(axis=1)).ravel()) - L, (a, b)

    def tab_nodes(self, tab, edge):
        """Edge nodes under a tab and their conductance share to the terminal."""
        jz = self.nz - 1 if edge == "top" else 0
        lo, hi = tab["s"] - self.c.tab_w / 2, tab["s"] + self.c.tab_w / 2
        ov = np.clip(np.minimum(self.s_f[1:], hi) - np.maximum(self.s_f[:-1], lo), 0, None)
        iss = np.nonzero(ov > 0)[0]
        return self.idx[iss, jz], ov[iss] / ov[iss].sum()


# ============================================================ electrical solve
# Internal-short types (Santhanagopalan et al. 2009 classification). The circuit
# is the same for all four - a conductance between the Al and Cu networks - but
# the type sets the plausible resistance and the layer that carries the heat.
# `offset_um`: radial position of the heat inside one winding period, measured
# outward from the inner face of the cathode sheet's Al foil [E]:
#   ... sep | anode coat | Cu | anode coat | sep | cath coat | Al | cath coat | sep ...
SHORT_TYPES = {
    "anode-cathode": dict(R_range=(1.0, 1e3),  offset_um=+70.0 + 6.9,   note="separator breach, coatings in series"),
    "Al-anode":      dict(R_range=(0.1, 10.0), offset_um=+70.0 + 13.7,  note="cathode coat penetrated; the dangerous one"),
    "Cu-cathode":    dict(R_range=(10., 1e4),  offset_um=+70.0 + 13.7 + 115.0, note="limited by the resistive LFP coat"),
    "Al-Cu":         dict(R_range=(0.01, 1.0), offset_um=+70.0 + 13.7 + 57.5, note="foil to foil, through both coats"),
}


class SheetCircuit:
    """phi_Al, phi_Cu on the sheet plus the positive terminal V (negative = 0)."""

    def __init__(self, sh: Sheet, R_tab_al=0.30e-3, R_tab_ni=0.65e-3, short=None, foil_gain=1.0):
        """foil_gain multiplies both foils' sheet conductance: 1 = 10 um foils of the
        teardown; > 1 = the effective value that makes the collection resistance
        agree with the bench (see calibrate_foil_gain)."""
        self.sh = sh
        c = sh.c
        self.foil_gain = foil_gain
        self.L_al, self.e_al = sh.foil_laplacian(foil_gain * SIG_AL * c.t_al * 1e-6)
        self.L_cu, self.e_cu = sh.foil_laplacian(foil_gain * SIG_CU * c.t_cu * 1e-6)
        self.G_al, self.G_cu = [], []          # (nodes, conductances) per tab
        for t in sh.tabs:
            if t["kind"] == "cathode":
                nodes, w = sh.tab_nodes(t, "top")
                # tab resistance plus a constriction term for the node edge half-cell
                self.G_al.append((nodes, w / R_tab_al, t))
            elif t["live"]:
                nodes, w = sh.tab_nodes(t, "bottom")
                self.G_cu.append((nodes, w / R_tab_ni, t))
        # short: dict(s, z, R, type="anode-cathode", side="outer") or None.
        # side="outer": the cathode faces the anode of the NEXT wrap, so the Cu
        # end of the short sits one circumference further along s.
        self.short = short
        if short is not None:
            short.setdefault("type", "anode-cathode"); short.setdefault("side", "outer")
            i_z = int(np.searchsorted(sh.z_f, short["z"]) - 1)
            i_s = int(np.searchsorted(sh.s_f, short["s"]) - 1)
            s_cu = short["s"] + (2 * np.pi * float(sh.c.r_of_s(short["s"])) if short["side"] == "outer" else 0.0)
            i_c = int(min(np.searchsorted(sh.s_f, s_cu) - 1, sh.ns - 1))
            self.short_node = int(sh.idx[i_s, i_z])
            self.short_node_cu = int(sh.idx[i_c, i_z])
            off = SHORT_TYPES[short["type"]]["offset_um"] * 1e-3
            self.short_r = float(sh.c.r_of_s(short["s"])) + (off if short["side"] == "outer" else -off)

    def solve(self, U, r, I):
        """I > 0: discharge current out of the positive terminal [A].
        Returns phi_al, phi_cu, V, j (A/m^2, + = charging direction)."""
        sh, n = self.sh, self.sh.n
        gA = sh.A / r                                  # S per node between the foils
        Dg = sp.diags(gA)
        Gt_al = np.zeros(n); Gt_cu = np.zeros(n)
        for nodes, g, _ in self.G_al:
            np.add.at(Gt_al, nodes, g)
        for nodes, g, _ in self.G_cu:
            np.add.at(Gt_cu, nodes, g)
        # unknowns x = [phi_al (n), phi_cu (n), V]
        # KCL Al:  L_al phi_al - gA (phi_al - phi_cu - U) - Gt_al (phi_al - V) = 0
        # KCL Cu:  L_cu phi_cu + gA (phi_al - phi_cu - U) - Gt_cu  phi_cu      = 0
        # terminal: sum Gt_al (phi_al - V) = I
        A11 = self.L_al - Dg - sp.diags(Gt_al)
        A12 = Dg
        A21 = Dg
        A22 = self.L_cu - Dg - sp.diags(Gt_cu)
        b1 = -gA * U
        b2 = gA * U
        if self.short is not None:
            ka, kc, gs = self.short_node, self.short_node_cu, 1.0 / self.short["R"]
            E = lambda i, j: sp.coo_matrix(([gs], ([i], [j])), shape=(n, n))
            A11 = A11 - E(ka, ka); A12 = A12 + E(ka, kc); A21 = A21 + E(kc, ka); A22 = A22 - E(kc, kc)
        col = sp.csr_matrix(Gt_al[:, None])
        M = sp.bmat([[A11, A12, col],
                     [A21, A22, None],
                     [sp.csr_matrix(Gt_al[None, :]), None, sp.csr_matrix([[-Gt_al.sum()]])]]).tocsc()
        rhs = np.concatenate([b1, b2, [I]])
        x = spla.spsolve(M, rhs)
        pa, pc, V = x[:n], x[n:2 * n], x[-1]
        j = (pa - pc - U) / r                       # A/m^2 of sheet; > 0 charging, < 0 discharging
        return pa, pc, V, j

    def heat(self, pa, pc, U, r, V):
        """Heat per sheet node [W]: foil ohmic, local r j^2, short. Tab heat [W] per tab."""
        sh = self.sh
        q = np.zeros(sh.n)
        for L, e, phi in ((self.L_al, self.e_al, pa), (self.L_cu, self.e_cu, pc)):
            a, b = e
            G = np.asarray(L[a, b]).ravel()
            p = G * (phi[a] - phi[b]) ** 2
            np.add.at(q, a, p / 2); np.add.at(q, b, p / 2)
        gA = sh.A / r
        q += gA * (pa - pc - U) ** 2
        q_short = 0.0
        if self.short is not None:
            q_short = (pa[self.short_node] - pc[self.short_node_cu]) ** 2 / self.short["R"]
            # deposited separately, at the layer the short type sets (see coupled_run)
        tab_q = []
        for nodes, g, t in self.G_al:
            tab_q.append((t, float((g * (pa[nodes] - V) ** 2).sum())))
        for nodes, g, t in self.G_cu:
            tab_q.append((t, float((g * pc[nodes] ** 2).sum())))
        return q, tab_q, q_short


# ============================================================ sheet <-> 3-D map
def sheet_to_grid(sh: Sheet, g, z0_cath, sub=1.0):
    """Sparse M (n3d x nsheet): fraction of each sheet node's area in each 3-D cell."""
    c = sh.c
    rows, cols, vals = [], [], []
    rmax = c.R_roll - 1e-6
    for i_s in range(sh.ns):
        s_pts = np.arange(sh.s_f[i_s] + sub / 2, sh.s_f[i_s + 1], sub)
        if len(s_pts) == 0:
            s_pts = np.array([sh.sc[i_s]])
        r = np.minimum(c.r_of_s(s_pts), rmax)
        th = c.theta_of_s(s_pts) % (2 * np.pi)
        ir = np.searchsorted(g.r_f, r) - 1
        it = np.minimum(np.searchsorted(g.th_f, th) - 1, len(g.th_f) - 2)
        for i_z in range(sh.nz):
            z = z0_cath + sh.zc[i_z]
            iz = np.searchsorted(g.z_f, z) - 1
            cells = (ir * g.shape[1] + it) * g.shape[2] + iz
            u, cnt = np.unique(cells, return_counts=True)
            rows.append(u); cols.append(np.full(len(u), sh.idx[i_s, i_z])); vals.append(cnt / cnt.sum())
    M = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(g.n, sh.n)).tocsr()
    return M


def tab_cells(g, c, tab, z_layout):
    """3-D cells of a tab strip (for its ohmic heat)."""
    L = {n: (a, b) for n, a, b in z_layout}
    gap = L["cap_gap"] if tab["kind"] == "cathode" else L["base_gap"]
    rc, tc, zc = g.centres()
    ir = int(np.argmin(np.abs(rc - tab["r"])))
    it = int(np.argmin(np.abs(np.angle(np.exp(1j * (tc - tab["theta"]))))))
    izs = np.nonzero((zc > gap[0]) & (zc < gap[1]))[0]
    return [(ir * g.shape[1] + it) * g.shape[2] + iz for iz in izs]


# ============================================================ coupled run
def coupled_run(I=2 * 15.41, t_end=600.0, dt=5.0, short=None, T_amb=298.15, refine=1.0, label="", asr=1.3e-3, h=22.4,
                local="thevenin", foil_gain=1.0, R_ext=0.0, n_s=16, n_z=2, soc0=0.8):
    c = Cell()
    g = build_grid(c, refine=refine)
    sh = Sheet(c, short_s=None if short is None else short["s"])
    circ = SheetCircuit(sh, short=short, foil_gain=foil_gain)
    Lz = {n: (a, b) for n, a, b in c.z_layout()}
    z0_cath = Lz["roll"][0] + (c.H_roll - c.H_cath) / 2
    M = sheet_to_grid(sh, g, z0_cath)
    w_back = sp.diags(1 / np.maximum(np.asarray(M.sum(axis=0)).ravel(), 1e-12)) @ M.T  # (n_sheet x n3d)
    tabc = {id(t): tab_cells(g, c, t, c.z_layout()) for t in sh.tabs}
    if short is not None:
        from thermal3d import box_source
        th_s = float(c.theta_of_s(short["s"]) % (2 * np.pi))
        w_short = box_source(g, 1.0, circ.short_r, th_s, z0_cath + short["z"],
                             d_r=short.get("d_r", 0.3), w_arc=short.get("w", 1.0), h_z=short.get("h", 1.0))
    q_areal = 15.41 * 3600 / sh.A.sum()
    if local == "mt":
        from mtlocal import MTColumnsLocal
        lm = MTColumnsLocal(sh, n_s=n_s, n_z=n_z, soc0=soc0, T0=T_amb)
        lm.dt = dt
        R_ext = lm.R_ext
    else:
        lm = TheveninLocal(sh.n, q_areal, asr=asr, soc0=soc0)
    side_wall = np.zeros(g.shape, bool); side_wall[-1] = True
    side_wall = side_wall.ravel(); side_wall_w = g.volumes().ravel() * side_wall; side_wall_w /= side_wall_w.sum()
    K, C, Gb = assemble(g, BC(h, h, h))
    sol = Solver(K, C)
    masks = surface_masks(g)
    T3 = np.zeros(g.n)                        # rise above ambient
    hist = []
    t = 0.0
    snap = None
    while t < t_end - 1e-9:
        Ts = T_amb + w_back @ T3
        U, r = lm.linearize(Ts)
        pa, pc, V, j = circ.solve(U, r, I)
        q_sheet, tab_q, q_short = circ.heat(pa, pc, U, r, V)
        q_sheet = q_sheet + lm.heat(j, Ts) * sh.A
        q3 = M @ q_sheet + I ** 2 * R_ext * side_wall_w
        if short is not None:
            q3 += q_short * w_short
        for tb, p in tab_q:
            cells = tabc[id(tb)]
            np.add.at(q3, cells, p / len(cells))
        T3 = sol._solve(dt, C / dt * T3 + q3, T3)
        lm.advance(j, Ts, dt)
        t += dt
        st = field_stats(g, T3, masks)
        st.update(t=t, V=float(V - I * R_ext), V_sheet=float(V), P_total=float(q3.sum()), P_short=float(q_short),
                  j_min=float(j.min()), j_max=float(j.max()), j_mean=float((j * sh.A).sum() / sh.A.sum()),
                  tab_currents=[float((g_ * (pa[n_] - V)).sum()) for n_, g_, _ in circ.G_al] +
                               [float((g_ * pc[n_]).sum()) for n_, g_, _ in circ.G_cu])
        hist.append(st)
    snap = dict(T3=T3.reshape(g.shape), j=j.reshape(sh.ns, sh.nz), q_sheet=(q_sheet / sh.A).reshape(sh.ns, sh.nz),
                pa=pa.reshape(sh.ns, sh.nz), pc=pc.reshape(sh.ns, sh.nz),
                s_f=sh.s_f, z_f=sh.z_f, r_f=g.r_f, th_f=g.th_f, zg_f=g.z_f, mat=g.mat)
    np.savez_compressed(OUT / f"coupled_{label}.npz", **snap)
    return dict(hist=hist, n_sheet=sh.n, n3d=g.n,
                tabs=[dict(kind=t["kind"], r=t["r"], live=t["live"]) for t in sh.tabs])


def collection_resistance(foil_gain=1.0, asr=0.1):
    """Collector resistance (foils + tabs) for near-uniform current, ohm."""
    c = Cell(); sh = Sheet(c); circ = SheetCircuit(sh, foil_gain=foil_gain)
    pa, pc, V, j = circ.solve(np.full(sh.n, 3.3), np.full(sh.n, asr), 1.0)
    return (3.3 - V) - asr / sh.A.sum()


def calibrate_foil_gain(R_target):
    """foil_gain such that the collection resistance equals R_target (secant)."""
    g0, g1 = 1.0, 10.0
    f0, f1 = collection_resistance(g0) - R_target, collection_resistance(g1) - R_target
    for _ in range(8):
        # R ~ a + b/g: secant in 1/g
        x0, x1 = 1 / g0, 1 / g1
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        g2 = 1 / max(x2, 1e-3)
        f2 = collection_resistance(g2) - R_target
        g0, f0, g1, f1 = g1, f1, g2, f2
        if abs(f2) < 0.01 * R_target:
            break
    return g1, f1 + R_target


def foil_resistance_check():
    """Uniform U, tiny r: the sheet resistance seen from the terminals with the
    local resistance removed, i.e. the collector (foil + tab) resistance."""
    c = Cell()
    sh = Sheet(c)
    circ = SheetCircuit(sh)
    out = {}
    for asr in (3e-3, 1e-1):
        U = np.full(sh.n, 3.3)
        r = np.full(sh.n, asr)
        pa, pc, V, j = circ.solve(U, r, 1.0)
        R_meas = (3.3 - V) / 1.0
        R_local = asr / sh.A.sum()
        out[f"asr_{asr}"] = dict(R_cell_mohm=R_meas * 1e3, R_local_mohm=R_local * 1e3,
                                 R_collect_mohm=(R_meas - R_local) * 1e3,
                                 j_max_over_mean=float(j.min() / ((j * sh.A).sum() / sh.A.sum())))
    return out


if __name__ == "__main__":
    t0 = time.time()
    chk = foil_resistance_check()
    print(json.dumps(chk, indent=1))
    c = Cell()
    tab = next(t for t in c.tabs() if t["kind"] == "cathode" and abs(t["r"] - 16.0) < 0.1)
    runs = {
        "2C_no_short": coupled_run(label="2C_no_short"),
        "2C_short_3ohm_mid": coupled_run(short=dict(s=c.s_of_r(13.0), z=c.H_cath / 2, R=3.0, type="Al-anode"),
                                         label="2C_short_3ohm_mid"),
        "rest_short_3ohm_mid": coupled_run(I=0.0, short=dict(s=c.s_of_r(13.0), z=c.H_cath / 2, R=3.0, type="Al-anode"),
                                           label="rest_short_3ohm_mid"),
    }
    res = dict(collector_check=chk, runs=runs, seconds=time.time() - t0)
    (OUT / "coupled.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in runs.items():
        h = v["hist"][-1]
        print(k, f"V={h['V']:.4f} P={h['P_total']:.3f} Psc={h['P_short']:.3f} "
                 f"Tin={h['max_internal']:.2f} Tside={h['max_side']:.2f}  tabs={np.round(h['tab_currents'],2)}")
    print(f"{time.time()-t0:.0f} s")

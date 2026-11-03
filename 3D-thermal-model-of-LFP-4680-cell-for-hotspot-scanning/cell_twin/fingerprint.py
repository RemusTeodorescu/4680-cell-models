"""Paper 1.2 - electrical fingerprint of an internal short, by position.

    python fingerprint.py

Seen from outside, a cell with a short inside is a two-port: port 1 is the
terminal, port 2 is the pair of foil points the short connects. Around an
operating point (small signal), the electrode sheet of sheet.py gives

    [V_term ]   [U]   [Z_cell  Z_tr] [I_term]
    [V_short] = [U] - [Z_tr    Z_th] [I_sc  ]       (reciprocal network)

and closing port 2 with the short, V_short = R_sc I_sc, gives

    I_sc    = (U - Z_tr I_term) / (R_sc + Z_th)
    V_term  = U - Z_cell I_term - Z_tr I_sc .

So the position of the short enters through two impedances, Z_th(position)
(the foil path and local electrode seen by the short) and Z_tr(position) (how
much of the short current shows up at the terminal). The response is
nonlinear in R_sc (hyperbolic), linear in the impedances. Each impedance is a
function of frequency: the local electrode has a series resistance and two
double-layer RC branches (cathode and anode), so the transient at onset also
depends on position.

The sheet is uniform along s (2 mm) for this study so that one factorisation
per frequency serves every position. Two foil cases, as in run_mt.py:
A = literal 10 um foils of the teardown, B = foils calibrated to the
collector resistance of paper 1.1 (foil_gain 27.5).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from geom3d import Cell, MM
from sheet import Sheet, SheetCircuit, SIG_AL, SIG_CU

OUT = Path(__file__).resolve().parent / "out"

# local electrode impedance per unit sheet area [ohm m^2], split of the 1.3e-3 ASR [E]:
# series (electrolyte, solid, contact) 30 %, cathode charge transfer 35 % with the
# cathode double layer, anode charge transfer 35 % with the anode double layer.
# Double-layer capacitances per sheet area from the paper 1.1 parameters
# (C_dl a_s L, both coated sides): cathode ~ 474 F/m^2, anode ~ 2.1 F/m^2.
ASR = 1.3e-3
LOCAL = dict(r_s=0.30 * ASR, r_c=0.35 * ASR, C_c=474.0, r_a=0.35 * ASR, C_a=2.1)


def z_local(f, p=LOCAL):
    w = 2 * np.pi * f
    return p["r_s"] + p["r_c"] / (1 + 1j * w * p["r_c"] * p["C_c"]) + p["r_a"] / (1 + 1j * w * p["r_a"] * p["C_a"])


class UniformSheet(Sheet):
    def __init__(self, cell, ds_mm=2.0, nz=12):
        super().__init__(cell, nz=nz)
        self.s_f = np.linspace(0, cell.L_cath, int(round(cell.L_cath / ds_mm)) + 1)
        self.ns = len(self.s_f) - 1
        self.n = self.ns * self.nz
        self.sc = 0.5 * (self.s_f[1:] + self.s_f[:-1])
        ds, dz = np.diff(self.s_f) * MM, np.diff(self.z_f) * MM
        self.A = (ds[:, None] * dz[None, :]).ravel()
        self.ds, self.dz = ds, dz
        self.idx = np.arange(self.n).reshape(self.ns, self.nz)


def port_matrix(circ: SheetCircuit, zloc):
    """System matrix of the sheet with local impedance zloc [ohm m^2] (complex ok),
    unknowns [phi_Al, phi_Cu, V]; same KCL as SheetCircuit.solve with U = 0."""
    sh, n = circ.sh, circ.sh.n
    gA = sh.A / zloc
    Dg = sp.diags(gA)
    Gt_al = np.zeros(n); Gt_cu = np.zeros(n)
    for nodes, g, _ in circ.G_al:
        np.add.at(Gt_al, nodes, g)
    for nodes, g, _ in circ.G_cu:
        np.add.at(Gt_cu, nodes, g)
    A11 = circ.L_al - Dg - sp.diags(Gt_al)
    A22 = circ.L_cu - Dg - sp.diags(Gt_cu)
    col = sp.csr_matrix(Gt_al[:, None])
    M = sp.bmat([[A11, Dg, col], [Dg, A22, None],
                 [sp.csr_matrix(Gt_al[None, :]), None, sp.csr_matrix([[-Gt_al.sum()]])]]).tocsc()
    return M.astype(complex)


def ports(circ, zloc, positions):
    """Z_cell and, for every (node_al, node_cu) pair, Z_th and Z_tr."""
    n = circ.sh.n
    lu = spla.splu(port_matrix(circ, zloc))
    # port 1: 1 A out of the terminal
    b = np.zeros(2 * n + 1, dtype=complex); b[-1] = 1.0
    x1 = lu.solve(b)
    Z_cell = -x1[-1]
    # port 2: 1 A from the Al node through the short into the Cu node.
    # KCL rows are 'sum of currents INTO the node = 0' with the injection on the rhs
    # entering with a minus sign (see the b1/b2 convention of SheetCircuit.solve).
    m = len(positions)
    Z_th = np.zeros(m, complex); Z_tr = np.zeros(m, complex)
    for c0 in range(0, m, 100):                  # in chunks: a dense right-hand side for all is GBs
        cols = range(c0, min(m, c0 + 100))
        B = np.zeros((2 * n + 1, len(cols)), dtype=complex)
        for i, p in enumerate(cols):
            ka, kc = positions[p]
            B[ka, i] = +1.0          # current leaving the Al network
            B[n + kc, i] = -1.0      # current entering the Cu network
        X = lu.solve(B)
        for i, p in enumerate(cols):
            ka, kc = positions[p]
            Z_th[p] = -(X[ka, i] - X[n + kc, i])
            Z_tr[p] = -X[-1, i]
    return Z_cell, Z_th, Z_tr


def check(circ, sh):
    """Compare the two-port prediction with a direct solve of SheetCircuit with a short."""
    zl = LOCAL["r_s"] + LOCAL["r_c"] + LOCAL["r_a"]
    s0, z0 = 2000.0, 35.0
    i_s = int(np.searchsorted(sh.s_f, s0) - 1); i_z = int(np.searchsorted(sh.z_f, z0) - 1)
    k = int(sh.idx[i_s, i_z])
    Zc, Zth, Ztr = ports(circ, zl, [(k, k)])
    U, Iterm, R = 3.30, 10.0, 0.1
    I_sc_pred = (U - Ztr[0].real * Iterm) / (R + Zth[0].real)
    V_pred = U - Zc.real * Iterm - Ztr[0].real * I_sc_pred
    c2 = SheetCircuit(sh, short=dict(s=s0, z=z0, R=R, type="Al-anode", side="inner"), foil_gain=circ.foil_gain)
    c2.short_node, c2.short_node_cu = k, k
    pa, pc, V, j = c2.solve(np.full(sh.n, U), np.full(sh.n, zl), Iterm)
    I_sc = (pa[k] - pc[k]) / R
    return dict(I_sc_pred=float(I_sc_pred), I_sc_direct=float(I_sc), V_pred=float(V_pred), V_direct=float(V))


def main():
    t0 = time.time()
    cell = Cell()
    sh = UniformSheet(cell, ds_mm=2.0, nz=12)
    freqs = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # positions: every 5th s node (10 mm), z rows 1, 6, 10 (near the base, mid, near the cap),
    # inner side (Al and Cu at the same s) and outer side (Cu one turn further)
    iz_list = {"base": 1, "mid": 6, "cap": 10}
    pos, meta = [], []
    for i_s in range(0, sh.ns, 5):
        s = sh.sc[i_s]
        for zname, i_z in iz_list.items():
            for side in ("inner", "outer"):
                s_cu = s + (2 * np.pi * float(cell.r_of_s(s)) if side == "outer" else 0.0)
                if s_cu >= cell.L_cath:
                    continue
                i_c = int(min(np.searchsorted(sh.s_f, s_cu) - 1, sh.ns - 1))
                pos.append((int(sh.idx[i_s, i_z]), int(sh.idx[i_c, i_z])))
                meta.append(dict(s=float(s), r=float(cell.r_of_s(s)), z=zname, side=side))
    res = dict(freqs=freqs, local=LOCAL, n_sheet=sh.n, positions=meta, cases={})
    for case, gain in (("A_literal", 1.0), ("B_calibrated", 27.51169200711839)):
        circ = SheetCircuit(sh, foil_gain=gain)
        out = dict(Z_cell=[], Z_th=[], Z_tr=[])
        for f in freqs:
            zl = z_local(f) if f > 0 else z_local(0.0).real
            Zc, Zth, Ztr = ports(circ, zl, pos)
            out["Z_cell"].append([float(np.real(Zc)), float(np.imag(Zc))])
            out["Z_th"].append([np.real(Zth).tolist(), np.imag(Zth).tolist()])
            out["Z_tr"].append([np.real(Ztr).tolist(), np.imag(Ztr).tolist()])
            print(case, f"f={f:g} Hz Z_cell={abs(Zc)*1e3:.3f} mOhm  Z_th {np.abs(Zth).min()*1e3:.2f}-{np.abs(Zth).max()*1e3:.2f} mOhm"
                  f"  Z_tr/Z_cell {np.abs(Ztr).min()/abs(Zc):.3f}-{np.abs(Ztr).max()/abs(Zc):.3f}  {time.time()-t0:.0f}s", flush=True)
        out["check"] = check(circ, sh)
        print(case, "check", out["check"], flush=True)
        res["cases"][case] = out
    (OUT / "fingerprint.json").write_text(json.dumps(res))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

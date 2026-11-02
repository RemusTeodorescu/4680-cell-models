"""Paper 1.2 - MT-TDFN columns as the local model of the electrode sheet.

Each column is the paper 1.1 model (gw_mt.MTTDFN, isothermal at the local
temperature of the 3-D field), standing for one patch of the unrolled sheet
(Kim et al. 2011, multi-scale multi-domain). The sheet nodes of a patch share
the column's linearisation:

    V_loc(j) = U_eff + r_eff * j        (V_loc = phi_Al - phi_Cu, j in A/m^2 of sheet, + = charge)

obtained each time step from a trial step of the column at the patch's
previous mean current (U_eff) and, every few steps, a second trial step at a
perturbed current (r_eff). The sheet is then solved, and every column is
advanced at its new patch-mean current. The column's own heat (reaction,
reversible, ohmic in the electrolyte and solid) is spread over the patch.

The column's external series resistance R_series is split: the collector part
(R_collect_mean of byd_parts) is replaced by the resolved foil network of the
sheet; the rest (can wall, contacts and the dRs identified in paper 1.1) stays
as a lumped resistance at the terminal, R_ext.

Parameters: the paper 1 calibration (mt.CAL) with the paper 1.1 interface
(k, C_dl), the shell and the resistive reactant of tw10_kw.json. The
particle populations and hysteresis of paper 1.1 need ocp_gitt.json, which
was not available here; they are switched off [to add].
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import os

import numpy as np

# The MT-TDFN code of paper 1.1 (gw_mt.py, mt.py, byd4680_params.py, tw10_kw.json, ...)
# is released with that paper. Point MTTDFN_DIR at it.
MTDIR = Path(os.environ.get("MTTDFN_DIR", Path(__file__).resolve().parents[1] / "mttdfn"))
sys.path.insert(0, str(MTDIR))

KW_KEYS = ('k_pos', 'k_neg', 'Cdl_pos', 'Cdl_neg', 'dRs', 'i0_theta_pos', 'i0_theta_neg',
           'sh_Rinf', 'sh_Zc', 'sh_tau', 'sh_tauZ', 'sh_thn', 'sh_m', 'rr_tau', 'sh_p', 'sh_Rrr',
           'rr_Zc', 'rr_spread')


def _build_column(T_amb_C, soc0, kwfile="tw10_kw.json", NX=(1, 6, 3, 6, 1)):
    import byd4680_params as B
    if not getattr(B, "_iso_patched", False):
        _o = B.params_byd4680

        def _p(*a, **k):
            p = _o(*a, **k); p.isothermal = True; return p
        B.params_byd4680 = _p
        B._iso_patched = True
    import mt
    K = json.load(open(MTDIR / kwfile))
    kw = {k: K[k] for k in KW_KEYS if k in K}
    Qnom = 15.41
    m, _ = mt.build(T_amb=T_amb_C, NX=NX, Qrem=(1.0 - soc0) * Qnom, **kw)
    return m


class MTColumnsLocal:
    """Local-model interface (linearize / advance / heat) backed by MT-TDFN columns."""

    def __init__(self, sheet, n_s=16, n_z=2, soc0=0.8, T0=298.15, refresh=5, dI_rel=0.05):
        self.sh = sheet
        self.A_tot = float(sheet.A.sum())
        # patches: equal-length bins along s, n_z bins in z
        s_edges = np.linspace(0, sheet.s_f[-1], n_s + 1)
        z_edges = np.linspace(0, sheet.z_f[-1], n_z + 1)
        i_s = np.clip(np.searchsorted(s_edges, sheet.sc) - 1, 0, n_s - 1)
        i_z = np.clip(np.searchsorted(z_edges, sheet.zc) - 1, 0, n_z - 1)
        self.patch = (i_s[:, None] * n_z + i_z[None, :]).ravel()
        self.K = n_s * n_z
        self.A_patch = np.bincount(self.patch, weights=sheet.A, minlength=self.K)
        self.cols = [_build_column(T0 - 273.15, soc0) for _ in range(self.K)]
        self.u = []
        for m in self.cols:
            m.p.C_rate = 1e-9
            u = m.initial_state()
            for dt in (1.0, 10.0):
                u, ok, _ = m.step(u, dt)
            self.u.append(u)
        p0 = self.cols[0].p
        import byd_parts as BP
        self.R_collect_col = float(BP.build().R_collect_mean)
        self.R_ext = float(p0.R_series - self.R_collect_col)     # stays at the terminal
        self.jbar = np.zeros(self.K)                              # A/m^2, + charge
        self.r_eff = np.full(self.K, np.nan)
        self.U_eff = np.zeros(self.K)
        self.Q_col = np.zeros(self.K)
        self.refresh, self.dI_rel, self.k_step = refresh, dI_rel, 0
        self._committed = True
        self.n_steps = 0

    # ---------------------------------------------------------- helpers
    def _Vloc(self, m, u):
        st = m.state(u)
        return m.cell_voltage(st) + m.p.I_cell * m.p.R_series, st

    def _set_current(self, m, j):
        """j (A/m^2 of sheet, + charge) -> column C_rate (+ discharge), whole-cell scale."""
        I = -j * self.A_tot                                  # A, + discharge
        cap = m.p.capacity_Ah
        m.p.C_rate = I / cap if abs(I) > 1e-9 else 1e-9

    def _trial(self, k, j, dt, T):
        m = self.cols[k]
        m.p.T0 = float(T)
        self._set_current(m, j)
        u, ok, _ = m.step(self.u[k], dt)
        if not ok:
            u, ok = self._substep(m, self.u[k], dt)
        return self._Vloc(m, u)[0]

    def _substep(self, m, u, dt, n=8):
        h = dt / n
        for _ in range(n):
            u, ok, _ = m.step(u, h)
            if not ok:
                return u, False
        return u, True

    def patch_T(self, T_node):
        return np.bincount(self.patch, weights=T_node * self.sh.A, minlength=self.K) / self.A_patch

    # ---------------------------------------------------------- interface
    def linearize(self, T_node, dt=None):
        dt = dt or self.dt
        Tp = self.patch_T(T_node)
        need_r = np.isnan(self.r_eff).any() or (self.k_step % self.refresh == 0)
        for k in range(self.K):
            j0 = self.jbar[k]
            V0 = self._trial(k, j0, dt, Tp[k])
            if need_r:
                dj = max(abs(j0) * self.dI_rel, 2.0)              # A/m^2
                V1 = self._trial(k, j0 + dj, dt, Tp[k])
                self.r_eff[k] = max((V1 - V0) / dj, 1e-6)
            self.U_eff[k] = V0 - self.r_eff[k] * j0
        self._Tp = Tp
        self._committed = False
        return self.U_eff[self.patch], self.r_eff[self.patch]

    def commit(self, j, dt=None):
        if self._committed:
            return
        dt = dt or self.dt
        jb = np.bincount(self.patch, weights=j * self.sh.A, minlength=self.K) / self.A_patch
        for k in range(self.K):
            m = self.cols[k]
            m.p.T0 = float(self._Tp[k])
            self._set_current(m, jb[k])
            u, ok, _ = m.step(self.u[k], dt)
            if not ok:
                u, ok = self._substep(m, self.u[k], dt)
            self.u[k] = u
            st = m.state(u)
            # column heat is for the whole cell at the patch's current density:
            # per unit sheet area it is Q / A_tot
            self.Q_col[k] = float((st["q"] * m.m.V).sum())
        self.jbar = jb
        self.k_step += 1
        self._committed = True
        self.n_steps += 1

    def heat(self, j, T_node):
        """Extra heat per sheet area beyond r_eff j^2 (which the circuit already counts):
        the column heat minus its r_eff jbar^2 part, spread over the patch in
        proportion to the local current."""
        self.commit(j)
        q_patch = self.Q_col / self.A_tot                           # W/m^2 at the patch mean
        extra = q_patch - self.r_eff * self.jbar ** 2
        jb = self.jbar[self.patch]
        w = np.where(np.abs(jb) > 1e-6, j / np.where(np.abs(jb) > 1e-6, jb, 1.0), 1.0)
        return extra[self.patch] * w

    def advance(self, j, T_node, dt):
        self.commit(j, dt)

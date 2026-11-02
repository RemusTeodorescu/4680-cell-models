"""Paper 1.2 - geometry and mesh of the BYD FC4680 for the 3-D thermal model.

    python geom3d.py            # prints the geometry audit

Coordinates: cylindrical (r, theta, z); z = 0 at the outside of the base
(negative end), z = H_can at the outside of the cap (positive end).

What this module provides
-------------------------
* `Cell`         every dimension, from byd_parts.py (Liu et al. 2025 teardown),
                 with the same [M]/[D]/[E] tags;
* spiral map     s (position along the cathode, 0..5.37 m) <-> (r, theta) of
                 the Archimedean winding, used to place tabs and, in step 4,
                 the electrode-sheet heat;
* tabs           radius, winding position s and angle of each tab;
* stack          through-plane and in-plane conductivity of the wound stack,
                 from the layer thicknesses and the paper 1 conductivities;
* `Grid`         a structured finite-volume grid on (r, theta, z) with a
                 material id, anisotropic conductivity and rho*cp per cell,
                 and the tab strips resolved in the two end gaps.

Assumptions not given by the teardown are tagged [E] where they are set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import byd_parts as B                     # noqa: E402

MM = 1e-3      # m per mm

# ----------------------------------------------------------------- materials
# SI units: k [W/m/K], rho*cp [J/m^3/K]. Electrode and separator k as the
# paper 1 parameter set (byd4680_params.py), metals and plastic handbook [L].
MATERIALS = {
    #            k_r     k_t     k_z     rho_cp
    "void":     (0.10,   0.10,   0.10,   1.20e6),   # electrolyte-wetted void / gas [E]
    "plastic":  (0.22,   0.22,   0.22,   2.04e6),   # PP/PET insulator, mandrel [L]
    "steel":    (15.0,   15.0,   15.0,   3.71e6),   # Ni-plated steel can [L]; byd_parts uses 15 W/m/K
    "Al":       (237.,   237.,   237.,   2.43e6),
    "Cu":       (401.,   401.,   401.,   3.45e6),
    "Ni":       (90.7,   90.7,   90.7,   3.95e6),
    "roll":     None,                              # anisotropic, from the stack below
    "tape":     (0.20,   0.20,   0.20,   1.9e6),    # outer wrap, PET [L]
}
MAT_ID = {name: i for i, name in enumerate(MATERIALS)}


@dataclass
class Cell:
    # envelope [mm]
    R_can: float = B.CAN_OD / 2 * 10          # 23.15 [M]
    wall: float = B.CAN_WALL * 10             # 0.418 [D]
    H_can: float = B.CAN_H * 10               # 80.2 [M]
    # roll [mm]
    R_core: float = B.core_radius() * 10      # 3.16 mandrel outer radius [D]
    R_roll: float = B.roll_radius_max() * 10  # 22.73 bore-filling [D]
    H_roll: float = B.AN_H * 10               # 71.0 anode height [M]
    H_cath: float = B.CATH_H * 10             # 69.0 [M]
    L_cath: float = B.CATH_L * 10             # 5370 mm [M]
    mandrel_h: float = B.MANDREL_H * 10       # 7.1 [M], a stub at mid height [E position]
    # ends [mm]
    t_disk: float = B.DISK_T * 10             # 0.31 [M]
    t_ins: float = B.PLASTIC_T * 10           # 0.29 [M]
    R_al_disk: float = B.CATHODE_DISK_OD / 2 * 10   # 20 [E]
    R_cu_disk: float = B.ANODE_DISK_OD / 2 * 10     # 22 [E]
    # tabs
    cath_tab_r: tuple = tuple(r * 10 for r in B.CATHODE_TAB_R)   # [M] (0.4 mm one: see note)
    an_tab_r: tuple = tuple(r * 10 for r in B.ANODE_TAB_R)       # [M]
    an_tab_dead: int = 0                      # [E] which anode tab is the PET-covered one
    tab_w: float = B.TAB_W * 10               # 4.0 [E]
    tab_t: float = 0.10                       # mm, tab foil thickness [E]
    # layer stack [um], per winding period
    t_cath_coat: float = (B.T_CATH_SHEET - B.T_AL) * 1e4     # both sides, through-sheet reading [M/D]
    t_al: float = B.T_AL * 1e4
    t_an_coat: float = (B.T_AN_SHEET - B.T_CU) * 1e4
    t_cu: float = B.T_CU * 1e4
    t_sep2: float = 2 * B.T_SEP * 1e4
    k_cath: float = 1.5                       # W/m/K [E] paper 1
    k_an: float = 1.04                        # [E] paper 1
    k_sep: float = 0.33                       # [E] paper 1
    rho_cp_roll: float = 0.0                  # set in __post_init__ [D] from byd_stack

    def __post_init__(self):
        # roll heat capacity from the phase inventory of byd_stack.py: 248.9 J/K
        V_roll = np.pi * ((self.R_roll * MM) ** 2 - (self.R_core * MM) ** 2) * self.H_roll * MM
        self.rho_cp_roll = 248.9 / V_roll

    # ---------------------------------------------------- stack conductivity
    def stack(self):
        t = np.array([self.t_cath_coat, self.t_al, self.t_an_coat, self.t_cu, self.t_sep2])
        k = np.array([self.k_cath, 237.0, self.k_an, 401.0, self.k_sep])
        k_through = t.sum() / (t / k).sum()
        k_inplane = (t * k).sum() / t.sum()
        return dict(period_um=float(t.sum()), k_through=float(k_through), k_inplane=float(k_inplane))

    @property
    def pitch(self):                                  # mm per turn
        return self.stack()["period_um"] * 1e-3

    # ------------------------------------------------------------ the spiral
    def r_of_s(self, s):
        """Radius [mm] at position s [mm] along the cathode (Archimedean)."""
        return np.sqrt(self.R_core ** 2 + np.asarray(s) * self.pitch / np.pi)

    def s_of_r(self, r):
        return np.pi * (np.asarray(r) ** 2 - self.R_core ** 2) / self.pitch

    def theta_of_s(self, s, phi0=0.0):
        """Angle [rad] of the winding at s; phi0 = angle of the winding start."""
        return phi0 + 2 * np.pi * (self.r_of_s(s) - self.R_core) / self.pitch

    # Liu et al. (2025): "The four aluminum cathode tabs are welded to the metallic
    # cathode disk at roughly 90 deg intervals." Which tab sits at which angle is
    # not given; radius order is assumed [E]. Anode tab angles are not reported.
    cath_tab_angles: tuple = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)

    def _snap(self, r_rep, alpha, phi0):
        """Radius nearest r_rep at which the winding passes angle alpha."""
        n = np.round((r_rep - self.R_core) / self.pitch - (alpha - phi0) / (2 * np.pi))
        r = self.R_core + self.pitch * ((alpha - phi0) / (2 * np.pi) % 1.0 + np.floor(n))
        cands = r + self.pitch * np.array([-1, 0, 1])
        cands = cands[cands > self.R_core]
        return float(cands[np.argmin(np.abs(cands - r_rep))])

    def tabs(self, phi0_cath=np.pi / 2, phi0_an=np.pi / 2, snap_angles=True):
        """Tab list with (kind, radius, s, theta, live). Radii are the teardown's.
        With snap_angles, each cathode tab is moved by less than half a wrap
        (< 0.16 mm in radius) so that it sits at its 90-degree position. The
        0.4 mm cathode tab lies inside the mandrel and is moved to the first wrap."""
        out = []
        for i, r in enumerate(self.cath_tab_r):
            rr = r if r > self.R_core else self.R_core + self.pitch
            if snap_angles:
                rr = self._snap(rr, self.cath_tab_angles[i], phi0_cath)
            s = float(self.s_of_r(rr))
            out.append(dict(kind="cathode", r=rr, r_reported=r, s=s,
                            theta=float(self.theta_of_s(s, phi0_cath) % (2 * np.pi)), live=True))
        for i, r in enumerate(self.an_tab_r):
            s = float(self.s_of_r(r))
            out.append(dict(kind="anode", r=r, r_reported=r, s=s,
                            theta=float(self.theta_of_s(s, phi0_an) % (2 * np.pi)),
                            live=(i != self.an_tab_dead)))
        return out

    # ------------------------------------------------------------ axial plan
    def z_layout(self):
        """Axial stack from the base, mm: list of (name, z0, z1)."""
        inner = self.H_can - 2 * self.wall - 2 * self.t_ins - 2 * self.t_disk
        gap = (inner - self.H_roll) / 2.0           # tab gap at each end [D]
        parts, z = [], 0.0
        for name, t in (("base_wall", self.wall), ("base_ins", self.t_ins), ("cu_disk", self.t_disk),
                        ("base_gap", gap), ("roll", self.H_roll), ("cap_gap", gap),
                        ("al_disk", self.t_disk), ("cap_ins", self.t_ins), ("cap_wall", self.wall)):
            parts.append((name, z, z + t))
            z += t
        return parts


# =================================================================== the grid
@dataclass
class Grid:
    cell: Cell
    r_f: np.ndarray            # radial faces, mm
    th_f: np.ndarray           # angular faces, rad
    z_f: np.ndarray            # axial faces, mm
    mat: np.ndarray = field(repr=False, default=None)        # (nr, nth, nz) int
    k: np.ndarray = field(repr=False, default=None)          # (3, nr, nth, nz) W/m/K
    rho_cp: np.ndarray = field(repr=False, default=None)     # (nr, nth, nz) J/m^3/K
    tab_mask: np.ndarray = field(repr=False, default=None)   # tab strips (bool)

    @property
    def shape(self):
        return (len(self.r_f) - 1, len(self.th_f) - 1, len(self.z_f) - 1)

    @property
    def n(self):
        a, b, c = self.shape
        return a * b * c

    def centres(self):
        rc = 0.5 * (self.r_f[1:] + self.r_f[:-1])
        tc = 0.5 * (self.th_f[1:] + self.th_f[:-1])
        zc = 0.5 * (self.z_f[1:] + self.z_f[:-1])
        return rc, tc, zc

    def volumes(self):
        """Cell volumes, m^3."""
        r2 = (self.r_f[1:] ** 2 - self.r_f[:-1] ** 2) / 2 * MM ** 2
        dth = np.diff(self.th_f)
        dz = np.diff(self.z_f) * MM
        return r2[:, None, None] * dth[None, :, None] * dz[None, None, :]


def _seg(z0, z1, n):
    return np.linspace(z0, z1, n + 1)


def refine_faces(f, center, h, half, grade=1.3):
    """Insert a block of spacing h over [center-half, center+half] into the face
    array f, graded geometrically (ratio `grade`) back to the local spacing."""
    f = np.asarray(f, float)
    base = float(np.median(np.diff(f[(f > center - 4 * half) & (f < center + 4 * half)]))) \
        if ((f > center - 4 * half) & (f < center + 4 * half)).sum() > 2 else float(np.median(np.diff(f)))
    n = int(np.ceil(half / h))
    core = center + h * np.arange(-n, n + 1)
    right, x, st = [], core[-1], h
    while st < base:
        st *= grade; x += st; right.append(x)
    left, x, st = [], core[0], h
    while st < base:
        st *= grade; x -= st; left.append(x)
    lo, hi = (left[-1] if left else core[0]), (right[-1] if right else core[-1])
    keep = f[(f < lo - 0.5 * base) | (f > hi + 0.5 * base)]
    return np.unique(np.concatenate([keep, left[::-1], core, right]))


def build_grid(cell: Cell | None = None, n_core=3, n_roll_r=40, n_th=48, n_roll_z=36,
               n_gap_z=3, refine=1.0, tab_phi0=(np.pi / 2, np.pi / 2), patch=None) -> Grid:
    """patch: optional local refinement dict(r0, th0, z0 [mm, rad, mm], h_inplane [mm],
    half [mm], h_r [mm] or None, half_r [mm]) around a short."""
    """Structured grid. `refine` scales the roll's radial, angular and axial counts."""
    cell = cell or Cell()
    nr = max(4, int(round(n_roll_r * refine)))
    nt = max(8, int(round(n_th * refine)))
    nz = max(6, int(round(n_roll_z * refine)))
    r_f = np.concatenate([_seg(0, cell.R_core, n_core)[:-1],
                          _seg(cell.R_core, cell.R_roll, nr)[:-1],
                          [cell.R_roll, cell.R_can - cell.wall, cell.R_can]])
    # R_roll equals the bore in this cell (bore-filling roll); avoid a zero cell
    r_f = np.unique(np.round(r_f, 9))
    th_f = np.linspace(0, 2 * np.pi, nt + 1)
    zs = []
    for name, z0, z1 in cell.z_layout():
        n = nz if name == "roll" else (n_gap_z if "gap" in name else 1)
        zs.append(_seg(z0, z1, n)[:-1])
    z_f = np.concatenate(zs + [[cell.H_can]])
    if patch is not None:
        P = patch
        z_f = refine_faces(z_f, P["z0"], P["h_inplane"], P["half"])
        th_f = refine_faces(th_f, P["th0"], P["h_inplane"] / P["r0"], P["half"] / P["r0"])
        th_f = th_f[(th_f >= 0) & (th_f <= 2 * np.pi)]
        th_f[0], th_f[-1] = 0.0, 2 * np.pi
        if P.get("h_r"):
            r_f = refine_faces(r_f, P["r0"], P["h_r"], P.get("half_r", 2.0))
    g = Grid(cell, r_f, th_f, z_f)
    _fill(g, tab_phi0)
    return g


def _fill(g: Grid, tab_phi0):
    c = g.cell
    rc, tc, zc = g.centres()
    R, T, Z = np.meshgrid(rc, tc, zc, indexing="ij")
    mat = np.full(R.shape, MAT_ID["void"], dtype=np.int16)
    layout = {name: (z0, z1) for name, z0, z1 in c.z_layout()}

    def zin(name):
        z0, z1 = layout[name]
        return (Z > z0) & (Z < z1)
    inside = R < c.R_can - c.wall
    # can
    mat[R > c.R_can - c.wall] = MAT_ID["steel"]
    mat[inside & (zin("base_wall") | zin("cap_wall"))] = MAT_ID["steel"]
    mat[inside & (zin("base_ins") | zin("cap_ins"))] = MAT_ID["plastic"]
    mat[(R < c.R_cu_disk) & zin("cu_disk")] = MAT_ID["Cu"]
    mat[(R < c.R_al_disk) & zin("al_disk")] = MAT_ID["Al"]
    # roll and mandrel
    roll = (R > c.R_core) & (R < c.R_roll) & zin("roll")
    mat[roll] = MAT_ID["roll"]
    zm = 0.5 * sum(layout["roll"])
    mandrel = (R < c.R_core) & (np.abs(Z - zm) < c.mandrel_h / 2)
    mat[mandrel] = MAT_ID["plastic"]
    # tabs: strips through the end gaps at the tab radius and angle
    tab_mask = np.zeros(R.shape, bool)
    tabs = c.tabs(*tab_phi0)
    dth = np.diff(g.th_f)[0]
    for tb in tabs:
        if not tb["live"]:
            continue
        gap = "cap_gap" if tb["kind"] == "cathode" else "base_gap"
        half = (c.tab_w / 2) / tb["r"]                         # angular half width, rad
        dang = np.angle(np.exp(1j * (T - tb["theta"])))
        ir = np.argmin(np.abs(rc - tb["r"]))
        sel = (np.abs(dang) <= max(half, dth / 2)) & zin(gap) & (np.abs(R - rc[ir]) < 1e-9)
        mat[sel] = MAT_ID["Al" if tb["kind"] == "cathode" else "Ni"]
        tab_mask |= sel
    # properties
    st = c.stack()
    k = np.empty((3,) + R.shape)
    rho_cp = np.empty(R.shape)
    for name, i in MAT_ID.items():
        m = mat == i
        if name == "roll":
            k[0][m], k[1][m], k[2][m] = st["k_through"], st["k_inplane"], st["k_inplane"]
            rho_cp[m] = c.rho_cp_roll
        else:
            kr, kt, kz, rc_ = MATERIALS[name]
            k[0][m], k[1][m], k[2][m], rho_cp[m] = kr, kt, kz, rc_
    # a tab strip is a thin foil inside a wider grid cell: scale its axial k by the
    # area fraction it really occupies, so the cell carries the strip's conductance
    vol_frac = np.zeros(R.shape)
    if tab_mask.any():
        dr = np.diff(g.r_f)[:, None, None] * np.ones(R.shape)
        arc = R * np.diff(g.th_f)[None, :, None]
        frac = (c.tab_t / dr) * np.minimum(1.0, c.tab_w / arc)
        vol_frac[tab_mask] = np.clip(frac[tab_mask], 0, 1)
        kv = MATERIALS["void"][2]
        for ax in range(3):
            kt_ = k[ax][tab_mask]
            k[ax][tab_mask] = vol_frac[tab_mask] * kt_ + (1 - vol_frac[tab_mask]) * kv
        rho_cp[tab_mask] = vol_frac[tab_mask] * rho_cp[tab_mask] + (1 - vol_frac[tab_mask]) * MATERIALS["void"][3]
    g.mat, g.k, g.rho_cp, g.tab_mask = mat, k, rho_cp, tab_mask


def audit(g: Grid | None = None):
    c = Cell()
    st = c.stack()
    g = g or build_grid(c)
    V = g.volumes()
    C_total = float((g.rho_cp * V).sum())
    out = dict(
        stack=st, pitch_mm=c.pitch, turns=(c.R_roll - c.R_core) / c.pitch,
        spiral_length_check_mm=float(c.s_of_r(c.R_roll)), L_cath_mm=c.L_cath,
        z_layout_mm=[(n, round(a, 3), round(b, 3)) for n, a, b in c.z_layout()],
        tabs=c.tabs(), grid_shape=g.shape, n_cells=g.n,
        heat_capacity_total_J_per_K=C_total, rho_cp_roll=c.rho_cp_roll,
    )
    return out


if __name__ == "__main__":
    a = audit()
    print(json.dumps(a, indent=1, default=float))

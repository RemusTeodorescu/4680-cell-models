"""
================================================================================
 2-D THERMAL-ELECTROCHEMICAL (DFN + T) FINITE-DIFFERENCE / CONTROL-VOLUME MODEL
 W.B. Gu and C.Y. Wang, "Thermal-Electrochemical Coupled Modeling of a
 Lithium-Ion Cell".
================================================================================

Governing equations implemented (paper numbering)
-------------------------------------------------
  [3]  Butler-Volmer            i_n  = i_o [exp(a_a F eta/RT) - exp(-a_c F eta/RT)]
  [4]  Exchange current         i_o  = k ce^aa (cs_max - cse)^aa cse^ac
  [5]  Surface overpotential    eta  = phis - phie - U - i_n Rf
  [6]  OCP temperature shift    U    = U_ref + (T - T_ref) dU/dT
  [7]  U_ref  LiyMn2O4 (positive electrode)
  [8]  U_ref  LixC6    (negative electrode)
  [9]  theta = cse / cs_max
  [10] Electrolyte Li+   d(eps_e ce)/dt = div(De_eff grad ce) + (1-t+) j/F
  [11] Bruggeman         De_eff = De eps_e^1.5
  [12] j^Li = a_s i_n in the electrodes, 0 in the separator
  [13] Solid Li          d(eps_s cs)/dt = -j/F              (see SIGN NOTE)
  [14] Particle balance   Ds (cse - cs)/l_se = -j/(a_s F)   (see SIGN NOTE)
  [15] l_se = rs/5
  [16] a_s  = 3 eps_s / rs
  [17] Electrolyte charge div(kap_eff grad phie) + div(kapD_eff grad ln ce) + j = 0
  [18] kapD_eff = 2 R T kap_eff/F (t+ - 1)(1 + dlnf/dlnce)
  [19] kappa(ce) polynomial
  [20] Solid charge       div(sig_eff grad phis) - j = 0
  [21] sig_eff = eps_s sigma
  [22] Energy             d(rho cp T)/dt = div(lam grad T) + q
  [23] rho cp = sum_k eps_k rho_k cp_k     (Table II already gives layer values)
  [24] lam    = sum_k eps_k lam_k          (Table II already gives layer values)
  [25] q = a_s i_n (phis - phie - U + T dU/dT)
           + sig_eff grad phis . grad phis
           + kap_eff grad phie . grad phie + kapD_eff grad ln ce . grad phie
  [26] Arrhenius   Phi = Phi_ref exp[(Eact/R)(1/T_ref - 1/T)]
  [27] uniform initial conditions
  [28] d ce/dn = d phie/dn = 0 on all boundaries
  [29] at y = H over the tabs:  -sig_eff dphis/dy = I,  -lam dT/dy = h(T - Ta)
  [30] all other boundaries:    dphis/dn = 0,           -lam dT/dn = h(T - Ta)

--------------------------------------------------------------------------------
SIGN NOTE
--------------------------------------------------------------------------------
Eqs. 13 and 14 are printed with +j/F and +j/(a_s F).  That is inconsistent with
Eqs. 10, 17, 20 and 25, which all use j^Li = a_s i_n with i_n > 0 for anodic
(oxidation) current.  With that convention the positive electrode has i_n < 0
during discharge while lithium is being inserted, so the solid-phase source must
be -j/F.  This code uses the consistent signs.  Set
Params.paper_literal_signs_13_14 = True to reproduce the printed form.

--------------------------------------------------------------------------------
OCP NOTE
--------------------------------------------------------------------------------
Eq. 7 prints the tanh prefactor as 0.565661.  The original source (Doyle et al.,
JES 143, 1890 (1996)) gives 0.0565661.  The printed value yields ~4.7 V OCV at
the stated initial SOC, contradicting the paper's own Fig. 2 (~4.2 V).  The
corrected value is the default; set Params.U_pos_tanh_coeff = 0.565661 to use
the printed one.

Table II labels U = "Eq. 7" under Anode and "Eq. 8" under Cathode.  Per the text
and Eq. 12, subscript 1 = LiyMn2O4 (positive, Eq. 7) and 2 = LixC6 (negative,
Eq. 8); the table columns are swapped.  The code follows the text.

--------------------------------------------------------------------------------
NUMERICS
--------------------------------------------------------------------------------
* Cell-centred control volumes: layer-conforming non-uniform x grid x uniform y
  grid.  Interface transport coefficients use the harmonic (series-resistance)
  mean so discontinuous properties are handled exactly and fluxes are
  conservative.
* Unknowns per control volume: ce, cs, cse, phie, phis, T.  A variable only owns
  a DOF where its phase exists (no electrolyte in the collectors, no solid-phase
  potential in the separator, no particles outside the electrodes).
* Backward Euler in time, with ONE global Newton solve per step over all
  unknowns simultaneously -> genuinely coupled, not segregated.
* Use exactly ONE control volume across each metal current collector.  With two
  or more, the interior x-face conductance inside the metal is O(1e10 S) and the
  residual of Eq. 20 is evaluated as the difference of numbers ~1e10 times
  larger than itself; the resulting cancellation puts a ~1e-4 floor under the
  Newton residual.  One cell removes those faces entirely and the solver
  converges quadratically to ~1e-11.  The collectors are equipotential across
  their thickness anyway, so nothing physical is lost.
* Jacobian by graph-coloured sparse finite differences on the exact 5-point
  stencil; SuperLU linear solves; backtracking line search; adaptive dt.

CGS units throughout, as in the paper: cm, s, A, V, mol, K, J, W.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize._numdiff import approx_derivative, group_columns

# ------------------------------------------------------------------ constants
F_FARADAY = 96487.0      # C/mol
R_GAS = 8.3143           # J/mol/K
T_REF = 298.15           # K -- Table II properties are quoted at 25 degC

CC_NEG, ANODE, SEP, CATHODE, CC_POS = 0, 1, 2, 3, 4
REGION_NAME = {CC_NEG: "Cu", ANODE: "anode", SEP: "separator",
               CATHODE: "cathode", CC_POS: "Al"}


# ============================================================================ #
#  PARAMETERS (Table II)                                                       #
# ============================================================================ #
@dataclass
class Layer:
    name: str
    kind: int
    L: float                 # cm
    rho: float               # g/cm^3
    cp: float                # J/g/K
    lam: float               # W/cm/K
    sigma: float = 0.0       # S/cm
    eps_e: float = 0.0
    eps_p: float = 0.0
    eps_f: float = 0.0
    rs: float = 0.0          # cm
    cs_max: float = 0.0      # mol/cm^3
    theta0: float = 0.0
    Ds: float = 0.0          # cm^2/s
    i0: float = 0.0          # A/cm^2 at the reference state
    alpha_a: float = 0.5
    alpha_c: float = 0.5
    Eact_i0: float = 0.0     # J/mol
    Eact_Ds: float = 0.0
    Eact_sig: float = 0.0
    dUdT: float = 0.0        # V/K
    # Alternative kinetics form.  If m_ref > 0 it replaces the Eq. 4 route via
    # a reference i0:   i0 = m_ref * ce^aa * (cs_max - cse)^aa * cse^ac
    # with m_ref given directly in CGS, (A/cm^2)(cm^3/mol)^(aa+ac).  This is how
    # modern parameter sets (Chen 2020 etc.) report kinetics.
    m_ref: float = 0.0
    # Explicit active-material fraction.  Modern sets quote eps_act and porosity
    # independently (binder/carbon are lumped), so eps_s is not 1-eps_e-eps_p-eps_f.
    eps_s_explicit: float = 0.0
    # Usable stoichiometry window (defaults = full utilisation, as Gu & Wang assume)
    theta_min: float = 0.0
    theta_max: float = 1.0

    @property
    def eps_s(self) -> float:
        if self.eps_s_explicit > 0.0:
            return self.eps_s_explicit
        return (1.0 - self.eps_e - self.eps_p - self.eps_f
                if self.kind in (ANODE, CATHODE) else 0.0)

    @property
    def a_s(self) -> float:                       # Eq. 16
        return 3.0 * self.eps_s / self.rs if self.rs > 0 else 0.0

    @property
    def l_se(self) -> float:                      # Eq. 15
        return self.rs / 5.0 if self.rs > 0 else 0.0

    @property
    def sigma_eff(self) -> float:                 # Eq. 21
        return (self.eps_s * self.sigma
                if self.kind in (ANODE, CATHODE) else self.sigma)

    @property
    def rho_cp(self) -> float:                    # Eq. 23
        return self.rho * self.cp


def default_layers() -> List[Layer]:
    """Table II of Gu & Wang."""
    um = 1.0e-4
    return [
        Layer("Cu", CC_NEG, L=10 * um, rho=9.0, cp=0.381, lam=3.8, sigma=6.0e5),
        Layer("anode", ANODE, L=128 * um, rho=2.5, cp=0.7, lam=0.05,
              sigma=1.0, eps_e=0.357, eps_p=0.146, eps_f=0.026,
              rs=12.5 * um, cs_max=0.02639, theta0=0.53, Ds=3.9e-10,
              i0=0.11e-3, Eact_i0=30e3, Eact_Ds=4e3, dUdT=0.0),
        Layer("separator", SEP, L=76 * um, rho=1.2, cp=0.7, lam=0.01,
              eps_e=0.724, eps_p=0.276, eps_f=0.0),
        Layer("cathode", CATHODE, L=190 * um, rho=1.5, cp=0.7, lam=0.05,
              sigma=0.038, eps_e=0.444, eps_p=0.186, eps_f=0.073,
              rs=8.5 * um, cs_max=0.02286, theta0=0.17, Ds=1.0e-9,
              i0=0.08e-3, Eact_i0=30e3, Eact_Ds=20e3, dUdT=0.0),
        Layer("Al", CC_POS, L=15 * um, rho=2.7, cp=0.870, lam=2.0, sigma=3.8e5),
    ]


@dataclass
class Params:
    layers: List[Layer] = field(default_factory=default_layers)

    # geometry
    H: float = 50.0            # cell height, cm
    depth: float = 1.0         # out-of-plane depth, cm

    # electrolyte
    ce0: float = 2.0e-3        # mol/cm^3
    De: float = 7.5e-7         # cm^2/s
    t_plus: float = 0.363
    Eact_De: float = 10e3      # J/mol
    Eact_kap: float = 20e3     # J/mol
    dlnf_dlnce: float = 0.0

    # operating conditions
    C_rate: float = 3.0
    T0: float = 298.15
    Ta: float = 298.15
    h_tab: float = np.inf      # W/cm^2/K, inf -> isothermal tab
    h_side: float = 0.0        # W/cm^2/K, 0 -> adiabatic sides (base case)
    # Per-face overrides for h_side.  None -> use h_side.  Needed whenever a
    # boundary of the computational domain is a symmetry plane rather than a
    # real surface (e.g. the half-symmetric double-side-coated stack of a wound
    # cell, whose x = 0 and x = L faces are foil centrelines and must be
    # adiabatic no matter how the outside of the cell is cooled).
    h_x: Optional[float] = None      # x = 0 and x = L faces
    h_y0: Optional[float] = None     # y = 0 face
    h_yH: Optional[float] = None     # y = H face, away from the tabs
    V_cut: float = 2.2

    # switches
    U_pos_tanh_coeff: float = 0.0565661
    paper_literal_signs_13_14: bool = False
    coupled: bool = True       # False -> properties frozen at T_ref (decoupled)
    isothermal: bool = False   # True  -> skip Eq. 22, T = T0
    Rf: float = 0.0            # ohm cm^2 (Eq. 5)
    # ---- particle closure (DFNT vs DDFNT) -----------------------------------
    # 'parabolic'  Gu & Wang Eqs. 14-15: the algebraic quasi-steady closure
    #              Ds (cse - cs)/l_se = -j/(a_s F), l_se = rs/5.  The surface
    #              follows the local current instantly.  This is DFNT.
    # 'quartic'    Subramanian, Diwakar & Tapriyal (2005) three-parameter
    #              profile.  Adds one state per particle control volume, the
    #              volume-averaged concentration gradient, so the surface
    #              responds to a current step partly at once (1/7) and partly
    #              with time constant rs^2/(30 Ds).  Reduces exactly to Eq. 14
    #              at constant current.  This is DDFNT (dynamic DFNT).
    particle_closure: str = "parabolic"
    # ---- where the reaction heat evaluates the open-circuit potential -----
    # 'surface'  Gu & Wang Eq. 25: U at the particle SURFACE concentration.
    #            The concentration overpotential inside the particles,
    #            U(c_mean) - U(c_surface), is then not heat: physically it is
    #            released later as heat of mixing, which Eq. 25 omits.
    #            Default, unchanged.
    # 'mean'     U at the particle MEAN concentration: the whole
    #            concentration overpotential counts as heat when it is
    #            incurred. The two options bracket the treatment of that
    #            energy: 'surface' counts none of it, 'mean' all of it at once.
    heat_ocp: str = "surface"

    # ---- pluggable correlations -------------------------------------------
    # All default to the Gu & Wang / Doyle 1996 expressions above.  Supplying a
    # callable swaps the chemistry without touching the solver.  Every callable
    # works in the CGS units of this module: ce in mol/cm^3, kappa in S/cm,
    # De in cm^2/s, U in V, theta dimensionless.
    ocp_neg: Optional[Callable] = None      # U_neg(theta) -> V
    ocp_pos: Optional[Callable] = None      # U_pos(theta) -> V
    # entropic coefficients as functions of stoichiometry, V/K. When given they
    # replace the constant Layer.dUdT of that electrode, both in the OCP's
    # temperature shift and in the reversible heat (evaluated where the OCP
    # is: the surface for the reaction, the mean for heat_ocp="mean").
    dUdT_neg_fn: Optional[Callable] = None  # dU_neg/dT(theta) -> V/K
    dUdT_pos_fn: Optional[Callable] = None  # dU_pos/dT(theta) -> V/K
    kappa_fn: Optional[Callable] = None     # kappa(ce)    -> S/cm
    De_fn: Optional[Callable] = None        # De(ce)       -> cm^2/s

    # ---- current collection ------------------------------------------------
    # 'tab'      Eq. 29 as written: current enters/leaves at y = H only, so the
    #            full length of the collector carries current along y.
    # 'tabless'  current is collected uniformly along y (the foil is welded to
    #            an end disc over the whole winding length).  The y-direction
    #            foil resistance then drops out of the 2-D problem; whatever
    #            remains (the short path across the electrode height, plus the
    #            end discs) belongs in R_series.
    collection: str = "tab"
    R_series: float = 0.0      # ohm, lumped series resistance for the whole cell
    # Conductance of the end-disc rail that shorts each collector along the
    # winding in the tabless case, S.  Only needs to be large compared with the
    # cell's own internal conductance (~100 S); 1e6 leaves ~20 uV across the
    # rail at 25 A, far below anything that matters, without the round-off
    # trouble that a true short would cause.
    G_rail: float = 1.0e6

    # ---- thermal topology --------------------------------------------------
    # 'cartesian'    y is a real Cartesian coordinate (the prismatic cell of
    #                Gu & Wang).  Conduction along y uses the layer lambda.
    # 'wound_radial' y is the UNROLLED WINDING LENGTH of a cylindrical cell.
    #                Then y is NOT a heat-transfer coordinate: two windings that
    #                are physically adjacent (one period apart radially) sit a
    #                whole turn apart in y.  In this mode the y-direction
    #                conduction is replaced by the correct cylindrical-shell
    #                conductance between successive wraps,
    #                    G = 2 pi lambda_r H_e / ln(r_{j+1}/r_j),
    #                with lambda_r the series (through-stack) average, and the
    #                outermost row is coupled to ambient through the can:
    #                    G_can = [ln(R_out/r_last)/(2 pi lambda_r H_e)
    #                              + 1/(h_can 2 pi R_out H_e)]^-1.
    #                The x-direction still resolves one half-stack between
    #                symmetry planes, so it carries no net radial flux and does
    #                not double count.
    thermal_topology: str = "cartesian"
    roll_r_core: float = 0.0   # cm, mandrel radius
    roll_r_out: float = 0.0    # cm, jelly-roll outer radius
    roll_height: float = 0.0   # cm, coated electrode height (out-of-plane)
    roll_period: float = 0.0   # cm, full winding period (both coated faces)
    h_can: float = 0.0         # W/cm^2/K at the can outer surface
    # 'two_node' (with thermal_topology='wound_radial', and normally ny=1): the
    #            roll is ONE thermal node at its heat-capacity-weighted mean
    #            temperature, joined to a massless SURFACE node by the exact
    #            internal resistance of a uniformly heated annulus with an
    #            adiabatic mandrel (two_node_R_int), and the surface node to
    #            ambient by h_can. The electrochemistry sees the roll node's
    #            temperature; the surface node is what a thermocouple on the can
    #            reads. Paper 1's validation model. Off by default, so every
    #            distributed result is unchanged.
    two_node: bool = False
    # exterior area the surface node convects through, cm^2. 0 keeps the
    # distributed model's side-over-roll-height area; paper 1's two-node model
    # sets the whole can exterior (side over the can height plus both ends),
    # because the chamber fan cools all of it -- as validate_lumped.py and the
    # bench predictions already assume.
    two_node_A_ext: float = 0.0
    # Heat capacity of the surface (can) node, J/K. 0 keeps the massless
    # surface node. Paper 1 (27 Sep 2026): the roll carries its own layers'
    # heat capacity and the can node the rest of the cell (can, end discs,
    # tabs, free electrolyte), so that roll + can = m c_p of the whole cell.
    # The node is eliminated exactly inside each backward-Euler step.
    two_node_C_can: float = 0.0
    # I^2 R_series heat in the thermal balance (paper 1, 27 Sep 2026). The
    # series resistance is the cell's own loss; its foil share (series_R_roll,
    # ohm) heats the roll, the remainder (tabs, welds, end discs) the can node.
    # Off by default: the heat is then outside the domain, as before.
    series_heat: bool = False
    series_R_roll: float = 0.0
    # extra constant power at the can node, W (sensitivity: terminal contacts)
    P_can_ext: float = 0.0
    kin_reg: float = 0.0       # >0: smooth floor on (cs_max-cse), cse in i0, fraction of cs_max
    R_heat_only: float = 0.0   # ohm; I^2 R heat in the can node that is NOT in the terminal voltage (holder contacts outside Kelvin sensing), bench calibration 30 Sep 2026

    # ---- effective OCP / transport accessors ------------------------------
    def U_neg(self, theta):
        return self.ocp_neg(theta) if self.ocp_neg else U_neg_ref(theta)

    def U_pos(self, theta):
        return (self.ocp_pos(theta) if self.ocp_pos
                else U_pos_ref(theta, self.U_pos_tanh_coeff))

    def kappa_of(self, ce):
        return self.kappa_fn(ce) if self.kappa_fn else kappa_ce(ce)

    def De_of(self, ce):
        return self.De_fn(ce) if self.De_fn else self.De * np.ones_like(ce)

    def layer(self, kind: int) -> Layer:
        for L in self.layers:
            if L.kind == kind:
                return L
        raise KeyError(REGION_NAME[kind])

    @property
    def L_cell(self) -> float:
        return sum(L.L for L in self.layers)

    def electrode_capacity(self, kind: int) -> float:
        """Usable capacity of one electrode, C per cm^2 of separator area.
        theta_min/theta_max bound the usable stoichiometry window; they default
        to 0/1, which reproduces the Gu & Wang assumption of full utilisation."""
        L = self.layer(kind)
        span = (L.theta0 - L.theta_min) if kind == ANODE else (L.theta_max - L.theta0)
        return L.eps_s * L.L * L.cs_max * span * F_FARADAY

    def areal_capacity(self) -> float:
        """Capacity of the limiting electrode, C per cm^2 of separator area."""
        return min(self.electrode_capacity(ANODE), self.electrode_capacity(CATHODE))

    @property
    def np_ratio(self) -> float:
        """Negative-to-positive capacity ratio (design margin against plating)."""
        return self.electrode_capacity(ANODE) / self.electrode_capacity(CATHODE)

    @property
    def separator_area(self) -> float:
        """Total electrochemically active area, cm^2 (H x depth)."""
        return self.H * self.depth

    @property
    def capacity_Ah(self) -> float:
        return self.areal_capacity() * self.separator_area / 3600.0

    @property
    def i_app(self) -> float:
        """A/cm^2 of separator (electrode) area."""
        return self.C_rate * self.areal_capacity() / 3600.0

    @property
    def I_cell(self) -> float:
        """Total cell current, A."""
        return self.i_app * self.separator_area

    @property
    def I_tab(self) -> float:
        """Eq. 29 current density at the tab face, A/cm^2 of tab cross-section."""
        return self.i_app * self.H / self.layer(CC_POS).L


# ============================================================================ #
#  MATERIAL FUNCTIONS                                                          #
# ============================================================================ #
def arrhenius(phi_ref, Eact, T):
    """Eq. 26; Eact may be scalar or array."""
    E = np.asarray(Eact, dtype=float)
    expo = np.where(E > 0.0, (E / R_GAS) * (1.0 / T_REF - 1.0 / T), 0.0)
    return phi_ref * np.exp(np.clip(expo, -50.0, 50.0))


def kappa_ce(ce):
    """Eq. 19, S/cm, ce in mol/cm^3."""
    c = np.clip(ce, 1.0e-9, 8.0e-3)
    return (4.1253e-4 + 5.007 * c - 4.7212e3 * c ** 2
            + 1.5094e6 * c ** 3 - 1.6018e8 * c ** 4)


def U_pos_ref(theta, tanh_coeff=0.0565661):
    """Eq. 7 -- LiyMn2O4 OCP at 25 degC."""
    th = np.clip(theta, 1.0e-6, 0.99)
    return (4.19829
            + tanh_coeff * np.tanh(-14.5546 * th + 8.60942)
            - 0.0275479 * (1.0 / (0.998432 - th) ** 0.492465 - 1.90111)
            - 0.157123 * np.exp(-0.04738 * th ** 8)
            + 0.810239 * np.exp(np.clip(-40.0 * (th - 0.133875), -60.0, 60.0)))


def U_neg_ref(theta):
    """Eq. 8 -- LixC6 (coke) OCP at 25 degC."""
    th = np.clip(theta, 1.0e-6, 1.0)
    return (-0.16
            + 1.32 * np.exp(np.clip(-3.0 * th, -60.0, 60.0))
            + 10.0 * np.exp(np.clip(-2000.0 * th, -60.0, 60.0)))


# ============================================================================ #
#  MESH                                                                        #
# ============================================================================ #
class Mesh2D:
    def __init__(self, params: Params, nx_per_layer: List[int], ny: int):
        self.p = params
        self.ny = ny
        dx, region = [], []
        for L, n in zip(params.layers, nx_per_layer):
            dx.extend([L.L / n] * n)
            region.extend([L.kind] * n)
        self.dx = np.asarray(dx, dtype=float)
        self.region = np.asarray(region, dtype=int)
        self.nx = self.dx.size
        self.xf = np.concatenate([[0.0], np.cumsum(self.dx)])
        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.dy = params.H / ny
        self.yc = (np.arange(ny) + 0.5) * self.dy
        self.dz = params.depth
        self.H = params.H

        self.V = np.tile(self.dx * self.dy * self.dz, (ny, 1))
        self.Ax = self.dy * self.dz
        self.Ay = np.tile(self.dx * self.dz, (ny, 1))
        self.DX = np.tile(self.dx, (ny, 1))

        self.mask_e = np.isin(self.region, [ANODE, SEP, CATHODE])
        self.mask_s = np.isin(self.region, [CC_NEG, ANODE, CATHODE, CC_POS])
        self.mask_p = np.isin(self.region, [ANODE, CATHODE])
        self.is_tab_neg = self.region == CC_NEG
        self.is_tab_pos = self.region == CC_POS
        self.col_layer = [params.layer(r) for r in self.region]

    def col_array(self, attr) -> np.ndarray:
        return np.asarray([getattr(L, attr) for L in self.col_layer], dtype=float)


# ============================================================================ #
#  FACE OPERATORS                                                              #
# ============================================================================ #
def face_conductance_x(k, mesh: Mesh2D):
    """Harmonic conductance of internal x-faces -> (ny, nx-1)."""
    kL, kR = k[:, :-1], k[:, 1:]
    dxL, dxR = mesh.DX[:, :-1], mesh.DX[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rL = np.where(kL > 0.0, dxL / (2.0 * np.where(kL > 0, kL, 1.0)), np.inf)
        rR = np.where(kR > 0.0, dxR / (2.0 * np.where(kR > 0, kR, 1.0)), np.inf)
    r = rL + rR
    return np.where(np.isfinite(r) & (r > 0.0), mesh.Ax / np.where(r > 0, r, 1.0), 0.0)


def face_conductance_y(k, mesh: Mesh2D):
    """Harmonic conductance of internal y-faces -> (ny-1, nx)."""
    if mesh.ny == 1:
        return np.zeros((0, mesh.nx))
    kB, kT = k[:-1, :], k[1:, :]
    h = 0.5 * mesh.dy
    with np.errstate(divide="ignore", invalid="ignore"):
        rB = np.where(kB > 0.0, h / np.where(kB > 0, kB, 1.0), np.inf)
        rT = np.where(kT > 0.0, h / np.where(kT > 0, kT, 1.0), np.inf)
    r = rB + rT
    A = mesh.Ay[:-1, :]
    return np.where(np.isfinite(r) & (r > 0.0), A / np.where(r > 0, r, 1.0), 0.0)


def _solve_equilibrated(J, b, n_pass=2):
    """Row/column equilibrated sparse solve.  The metal current collectors have
    sigma ~ 4e5 S/cm next to sigma_eff ~ 1e-2 S/cm in the electrodes, so the raw
    Jacobian has a condition number ~1e14; equilibration recovers ~1e-12
    accuracy in the Newton update."""
    A = sp.csr_matrix(J, copy=True)
    n = A.shape[0]
    dr = np.ones(n)
    dc = np.ones(n)
    for _ in range(n_pass):
        r = np.asarray(np.abs(A).max(axis=1).todense()).ravel()
        r[r == 0.0] = 1.0
        A = sp.diags(1.0 / r) @ A
        dr *= 1.0 / r
        c = np.asarray(np.abs(A).max(axis=0).todense()).ravel()
        c[c == 0.0] = 1.0
        A = A @ sp.diags(1.0 / c)
        dc *= 1.0 / c
    y = spla.spsolve(sp.csc_matrix(A), dr * b)
    return dc * y


def divergence(u, Gx, Gy):
    """sum_faces G (u_nb - u_P)  ==  integral_V div(k grad u) dV."""
    d = np.zeros_like(u)
    fx = Gx * (u[:, 1:] - u[:, :-1])
    d[:, :-1] += fx
    d[:, 1:] -= fx
    if Gy.size:
        fy = Gy * (u[1:, :] - u[:-1, :])
        d[:-1, :] += fy
        d[1:, :] -= fy
    return d


def face_quad(a, b, Gx, Gy):
    """(1/2) sum_faces G da db, split between the two adjacent cells;
    approximates integral_V k grad(a).grad(b) dV."""
    d = np.zeros_like(a)
    qx = 0.5 * Gx * (a[:, 1:] - a[:, :-1]) * (b[:, 1:] - b[:, :-1])
    d[:, :-1] += qx
    d[:, 1:] += qx
    if Gy.size:
        qy = 0.5 * Gy * (a[1:, :] - a[:-1, :]) * (b[1:, :] - b[:-1, :])
        d[:-1, :] += qy
        d[1:, :] += qy
    return d


# ============================================================================ #
#  MODEL                                                                       #
# ============================================================================ #
def two_node_R_int(r_i, r_o, lam, H):
    """Volume-mean minus outer-surface temperature per watt, K/W, for an
    annulus r_i..r_o of height H and conductivity lam with uniform heat
    generation and an adiabatic inner face (the mandrel):

        T(r) - T(r_o) = q/(2 lam) [ (r_o^2 - r^2)/2 - r_i^2 ln(r_o/r) ]

    averaged over the annulus and divided by Q = q pi (r_o^2 - r_i^2) H. Reduces
    to 1/(8 pi lam H) for a solid cylinder. Units follow the arguments (cm and
    W/cm/K give K/W)."""
    a2 = r_o ** 2 - r_i ** 2
    br = a2 ** 2 / 4.0 - 2.0 * r_i ** 2 * (a2 / 4.0 - r_i ** 2 / 2.0 * np.log(r_o / r_i))
    return br / (2.0 * lam * np.pi * H * a2 ** 2)


class GuWang2D:
    VARS = ("ce", "cs", "cse", "phie", "phis", "T")

    def __init__(self, params: Params, nx_per_layer=(1, 16, 8, 20, 1), ny=20):
        self.p = params
        self.m = Mesh2D(params, list(nx_per_layer), ny)
        assert params.t_plus < 1.0, "kapD_eff sign handling assumes t+ < 1"
        if params.heat_ocp not in ("surface", "mean"):
            raise ValueError(f"unknown heat_ocp {params.heat_ocp!r}")
        if params.particle_closure not in ("parabolic", "quartic"):
            raise ValueError(f"unknown particle_closure {params.particle_closure!r}")
        self.quartic = params.particle_closure == "quartic"
        # DDFNT carries one extra unknown per particle control volume:
        # qs = rs * (volume-averaged radial concentration gradient), mol/cm^3.
        self.VARS = GuWang2D.VARS + (("qs",) if self.quartic else ())
        self._build_dofs()
        self._build_static()
        self._S = None
        self._groups = None

    # ------------------------------------------------------------------ DOFs
    def _build_dofs(self):
        m, ny, nx = self.m, self.m.ny, self.m.nx
        self.masks = {"ce": m.mask_e, "cs": m.mask_p, "cse": m.mask_p,
                      "phie": m.mask_e, "phis": m.mask_s,
                      "T": np.ones(nx, dtype=bool), "qs": m.mask_p}
        self.idx: Dict[str, np.ndarray] = {}
        n = 0
        for v in self.VARS:
            I = -np.ones((ny, nx), dtype=int)
            if not (v == "T" and self.p.isothermal):
                cols = np.where(self.masks[v])[0]
                I[:, cols] = np.arange(n, n + cols.size * ny).reshape(ny, cols.size)
                n += cols.size * ny
            self.idx[v] = I
        self.ndof = n

    # -------------------------------------------------------- static property
    def _build_static(self):
        m, p = self.m, self.p
        ny, nx = m.ny, m.nx
        col = m.col_array
        tile = lambda a: np.tile(a, (ny, 1))

        self.eps_e = tile(col("eps_e"))
        self.eps_s = tile(col("eps_s"))
        self.a_s = tile(col("a_s"))
        self.a_s_safe = np.where(self.a_s > 0, self.a_s, 1.0)
        self.l_se = tile(np.where(col("l_se") > 0, col("l_se"), 1.0))
        self.cs_max = tile(np.where(col("cs_max") > 0, col("cs_max"), 1.0))
        self.sig_eff = tile(col("sigma_eff"))
        self.rho_cp = tile(col("rho_cp"))
        self.lam = tile(col("lam"))
        self.alpha_a = tile(col("alpha_a"))
        self.alpha_c = tile(col("alpha_c"))
        self.Ds_ref = tile(col("Ds"))
        self.Eact_Ds = tile(col("Eact_Ds"))
        self.Eact_i0 = tile(col("Eact_i0"))
        self.dUdT = tile(col("dUdT"))
        self.is_anode = tile(m.region == ANODE)
        self.mask_p2 = tile(m.mask_p)

        # Eq. 4: kinetic constant.  Either given directly as m_ref, or back-
        # calculated from a reference exchange current density as Gu & Wang do.
        kk = np.zeros(nx)
        for i, L in enumerate(m.col_layer):
            if L.kind not in (ANODE, CATHODE):
                continue
            if L.m_ref > 0.0:
                kk[i] = L.m_ref
            else:
                cse0 = L.theta0 * L.cs_max
                kk[i] = L.i0 / (p.ce0 ** L.alpha_a
                                * (L.cs_max - cse0) ** L.alpha_a
                                * cse0 ** L.alpha_c)
        self.k_react = tile(kk)

        # static conductances (lam and sigma have no T dependence here)
        self.GxT = face_conductance_x(self.lam, m)
        self.GyT = face_conductance_y(self.lam, m)
        self.lam_radial = m.dx.sum() / np.sum(m.dx / col("lam"))
        self.roll_r = None
        if p.thermal_topology == "wound_radial":
            self._wound_radial_conduction()
        elif p.thermal_topology != "cartesian":
            raise ValueError(f"unknown thermal_topology {p.thermal_topology!r}")
        self.Gxs = face_conductance_x(self.sig_eff, m)
        self.Gys = face_conductance_y(self.sig_eff, m)

        # ---- current collection ------------------------------------------
        # 'tab'     : Eq. 29 exactly -- injection and grounding at y = H only,
        #             so the collector carries the whole current along y.
        # 'tabless' : injection and grounding distributed uniformly along y, so
        #             the y-direction foil resistance leaves the problem.  Both
        #             cases inject the same total current, i_app * H * depth.
        self.G_phis_dir = np.zeros((ny, nx))
        self.A_tab_pos = np.zeros((ny, nx))
        neg, pos = m.is_tab_neg, m.is_tab_pos
        if p.collection == "tabless":
            self.G_phis_dir[:, neg] = (self.sig_eff[:, neg] * m.Ay[:, neg]
                                       / (0.5 * m.dy))
            self.A_tab_pos[:, pos] = m.Ay[:, pos] / ny
            # The end disc welded along the whole winding shorts the foil in y.
            # Without it, a per-row current injection would FORCE every winding
            # position to carry the same current, and the coldest wrap (which is
            # also the most polarised) would hit the cut-off while the rest of
            # the cell still had capacity -- an artefact, not physics.  The rail
            # makes each collector equipotential so current redistributes to the
            # hotter, easier wraps, which is what a tabless cell actually does.
            # Its own resistance is negligible and is carried in R_series.
            if m.ny > 1:
                self.Gys[:, neg | pos] = p.G_rail
        elif p.collection == "tab":
            self.G_phis_dir[-1, neg] = (self.sig_eff[-1, neg] * m.Ay[-1, neg]
                                        / (0.5 * m.dy))
            self.A_tab_pos[-1, pos] = m.Ay[-1, pos]
        else:
            raise ValueError(f"unknown collection mode {p.collection!r}")

        # thermal boundary conductances (Eqs. 29, 30)
        self.G_bcT = np.zeros((ny, nx))
        self._thermal_bc()

        # ---- hooks for coupling this slab into a larger problem -------------
        # Both default to None, i.e. to the standalone behaviour, so nothing
        # below changes unless a caller sets them. They exist so that an outer
        # solver (see gw3d.py) can attach this 2-D station to neighbours in a
        # third direction without the 2-D physics knowing anything about it.
        #
        #   T_env  per-cell environment temperature seen through G_bcT, in K.
        #          Scalar p.Ta when None. An axial neighbour is imposed by
        #          folding its conductance into G_bcT and its temperature into
        #          the G-weighted mean here, which is exact for a linear
        #          conduction term.
        #   P_ext  per-cell external power, in W (NOT volumetric). Used for
        #          heat that enters from outside the electrochemistry, such as
        #          terminal contact resistance, which is a boundary effect and
        #          must not be smeared through the heat-generation expression.
        self.T_env = None
        self.P_ext = None

        self.scale = {"ce": p.ce0,
                      "cs": float(col("cs_max").max()),
                      "cse": float(col("cs_max").max()),
                      "phie": 1.0, "phis": 1.0, "T": 100.0,
                      "qs": float(col("cs_max").max())}
        # particle radius, rs = 5 l_se (Eq. 15); 5.0 outside the electrodes
        self.r_s = 5.0 * self.l_se

        Vref = float(np.mean(m.V))
        self._Vref = Vref
        self._jscale = max(p.i_app, 1e-12) / p.layer(CATHODE).L * Vref
        Dsm = float(np.mean(self.Ds_ref[self.Ds_ref > 0]))
        lsm = float(np.mean(self.l_se[self.mask_p2]))
        self._csescale = Dsm * float(col("cs_max").max()) / lsm

    def _wound_radial_conduction(self):
        """Replace the along-y conduction by radial conduction between wraps.

        The Archimedean-spiral map from unrolled length s to radius is
            s(r) = pi (r^2 - r_core^2) / period   ->   r(s) = sqrt(r_core^2 + p s / pi)
        Each model row therefore sits at a known radius, and successive rows are
        successive wrap groups.  Conductances are per model row, i.e. they carry
        the whole cell's radial heat flow, and are distributed over the row's
        control volumes in proportion to their volume so the discrete operator
        stays conservative.
        """
        m, p = self.m, self.p
        for name, v in (("roll_r_core", p.roll_r_core), ("roll_r_out", p.roll_r_out),
                        ("roll_height", p.roll_height), ("roll_period", p.roll_period)):
            if v <= 0.0:
                raise ValueError(f"thermal_topology='wound_radial' needs {name} > 0")
        lam_r = self.lam_radial
        r_of = lambda s: np.sqrt(p.roll_r_core ** 2 + p.roll_period * s / np.pi)
        rc = r_of(m.yc)                                   # row centre radii
        self.roll_r = rc

        # inter-row (radial) conductance, W/K, one per gap
        G_row = 2.0 * np.pi * lam_r * p.roll_height / np.log(rc[1:] / rc[:-1])
        # distribute each row-to-row conductance over the x cells by volume share
        w = m.V[0, :] / m.V[0, :].sum()
        self.GyT = G_row[:, None] * w[None, :]

        # outermost wrap -> can -> ambient
        R_out = max(p.roll_r_out, rc[-1] * 1.0001)
        if p.two_node:
            # roll node -> surface node: the exact mean-to-surface resistance of
            # the whole annulus, so the single row stands for the whole roll
            R_cond = two_node_R_int(p.roll_r_core, R_out, lam_r, p.roll_height)
        else:
            R_cond = np.log(R_out / rc[-1]) / (2.0 * np.pi * lam_r * p.roll_height)
        A_conv = (p.two_node_A_ext if (p.two_node and p.two_node_A_ext > 0.0)
                  else 2.0 * np.pi * R_out * p.roll_height)
        R_conv = (np.inf if p.h_can <= 0.0 else 1.0 / (p.h_can * A_conv))
        self.G_can_total = 0.0 if not np.isfinite(R_conv) else 1.0 / (R_cond + R_conv)
        # Can node with heat capacity: the roll then sees the can node through
        # R_cond only; the can node sees ambient through R_conv. Handled in
        # residual(), so the static boundary conductance is left out here.
        self.can_node = bool(p.two_node and p.two_node_C_can > 0.0)
        if self.can_node:
            self.G_can_total = 0.0
            self.G_int = 1.0 / R_cond
            self.hA_can = 0.0 if not np.isfinite(R_conv) else 1.0 / R_conv
            self.w_can = np.zeros_like(m.V)
            self.w_can[-1, :] = m.V[-1, :] / m.V[-1, :].sum()
            self.T_can = p.T0
        # kept for post-processing: lets the outer-wrap temperature be projected
        # onto the can surface, and gives the lumped thermal time constant
        self.R_cond_out, self.R_conv_can, self.roll_R_out = R_cond, R_conv, R_out

    def thermal_time_constant(self):
        """Lumped tau = C_th / G_can, s.  Sets how far a discharge gets towards
        a steady temperature: a discharge lasting << tau is effectively
        adiabatic, >> tau reaches a steady profile."""
        C_th = float((self.rho_cp * self.m.V).sum())
        if getattr(self, "can_node", False):
            C_th += self.p.two_node_C_can
            G = 1.0 / (1.0 / self.G_int + (np.inf if self.hA_can <= 0.0
                                           else 1.0 / self.hA_can))
            return np.inf if G <= 0.0 else C_th / G
        G = getattr(self, "G_can_total", 0.0)
        return np.inf if G <= 0.0 else C_th / G

    def _series_heat(self):
        """(can-node W, roll W) of I^2 R_series, plus the external can power."""
        p = self.p
        Qc = p.P_can_ext + p.I_cell ** 2 * p.R_heat_only
        if not p.series_heat:
            return Qc, 0.0
        I2 = p.I_cell ** 2
        return (Qc + I2 * max(p.R_series - p.series_R_roll, 0.0),
                I2 * p.series_R_roll)

    def accept_step(self, u, dt):
        """Advance the can node after an accepted step (with the new roll
        temperature, as in residual()). No-op without a can node."""
        if not getattr(self, "can_node", False) or self.p.isothermal:
            return
        p = self.p
        Tr = float((self.w_can * self.state(u)["T"]).sum())
        Cd = p.two_node_C_can / dt
        Qc, _ = self._series_heat()
        self.T_can = ((Cd * self.T_can + self.G_int * Tr + self.hA_can * p.Ta + Qc)
                      / (Cd + self.G_int + self.hA_can))

    def surface_temperature(self, T_roll_mean):
        """Can-surface temperature, K: the can node when it has heat capacity,
        else the projection of the massless surface node."""
        if getattr(self, "can_node", False):
            return self.T_can
        return self.can_temperature(T_roll_mean)

    def can_temperature(self, T_outer_row):
        """Project the outermost wrap temperature onto the can outer surface by
        splitting the series conduction/convection resistance."""
        if not np.isfinite(self.R_conv_can):
            return T_outer_row
        f = self.R_conv_can / (self.R_cond_out + self.R_conv_can)
        return self.p.Ta + (T_outer_row - self.p.Ta) * f

    def _thermal_bc(self):
        m, p = self.m, self.p
        G = self.G_bcT

        def hface(h, lam, A, d):
            lam = np.asarray(lam, dtype=float)
            A = np.asarray(A, dtype=float)
            if np.isinf(h):
                return lam * A / d                     # Dirichlet T = Ta
            if h == 0.0:
                return np.zeros_like(lam * A)
            return h * A * lam / (lam + h * d)         # convection + half-cell

        if p.thermal_topology == "wound_radial":
            # The only real external surface is the can, reached from the
            # outermost wrap.  Everything else (x faces, y = 0) is a symmetry
            # plane or the mandrel: adiabatic.
            w = m.V[-1, :] / m.V[-1, :].sum()
            G[-1, :] += self.G_can_total * w
            return

        pick = lambda h: p.h_side if h is None else h
        h_x, h_y0, h_yH = pick(p.h_x), pick(p.h_y0), pick(p.h_yH)
        d = 0.5 * m.dy
        # y = H: the tab columns get h_tab (Eq. 29); the rest get h_yH (Eq. 30).
        # With tabless collection there is no tab, so the whole face uses h_yH.
        if p.collection == "tabless":
            G[-1, :] += hface(h_yH, self.lam[-1, :], m.Ay[-1, :], d)
        else:
            is_tab = m.is_tab_neg | m.is_tab_pos
            G[-1, :] += np.where(is_tab,
                                 hface(p.h_tab, self.lam[-1, :], m.Ay[-1, :], d),
                                 hface(h_yH, self.lam[-1, :], m.Ay[-1, :], d))
        G[0, :] += hface(h_y0, self.lam[0, :], m.Ay[0, :], d)
        G[:, 0] += hface(h_x, self.lam[:, 0], m.dy * m.dz, 0.5 * m.dx[0])
        G[:, -1] += hface(h_x, self.lam[:, -1], m.dy * m.dz, 0.5 * m.dx[-1])

    # ------------------------------------------------------------ pack/unpack
    def initial_state(self) -> np.ndarray:
        """Eq. 27 for ce, cs, T.  The potentials are algebraic (no time
        derivative), so they are pre-loaded with the analytic uniform-reaction
        estimate -- otherwise the very first Newton solve has to travel through
        the Butler-Volmer exponential from eta = 0."""
        m, p = self.m, self.p
        ny, nx = m.ny, m.nx
        a, c = p.layer(ANODE), p.layer(CATHODE)
        Ua0 = float(p.U_neg(a.theta0))
        Uc0 = float(p.U_pos(c.theta0))

        RTF2 = 2.0 * R_GAS * p.T0 / F_FARADAY

        def consistent(L, i_n, U_fun):
            """Eq. 14 at the target uniform reaction current, then Eqs. 3-5."""
            cs0 = L.theta0 * L.cs_max
            cse = cs0 - L.l_se * i_n / (L.Ds * F_FARADAY)
            cse = float(np.clip(cse, 1e-4 * L.cs_max, 0.999 * L.cs_max))
            if L.m_ref > 0.0:
                k = L.m_ref
            else:
                k = L.i0 / (p.ce0 ** L.alpha_a * (L.cs_max - cs0) ** L.alpha_a
                            * cs0 ** L.alpha_c)
            i0 = (k * p.ce0 ** L.alpha_a * (L.cs_max - cse) ** L.alpha_a
                  * cse ** L.alpha_c)
            eta = RTF2 * np.arcsinh(i_n / (2.0 * i0))
            return cse, float(U_fun(cse / L.cs_max)), float(eta)

        ina = p.i_app / (a.a_s * a.L)             # anodic   (> 0 on discharge)
        inc = -p.i_app / (c.a_s * c.L)            # cathodic (< 0 on discharge)
        cse_a, Ua, eta_a = consistent(a, ina, p.U_neg)
        cse_c, Uc, eta_c = consistent(c, inc, p.U_pos)
        phie0 = -Ua - eta_a
        phis_pos = phie0 + Uc + eta_c

        CS = np.tile(m.col_array("theta0") * m.col_array("cs_max"), (ny, 1))
        CSE = CS.copy()
        CSE[:, m.region == ANODE] = cse_a
        CSE[:, m.region == CATHODE] = cse_c
        neg_side = np.tile(np.isin(m.region, [CC_NEG, ANODE]), (ny, 1))
        d = dict(ce=np.full((ny, nx), p.ce0),
                 cs=CS, cse=CSE,
                 phie=np.full((ny, nx), phie0),
                 phis=np.where(neg_side, 0.0, phis_pos),
                 T=np.full((ny, nx), p.T0),
                 qs=np.zeros((ny, nx)))      # DDFNT: particles start relaxed
        self.OCV0 = Uc0 - Ua0
        return self.pack(d)

    def pack(self, d) -> np.ndarray:
        u = np.zeros(self.ndof)
        for v in self.VARS:
            I = self.idx[v]
            sel = I >= 0
            if sel.any():
                u[I[sel]] = d[v][sel] / self.scale[v]
        return u

    def unpack(self, u) -> Dict[str, np.ndarray]:
        m, p = self.m, self.p
        ny, nx = m.ny, m.nx
        defaults = {"ce": p.ce0, "cs": 1.0, "cse": 1.0,
                    "phie": 0.0, "phis": 0.0, "T": p.T0, "qs": 0.0}
        out = {}
        for v in self.VARS:
            I = self.idx[v]
            A = np.full((ny, nx), defaults[v], dtype=float)
            sel = I >= 0
            if sel.any():
                A[sel] = u[I[sel]] * self.scale[v]
            out[v] = A
        return out

    # ---------------------------------------------------------------- physics
    def _properties(self, ce, T):
        p = self.p
        Tp = T if p.coupled else np.full_like(T, T_REF)
        De_eff = arrhenius(p.De_of(ce), p.Eact_De, Tp) * self.eps_e ** 1.5  # Eqs. 11, 26
        kap_eff = arrhenius(p.kappa_of(ce), p.Eact_kap, Tp) * self.eps_e ** 1.5
        kapD_mag = (2.0 * R_GAS * T * kap_eff / F_FARADAY
                    * (1.0 - p.t_plus) * (1.0 + p.dlnf_dlnce))          # Eq. 18, |.|
        Ds = arrhenius(self.Ds_ref, self.Eact_Ds, Tp)
        kr = arrhenius(self.k_react, self.Eact_i0, Tp)
        return De_eff, kap_eff, kapD_mag, Ds, kr

    def _dUdT(self, theta):
        """Entropic coefficient at stoichiometry theta, V/K: the constant
        Layer.dUdT unless Params supplies dUdT_neg_fn / dUdT_pos_fn."""
        p = self.p
        if p.dUdT_neg_fn is None and p.dUdT_pos_fn is None:
            return self.dUdT
        out = self.dUdT
        if p.dUdT_neg_fn is not None:
            out = np.where(self.is_anode, p.dUdT_neg_fn(theta), out)
        if p.dUdT_pos_fn is not None:
            out = np.where(self.is_anode, out, p.dUdT_pos_fn(theta))
        return np.where(self.mask_p2, out, 0.0)

    def _U_heat(self, cs, U_surface, T):
        """OCP entering the reaction heat (see Params.heat_ocp) and the
        entropic coefficient at the same concentration."""
        if self.p.heat_ocp == "surface":
            return U_surface
        th = np.clip(cs / self.cs_max, 1.0e-6, 1.0 - 1.0e-6)
        U = np.where(self.is_anode, self.p.U_neg(th), self.p.U_pos(th))
        return U + (T - T_REF) * self._dUdT(th)

    def _dUdT_heat(self, cs, cse):
        """Entropic coefficient entering the reversible heat, evaluated at
        the same concentration as the OCP of the reaction heat."""
        c = cs if self.p.heat_ocp == "mean" else cse
        return self._dUdT(np.clip(c / self.cs_max, 1.0e-6, 1.0 - 1.0e-6))

    def _reaction(self, ce, cse, phie, phis, T, kr):
        p = self.p
        theta = np.clip(cse / self.cs_max, 1.0e-6, 1.0 - 1.0e-6)
        U = np.where(self.is_anode, p.U_neg(theta), p.U_pos(theta))     # Eqs. 7-9
        U = U + (T - T_REF) * self._dUdT(theta)                         # Eq. 6
        if p.kin_reg > 0.0:
            # smooth floor (bench calibration, 30 Sep 2026): keeps i0 and its
            # derivative non-zero as the particle surface saturates, so Newton
            # can follow the end-of-discharge knee down to 2.0 V
            d = p.kin_reg * self.cs_max
            sp = lambda x: 0.5 * (x + np.sqrt(x * x + d * d))
            i0 = (kr * np.clip(ce, 1e-12, None) ** self.alpha_a
                  * sp(self.cs_max - cse) ** self.alpha_a * sp(cse) ** self.alpha_c)
        else:
            i0 = (kr * np.clip(ce, 1e-12, None) ** self.alpha_a
                  * np.clip(self.cs_max - cse, 1e-12, None) ** self.alpha_a
                  * np.clip(cse, 1e-12, None) ** self.alpha_c)              # Eq. 4
        eta = phis - phie - U                                           # Eq. 5
        FRT = F_FARADAY / (R_GAS * T)
        i_n = i0 * (np.exp(np.clip(self.alpha_a * FRT * eta, -60, 60))
                    - np.exp(np.clip(-self.alpha_c * FRT * eta, -60, 60)))  # Eq. 3
        i_n = np.where(self.mask_p2, i_n, 0.0)
        return i_n, self.a_s * i_n, U, eta                              # Eq. 12

    def state(self, u) -> Dict[str, np.ndarray]:
        """All derived fields for a given solution vector (no time derivative)."""
        m, p = self.m, self.p
        s = self.unpack(u)
        ce, cs, cse = s["ce"], s["cs"], s["cse"]
        phie, phis, T = s["phie"], s["phis"], s["T"]
        if p.isothermal:
            T = np.full_like(T, p.T0)
        De_eff, kap_eff, kapD_mag, Ds, kr = self._properties(ce, T)
        i_n, jLi, U, eta = self._reaction(ce, cse, phie, phis, T, kr)

        Gxe, Gye = face_conductance_x(kap_eff, m), face_conductance_y(kap_eff, m)
        GxD = -face_conductance_x(kapD_mag, m)
        GyD = -face_conductance_y(kapD_mag, m)
        lnce = np.log(np.clip(ce, 1e-12, None))

        q_rxn = jLi * (phis - phie - self._U_heat(cs, U, T)
                       + T * self._dUdT_heat(cs, cse))
        q_js = face_quad(phis, phis, self.Gxs, self.Gys) / m.V
        q_je = face_quad(phie, phie, Gxe, Gye) / m.V
        q_jd = face_quad(lnce, phie, GxD, GyD) / m.V
        return dict(ce=ce, cs=cs, cse=cse, phie=phie, phis=phis, T=T,
                    qs=s.get("qs", np.zeros_like(ce)),
                    i_n=i_n, jLi=jLi, U=U, eta=eta,
                    theta=np.clip(cse / self.cs_max, 0, 1),
                    kap_eff=kap_eff, De_eff=De_eff, Ds=Ds,
                    q_rxn=q_rxn, q_joule_s=q_js, q_joule_e=q_je + q_jd,
                    q=q_rxn + q_js + q_je + q_jd,
                    Gxe=Gxe, Gye=Gye, GxD=GxD, GyD=GyD)

    # --------------------------------------------------------------- residual
    def residual(self, u, u_old, dt):
        m, p = self.m, self.p
        s = self.unpack(u)
        s_old = self.unpack(u_old)
        ce, cs, cse = s["ce"], s["cs"], s["cse"]
        phie, phis = s["phie"], s["phis"]
        T = np.full_like(s["T"], p.T0) if p.isothermal else s["T"]
        V = m.V

        De_eff, kap_eff, kapD_mag, Ds, kr = self._properties(ce, T)
        i_n, jLi, U, eta = self._reaction(ce, cse, phie, phis, T, kr)
        sgn = 1.0 if p.paper_literal_signs_13_14 else -1.0

        Gxc, Gyc = face_conductance_x(De_eff, m), face_conductance_y(De_eff, m)
        Gxe, Gye = face_conductance_x(kap_eff, m), face_conductance_y(kap_eff, m)
        GxD = -face_conductance_x(kapD_mag, m)      # kapD_eff < 0 for t+ < 1
        GyD = -face_conductance_y(kapD_mag, m)
        lnce = np.log(np.clip(ce, 1e-12, None))

        R = {}
        # Eq. 10
        R["ce"] = (self.eps_e * V * (ce - s_old["ce"]) / dt
                   - divergence(ce, Gxc, Gyc)
                   - (1.0 - p.t_plus) / F_FARADAY * jLi * V)
        # Eq. 13
        R["cs"] = (self.eps_s * V * (cs - s_old["cs"]) / dt
                   - sgn * jLi * V / F_FARADAY)
        if not self.quartic:
            # Eq. 14 (DFNT)
            R["cse"] = Ds * (cse - cs) / self.l_se - sgn * jLi / (self.a_s_safe * F_FARADAY)
        else:
            # DDFNT: Subramanian et al. (2005) quartic profile replaces Eq. 14.
            # N is the molar flux OUT of the particle surface (mol/cm^2/s),
            # with the same sign convention as Eq. 13, and qs = rs * qbar:
            #   d(qs)/dt = -30 Ds qs / rs^2 - (45/2) N / rs
            #   35 Ds (cse - cs)/rs - 8 Ds qs/rs = -N
            # At steady state qs = -3 N rs/(4 Ds) and the second line becomes
            # Ds (cse - cs)/(rs/5) = -N, i.e. Eq. 14 exactly.  The algebraic
            # residual is divided by 7 so its scale matches Eq. 14's.
            rs, qs = self.r_s, s["qs"]
            N = -sgn * jLi / (self.a_s_safe * F_FARADAY)
            R["cse"] = (Ds * (35.0 * (cse - cs) - 8.0 * qs) / rs + N) / 7.0
            R["qs"] = (V * (qs - s_old["qs"]) / dt
                       + V * (30.0 * Ds * qs / rs ** 2 + 22.5 * N / rs))
        # Eq. 17
        R["phie"] = (divergence(phie, Gxe, Gye)
                     + divergence(lnce, GxD, GyD)
                     + jLi * V)
        # Eq. 20 with Eq. 29 tab BCs
        R["phis"] = (divergence(phis, self.Gxs, self.Gys)
                     + self.G_phis_dir * (0.0 - phis)
                     - self.A_tab_pos * p.I_tab
                     - jLi * V)
        # Eq. 25 + Eq. 22
        if p.isothermal:
            R["T"] = np.zeros_like(T)
        else:
            q = (jLi * (phis - phie - self._U_heat(cs, U, T)
                      + T * self._dUdT_heat(cs, cse))
                 + (face_quad(phis, phis, self.Gxs, self.Gys)
                    + face_quad(phie, phie, Gxe, Gye)
                    + face_quad(lnce, phie, GxD, GyD)) / V)
            T_env = p.Ta if self.T_env is None else self.T_env
            R["T"] = (self.rho_cp * V * (T - s_old["T"]) / dt
                      - divergence(T, self.GxT, self.GyT)
                      - self.G_bcT * (T_env - T)
                      - q * V)
            if self.P_ext is not None:
                R["T"] = R["T"] - self.P_ext
            if self.can_node:
                # Can node C_c dTc/dt = G_i (T_r - T_c) - hA (T_c - T_a) + Q_c,
                # backward Euler, solved for T_c and substituted: the roll sees
                # an effective conductance, environment and power. Exact.
                Gi, hA, Cd = self.G_int, self.hA_can, p.two_node_C_can / dt
                Qc, Qr = self._series_heat()
                D = Cd + Gi + hA
                G_eff = Gi * (Cd + hA) / D
                T_eff = (Cd * self.T_can + hA * p.Ta) / max(Cd + hA, 1e-300)
                R["T"] = (R["T"] - self.w_can * (G_eff * (T_eff - T)
                                                 + Gi * Qc / D)
                          - Qr * V / V.sum())

        rs = self._scales(dt)
        res = np.zeros(self.ndof)
        for v in self.VARS:
            I = self.idx[v]
            sel = I >= 0
            if sel.any():
                res[I[sel]] = R[v][sel] / rs[v]
        return res

    def _scales(self, dt):
        Vr = self._Vref
        return {"ce": self.p.ce0 * Vr / dt,
                "cs": self.scale["cs"] * Vr / dt,
                "cse": self._csescale,
                "phie": self._jscale, "phis": self._jscale,
                "T": float(self.rho_cp.mean()) * Vr / dt,
                "qs": self.scale["qs"] * Vr / dt}

    # --------------------------------------------------------------- Jacobian
    def build_sparsity(self):
        if self._S is not None:
            return self._S, self._groups
        ny, nx = self.m.ny, self.m.nx
        cell = [[[] for _ in range(nx)] for _ in range(ny)]
        for v in self.VARS:
            I = self.idx[v]
            for j in range(ny):
                for i in range(nx):
                    if I[j, i] >= 0:
                        cell[j][i].append(int(I[j, i]))
        rows, cols = [], []
        for j in range(ny):
            for i in range(nx):
                tgt = cell[j][i]
                if not tgt:
                    continue
                st = [(j, i)]
                if i > 0:
                    st.append((j, i - 1))
                if i < nx - 1:
                    st.append((j, i + 1))
                if j > 0:
                    st.append((j - 1, i))
                if j < ny - 1:
                    st.append((j + 1, i))
                src = [d for (a, b) in st for d in cell[a][b]]
                for r in tgt:
                    rows.extend([r] * len(src))
                    cols.extend(src)
        S = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                          shape=(self.ndof, self.ndof))
        S.data[:] = 1
        self._S, self._groups = S, group_columns(S)
        return self._S, self._groups

    def jacobian(self, u, u_old, dt, f0=None):
        S, groups = self.build_sparsity()
        if f0 is None:
            f0 = self.residual(u, u_old, dt)
        J = approx_derivative(lambda z: self.residual(z, u_old, dt), u,
                              method="2-point", f0=f0, sparsity=(S, groups))
        return sp.csc_matrix(J), f0

    # ----------------------------------------------------------------- Newton
    #  Globalisation: (i) a physical step limiter (bounded change of the phase
    #  potentials, temperature and concentrations per iteration -- essential
    #  because Butler-Volmer is exponential and an undamped Newton step
    #  overflows it), then (ii) a backtracking line search on the 2-norm.
    DPHI_MAX = 0.12      # V per Newton iteration (keeps a_F eta/RT bounded)
    DT_MAX_IT = 8.0      # K per Newton iteration
    DC_FRAC = 0.4        # max fractional change of a concentration
    GROWTH_MAX = 1.0e3   # tolerated transient growth of ||R||_2

    def _limit_step(self, u, du):
        lam = 1.0
        for v, cap in (("phie", self.DPHI_MAX), ("phis", self.DPHI_MAX),
                       ("T", self.DT_MAX_IT)):
            I = self.idx[v]
            sel = I >= 0
            if not sel.any():
                continue
            mx = float(np.abs(du[I[sel]]).max()) * self.scale[v]
            if mx > cap:
                lam = min(lam, cap / mx)
        for v in ("ce", "cs", "cse"):
            I = self.idx[v]
            sel = I >= 0
            if not sel.any():
                continue
            x, d = u[I[sel]], du[I[sel]]
            bad = d < -self.DC_FRAC * np.abs(x)
            if np.any(bad):
                lam = min(lam, float(np.min(-self.DC_FRAC * np.abs(x[bad]) / d[bad])))
            # Keep the intercalation fraction inside (0, 1): at either end Eq. 4
            # drives i_o -> 0 and the OCP derivative diverges, so an unbounded
            # step lands outside the physical range and kills Newton.  Only the
            # headroom that still exists is limited -- a cell already at the
            # bound is left to Newton to pull back, otherwise one saturated
            # control volume would freeze the whole step.
            if v in ("cs", "cse"):
                head = 0.999 * self.cs_max[sel] / self.scale[v] - x
                lim = (d > 0.0) & (head > 0.0)
                if np.any(lim):
                    lam = min(lam, float(np.min(0.8 * head[lim] / d[lim])))
        return max(lam, 1e-4)

    def step(self, u_old, dt, tol=5e-8, tol_relax=1e-6, maxit=25, verbose=False):
        u = u_old.copy()
        f = self.residual(u, u_old, dt)
        stall, prev_inf = 0, np.inf
        for it in range(maxit):
            n_inf = np.linalg.norm(f, np.inf)
            n_2 = np.linalg.norm(f)
            if verbose:
                print(f"      newton {it:2d}  |R|inf={n_inf:.3e}  |R|2={n_2:.3e}")
            if n_inf < tol:
                return u, True, it
            # stagnation at the finite-difference Jacobian noise floor
            if n_inf > 0.7 * prev_inf and n_inf < tol_relax:
                return u, True, it
            prev_inf = n_inf
            J, f = self.jacobian(u, u_old, dt, f0=f)
            try:
                du = _solve_equilibrated(J, -f)
            except Exception:
                return u, False, it
            if not np.all(np.isfinite(du)):
                return u, False, it
            # Safeguarded (not strictly monotone) damped Newton: the physical
            # limiter already bounds the step, so we only backtrack when the
            # residual blows up or goes non-finite.  Requiring monotone descent
            # here stalls, because the joule term of Eq. 25 is quadratic in the
            # potential increment and dominates the norm on the way to the
            # solution.
            lam0 = self._limit_step(u, du)
            lam, ok = lam0, False
            ut = ft = None
            for _ in range(12):
                ut = u + lam * du
                ft = self.residual(ut, u_old, dt)
                if np.all(np.isfinite(ft)) and \
                        np.linalg.norm(ft) < self.GROWTH_MAX * n_2:
                    ok = True
                    break
                lam *= 0.5
            if not ok:
                return u, False, it
            u, f = ut, ft
            if lam >= 0.999 and np.linalg.norm(f) >= n_2:
                stall += 1
                if stall > 6:
                    return u, np.linalg.norm(f, np.inf) < tol_relax, it
            else:
                stall = 0
        return u, np.linalg.norm(f, np.inf) < tol_relax, maxit

    # -------------------------------------------------------------------- run
    def run(self, t_end=None, dt0=0.02, dt_max=20.0, dV_max=0.02,
            n_snap=40, dod_max=105.0, verbose=True):
        """Backward-Euler march with dt controlled by the Newton iteration count
        and by the change in cell voltage per step (keeps the knee at the end of
        discharge resolved)."""
        p = self.p
        Q = p.areal_capacity()
        if t_end is None:
            t_end = 3600.0 / p.C_rate * 1.2
        u = self.initial_state()
        st = self.state(u)
        V_prev = self.cell_voltage(st)
        t, dt, nstep = 0.0, dt0, 0
        keys = ("t", "DOD", "V", "Tavg", "Tmax", "Tmin",
                "q_rxn", "q_joule", "q_tot", "OCV")
        hist = {k: [] for k in keys}
        self.snapshots = []
        self.snapshot_t = []
        next_snap, snap_every = 0.0, 100.0 / n_snap
        t0 = time.time()
        self.end_reason = "t_end"
        n_tiny = 0
        while t < t_end:
            dtl = min(dt, t_end - t)
            un, ok, nit = self.step(u, dtl)
            V = self.cell_voltage(self.state(un)) if ok else None
            reject = (not ok) or (abs(V - V_prev) > dV_max and dtl > 1.5 * dt0)
            # A step so small that Newton accepts the old state (nit = 0) is
            # not progress: reject-shrink-accept-grow cycles can then hold dt
            # near zero indefinitely without ever reaching the 1e-5 s test
            # below. Fifty consecutive sub-millisecond steps count as a
            # collapse.
            n_tiny = n_tiny + 1 if (not reject and dtl < 1e-3) else (
                n_tiny if reject else 0)
            if reject or n_tiny > 50:
                dt = 0.3 * dtl
                if dt < 1e-5 or n_tiny > 50:
                    # The knee at the end of discharge is genuinely stiff: i_o
                    # collapses as an electrode saturates.  If the last good
                    # state is already at the cutoff, or the particle surface
                    # of an electrode is saturated (Eq. 14 has no solution a
                    # step further: the solid-diffusion limit), stop cleanly
                    # rather than calling it a failure.
                    tail = (hist["V"][-1] if hist["V"] else np.inf)
                    sat = self._surface_saturated(u)
                    if tail <= p.V_cut + 0.15 or sat:
                        self.end_reason = ("saturation" if sat and
                                           tail > p.V_cut + 0.15 else "knee")
                        if verbose:
                            print(f"  cut-off ({self.end_reason}): V={tail:.4f}"
                                  f" V at DOD={hist['DOD'][-1]:.2f}%")
                    else:
                        self.end_reason = "collapse"
                        print(f"  ! time step collapsed at V={tail:.4f} V, "
                              f"DOD={hist['DOD'][-1] if hist['DOD'] else 0:.2f}%")
                    break
                continue
            u, t, nstep = un, t + dtl, nstep + 1
            st = self.state(u)
            dod = 100.0 * p.i_app * t / Q
            self._record(hist, t, dod, V, st)
            if dod >= next_snap:
                self.snapshots.append((dod, {k: st[k].copy() for k in
                                             ("T", "jLi", "ce", "phie", "phis",
                                              "theta", "q", "i_n", "U")}))
                self.snapshot_t.append(t)
                next_snap += snap_every
            if verbose and nstep % 25 == 0:
                print(f"  t={t:8.1f}s  DOD={dod:6.2f}%  V={V:6.4f}V  "
                      f"Tavg={hist['Tavg'][-1]-273.15:6.2f}C  "
                      f"dt={dtl:6.2f}s  nit={nit}")
            if V < p.V_cut or dod > dod_max:
                self.end_reason = "cutoff" if V < p.V_cut else "dod_max"
                if verbose:
                    print(f"  cut-off: V={V:.4f} V at DOD={dod:.2f}%")
                break
            g = 1.7 if nit <= 4 else (1.2 if nit <= 8 else 0.8)
            if abs(V - V_prev) > 0.5 * dV_max:
                g = min(g, 1.0)
            dt = min(dt * g, dt_max)
            V_prev = V
        if verbose:
            print(f"  {nstep} steps in {time.time()-t0:.1f} s")
        self.history = {k: np.asarray(v) for k, v in hist.items()}
        self.u_final = u
        self.failed = nstep == 0
        if self.failed:
            raise RuntimeError(
                "no time step converged: the requested current is not "
                "sustainable by this cell.  Check surface_saturation() -- if it "
                "reports a required theta outside (0, 1), Eq. 14 has no solution "
                "at this rate and the electrode surface saturates on contact.")
        return self.history

    def _surface_saturated(self, u, tol=1e-3):
        """True if, at state u, the particle surface of the electrode that is
        being filled is at (or past) its limit in any control volume: the
        cathode at cs_max on discharge, the anode at cs_max on charge.  Past
        that point Eq. 14 (or the quartic closure) cannot carry the local
        current and the terminal voltage falls vertically: the solid-diffusion
        end of discharge, not a numerical failure."""
        s = self.unpack(u)
        th = s["cse"] / self.cs_max
        filling = ~self.is_anode if self.p.I_cell >= 0 else self.is_anode
        mask = filling & self.mask_p2
        return bool(np.any(th[mask] >= 1.0 - tol))

    def surface_saturation(self):
        """Stoichiometry Eq. 14 demands at the particle surface for the applied
        current, assuming a uniform reaction distribution.  Values outside
        (0, 1) mean the rate is beyond what solid diffusion can supply and no
        steady solution exists -- the cell simply cannot hold the current."""
        p = self.p
        out = {}
        for kind, sgn in ((ANODE, +1.0), (CATHODE, -1.0)):
            L = p.layer(kind)
            i_n = sgn * p.i_app / (L.a_s * L.L)
            cs0 = L.theta0 * L.cs_max
            theta_se = (cs0 - L.l_se * i_n / (L.Ds * F_FARADAY)) / L.cs_max
            out[REGION_NAME[kind]] = float(theta_se)
        return out

    # ------------------------------------------------------------ diagnostics
    def cell_voltage(self, st):
        """Terminal voltage.  For 'tab' collection this is read at y = H, where
        Eq. 29 is applied; for 'tabless' the collectors are grounded/fed along
        the whole winding, so the average over y is the terminal pair.  R_series
        carries whatever ohmic path the 2-D domain does not resolve (foil-to-end-
        disc conduction across the electrode height, welds, cans, terminals)."""
        m, p = self.m, self.p
        if p.collection == "tabless":
            vp = float(np.mean(st["phis"][:, m.is_tab_pos]))
            vn = float(np.mean(st["phis"][:, m.is_tab_neg]))
        else:
            vp = float(np.mean(st["phis"][-1, m.is_tab_pos]))
            vn = float(np.mean(st["phis"][-1, m.is_tab_neg]))
        return vp - vn - p.I_cell * p.R_series

    def _record(self, hist, t, dod, V, st):
        Vc, m = self.m.V, self.m
        vol = float(Vc.sum())
        wcp = self.rho_cp * Vc
        hist["t"].append(t)
        hist["DOD"].append(dod)
        hist["V"].append(V)
        hist["Tavg"].append(float((st["T"] * wcp).sum() / wcp.sum()))
        hist["Tmax"].append(float(st["T"].max()))
        hist["Tmin"].append(float(st["T"].min()))
        hist["q_rxn"].append(float((st["q_rxn"] * Vc).sum()) / vol)
        hist["q_joule"].append(float(((st["q_joule_s"] + st["q_joule_e"]) * Vc).sum()) / vol)
        hist["q_tot"].append(float((st["q"] * Vc).sum()) / vol)
        # area-weighted OCV of the two electrodes (diagnostic)
        ja = self.is_anode & self.mask_p2
        jc = (~self.is_anode) & self.mask_p2
        hist["OCV"].append(float(st["U"][jc].mean() - st["U"][ja].mean()))

    def check_current_balance(self, u):
        """Total reaction current in each electrode vs. applied current."""
        st = self.state(u)
        Vc = self.m.V
        Ia = float((st["jLi"] * Vc)[self.is_anode & self.mask_p2].sum())
        Ic = float((st["jLi"] * Vc)[(~self.is_anode) & self.mask_p2].sum())
        Iapp = self.p.i_app * self.m.H * self.m.dz
        return dict(I_anode=Ia, I_cathode=Ic, I_applied=Iapp,
                    err_anode=(Ia - Iapp) / Iapp, err_cathode=(Ic + Iapp) / Iapp)

    def check_energy_balance(self, u):
        """Volumetric heat vs. I*(U_avg - V):  global consistency of Eq. 25."""
        st = self.state(u)
        Vc = self.m.V
        Q = float((st["q"] * Vc).sum())
        Iapp = self.p.i_app * self.m.H * self.m.dz
        ja = self.is_anode & self.mask_p2
        jc = (~self.is_anode) & self.mask_p2
        wa = np.abs(st["jLi"] * Vc)
        Uh = self._U_heat(st["cs"], st["U"], st["T"])      # = U for 'surface'
        Ua = float((Uh * wa)[ja].sum() / max(wa[ja].sum(), 1e-30))
        Uc = float((Uh * wa)[jc].sum() / max(wa[jc].sum(), 1e-30))
        # R_series sits OUTSIDE the computational domain, so its I^2 R heat is
        # not part of integral(q dV) and must be removed from the reference.
        Q_ext = Iapp ** 2 * self.p.R_series
        Q_ref = Iapp * ((Uc - Ua) - self.cell_voltage(st)) - Q_ext
        return dict(Q_model=Q, Q_IdV=Q_ref, Q_series=Q_ext,
                    rel_err=(Q - Q_ref) / max(abs(Q_ref), 1e-30))


# ============================================================================ #
if __name__ == "__main__":
    p = Params(C_rate=3.0, h_tab=np.inf, h_side=0.0)
    print(f"cell width     = {p.L_cell*1e4:8.1f} um")
    print(f"areal capacity = {p.areal_capacity()/3.6:8.3f} mAh/cm^2")
    print(f"i_app @ {p.C_rate:.0f}C   = {p.i_app*1e3:8.3f} mA/cm^2")
    print(f"I_tab          = {p.I_tab:8.2f} A/cm^2")
    mdl = GuWang2D(p, nx_per_layer=(1, 14, 7, 18, 1), ny=12)
    print(f"unknowns       = {mdl.ndof}")
    mdl.run(dt0=0.02, dt_max=20.0)

"""
================================================================================
 TESLA-STYLE 4680 CELL  --  NMC811 / graphite, liquid carbonate electrolyte
 2-D thermal-electrochemical (DFN + T) model, built on the solver in gw2d.py
================================================================================

This is a SECOND parameterisation of the same governing equations (Gu & Wang
Eqs. 10, 13, 14, 17-25 with closures 3-9, 11, 12, 15, 16, 18, 19, 21, 26).  Not
one line of the physics changes.  What changes is the chemistry, the transport
correlations, the layer stack, and -- most importantly -- how current is
collected.

--------------------------------------------------------------------------------
PROVENANCE TAGS.  READ THIS FIRST.
--------------------------------------------------------------------------------
No public source gives a complete, self-consistent DFN parameter set for a
Tesla 4680.  Every value below therefore carries a tag:

  [M]  MEASURED / REPORTED for the 4680 itself (teardown or manufacturer data)
  [L]  LITERATURE value for the same chemistry from a cited published set,
       transferred here on the assumption that a 4680 NMC811/graphite electrode
       behaves like the well-characterised LG M50 NMC811/graphite electrode
  [E]  EXTRAPOLATED / ESTIMATED BY THIS MODEL.  No source found.  The comment
       states exactly how the number was arrived at and what it is anchored to.

Anything tagged [E] is an engineering estimate, not data.  Treat every [E] value
as a fitting parameter, not as a fact about a real cell.  `provenance_report()`
prints the full audit trail, and `run_4680.py` writes it to disk.

--------------------------------------------------------------------------------
SOURCES
--------------------------------------------------------------------------------
[S1] Chen, Thornton, Marinescu, Drummond, Zhang, Offer et al.,
     "Development of Experimental Techniques for Parameterization of Multi-scale
     Lithium-ion Battery Models", J. Electrochem. Soc. 167, 080534 (2020).
     LG M50 21700, NMC811 / graphite.  https://doi.org/10.1149/1945-7111/ab9050
[S2] PyBaMM `Chen2020` parameter set -- the calibrated form of [S1] used in
     practice, including the OCP fits reproduced verbatim below.
[S3] Nyman, Behm, Lindbergh, "Electrochemical characterisation and modelling of
     the mass transport phenomena in LiPF6-EC-EMC electrolyte",
     Electrochim. Acta 53, 6356 (2008).
[S4] M. J. Lacey, "Commentary on the teardown of the Tesla 4680 cell" (2022):
     positive electrode ~3.3 m long, 70 mm wide, >5 mAh/cm^2, 24-25 Ah, NMC811
     with a graphite (no-silicon) anode, dry-coated anode, Cu foil down to
     4.5 um.  https://lacey.se/2022/07/18/commentary-tesla-4680-teardown
[S5] Battery Design, "Tesla 4680 Cell": 22-23.35 Ah, 3.7 V nominal, 46 x 80 mm,
     NMC811, cathode ~85 um, anode ~121 um, Al foil 9-10 um, Cu foil 6-8 um,
     cathode porosity 30-40 %, DCIR 5.4-7 mOhm, 0.355 kg.
     https://www.batterydesign.net/tesla-4680-cell/
[S6] Ank et al., "Lithium-Ion Cells in Automotive Applications: Tesla 4680
     Cylindrical Cell Teardown and Characterization", J. Electrochem. Soc. 170,
     120536 (2023).  Referenced for format-level geometry.

--------------------------------------------------------------------------------
CS_MAX NOTE -- why two different values circulate, and which one is used here
--------------------------------------------------------------------------------
[S1] Table VII MEASURES cs_max = 51555 mol/m^3 (NMC811) and 30555 (graphite).
The calibrated [S2] set instead uses 63104 and 33133, obtained by fitting the
whole model to discharge curves.  Both are "Chen2020" in common usage.

This file uses the MEASURED values for everything that determines capacity
(cs_max, active fractions, thicknesses) and the CALIBRATED values for the
kinetic and transport group (rs, Ds, m_ref), which were fitted together and are
not separable.  The reason is a consistency test: with the measured cs_max, the
independently reported 85 um cathode coating [S5] and the independently reported
capacity of 22-25 Ah [S4, S5] agree without any tuning -- the model predicts
5.04 mAh/cm^2 and 23.3 Ah.  With the calibrated cs_max the same 85 um coating
would give 5.95 mAh/cm^2 and 27.5 Ah, above every reported 4680 capacity.

Mixing two variants of one parameter set is not ideal and is flagged here
rather than hidden.  If it matters for your use, the honest move is to refit
Ds and m_ref against the measured cs_max.

--------------------------------------------------------------------------------
GEOMETRY: HOW A CYLINDRICAL JELLY ROLL BECOMES A 2-D SLAB
--------------------------------------------------------------------------------
The solver works on a Cartesian x-y slab.  The wound cell maps onto it exactly:

    x  = through-stack coordinate  (Cu | anode | separator | cathode | Al)
    y  = UNROLLED WINDING LENGTH   (0 = core, H = outer wrap at the can wall)
    z  = electrode height          (out-of-plane, the model's `depth`)

Two consequences worth being explicit about:

1.  HALF-SYMMETRIC STACK.  A wound electrode is coated on both faces, so the
    repeating unit is  Al | cath | sep | anode | Cu | anode | sep | cath | Al.
    The model resolves HALF of that -- from the Cu centreline to the Al
    centreline -- and doubles `depth` to 2 x 70 mm to account for the two coated
    faces.  The foil thicknesses below are therefore HALVED.  The x = 0 and
    x = L boundaries are symmetry planes, so `h_x = 0` (adiabatic) regardless of
    how the outside of the cell is cooled.  Getting this wrong double-counts the
    foil and halves the active area.

2.  ANISOTROPIC THERMAL CONDUCTION COMES OUT FOR FREE.  Because the metal foils
    are resolved as layers, conduction along y automatically picks up their high
    conductivity (parallel path, ~15 W/m/K) while conduction across x sees the
    coatings in series (~1.3 W/m/K).  No anisotropy tensor is needed.

--------------------------------------------------------------------------------
TABLESS COLLECTION -- WHY THIS MODEL IS WORTH RUNNING
--------------------------------------------------------------------------------
Gu & Wang's whole 2-D argument is about what happens when current has to travel
the full length of a current collector to reach a tab.  A 4680 is the industrial
answer to exactly that problem: the foils are left uncoated along one edge and
welded to end discs over the WHOLE winding length, so current leaves sideways
over ~35 mm instead of travelling 3.3 m.

In this model that is the `collection` switch:

    collection='tab'      Eq. 29 as written -- injection at y = H only.  The
                          model then resolves the full 3.3 m foil path.
    collection='tabless'  injection distributed uniformly along y.  The y-path
                          resistance leaves the problem; what remains is the
                          short cross-height path, which is lumped into
                          R_series (computed analytically in `foil_resistance`).

Running the same cell both ways quantifies what tabless buys.  See
`compare_tab_vs_tabless()`.

CGS units throughout (cm, s, A, V, mol, K, J, W), matching gw2d.py.
================================================================================
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from gw2d import (ANODE, CATHODE, CC_NEG, CC_POS, SEP, F_FARADAY, R_GAS, T_REF,
                  GuWang2D, Layer, Params)

UM = 1.0e-4       # cm per micrometre

# Provenance registry: name -> (tag, value, unit, note).  Filled as the module
# defines each number, then printed by provenance_report().
PROV: Dict[str, Tuple[str, object, str, str]] = {}


def _p(name, tag, value, unit, note):
    PROV[name] = (tag, value, unit, note)
    return value


# ============================================================================ #
#  OPEN-CIRCUIT POTENTIALS                                                     #
#  Both reproduced verbatim from the PyBaMM Chen2020 parameter set [S1, S2].   #
#  Argument is the stoichiometry theta = cse / cs_max, result in volts.        #
# ============================================================================ #
def dUdT_graphite_oregan2022(theta):
    """[L] Entropic coefficient of the LG M50 graphite(-SiOx) electrode,
    V/K, O'Regan, Brosa Planella, Widanage & Kendrick, Electrochim. Acta 425
    (2022) 140700 -- the same electrodes Chen2020 parameterised and this set
    transfers. Coefficients as in PyBaMM's ORegan2022 set (fit in mV/K; the
    tanh term is switched off there, as here)."""
    x = np.asarray(theta, dtype=float)
    a0, a2, b2, c0, c2 = -0.1112, 0.3561, 0.08309, 0.02914, 0.004616
    return (a0 * x + c0 + a2 * np.exp(-((x - b2) ** 2) / c2)) * 1e-3


def dUdT_nmc811_oregan2022(theta):
    """[L] Entropic coefficient of the LG M50 NMC811 electrode, V/K,
    O'Regan et al. (2022); coefficients as in PyBaMM's ORegan2022 set."""
    y = np.asarray(theta, dtype=float)
    a1, a2, b1, b2, c1, c2 = 0.04006, -0.06656, 0.2828, 0.8032, 0.0009855, 0.02179
    return (a1 * np.exp(-((y - b1) ** 2) / c1)
            + a2 * np.exp(-((y - b2) ** 2) / c2)) * 1e-3


# Window averages of the two entropic functions, for the parameter tables and
# as the constant fallback. Windows: the calibrated ones (THETA_* below),
# graphite 0.02741-0.70063, NMC811 0.26379-0.88946.
DUDT_NEG_AVG = float(np.mean(dUdT_graphite_oregan2022(np.linspace(0.02741, 0.70063, 2001))))
DUDT_POS_AVG = float(np.mean(dUdT_nmc811_oregan2022(np.linspace(0.26379, 0.88946, 2001))))

# Positive-electrode diffusion time tau_c = r_s^2 / D_s, s. [E]
# CALIBRATED 1 Oct 2026 (paper 1 v2, Section 5.1) on the 1C/2C charge test of
# Ank et al. (2023, Fig. 16): Nelder-Mead on the can temperature (mean of
# cap/centre/bottom) and the charge at 4.2 V, together with EACT_DS_POS_CAL,
# THETA_N0 and R_CONTACT_ANK below (data/calibration/nmc).
# Result 8031 s, within x1.2 of the transferred Chen2020 group,
# (5.22 um)^2 / 4.0e-15 m^2/s = 6812 s. The transferred set of the paper used
# 700 s (TAU_C_POS_WORKING), chosen below the resistance-based heat of
# About:Energy; the open thermal data revise that choice. The group, not D_s,
# is the declared quantity: Ank et al. report 3-16 um cathode particles.
TAU_C_POS = 8031.0
TAU_C_POS_WORKING = 700.0      # the transferred set of paper 1 v2 (step 1)
# Activation energy of the positive-electrode diffusivity, J/mol. [E]
# Calibrated with TAU_C_POS (transferred set: 25 kJ/mol, a typical NMC value).
EACT_DS_POS_CAL = 11.0e3
# Holder contact resistance, heat only (I^2 R at the can node, outside the
# voltage path), ohm. [E] Calibrated with TAU_C_POS; equals 0.5 mOhm per
# terminal, the value Ank et al. estimate for their holder. A property of the
# holder of the thermal test, not of the cell; zero in the plain parameter set.
R_CONTACT_ANK = 0.94e-3

# Stoichiometry windows, calibrated to the cut-off voltages (review B6,
# author, 27 Sep 2026; Birkl et al. 2015, Schmalstieg et al. 2018). The
# Chen2020 OCP functions and the electrode loadings are kept; the two 100 %-SOC
# stoichiometries are solved so that OCV(100 %) = 4.2 V and a C/20 discharge to
# 2.5 V delivers 22.31 Ah (Ank et al. 2023). p1_window_calibration.py. Checks
# not used: C/3 gives 22.083 Ah against 22.078 measured (+0.02 %); the graphite
# central dV/dQ peak falls at 19.6 Ah against about 14 Ah in Ank's DVA, which
# the transferred graphite-SiOx OCP shape does not reproduce (Ank: no Si).
THETA_N0, THETA_P0 = 0.70031, 0.26379          # [E] at 100 % SOC; anode 0.70063 from the window calibration, 0.70031 after the thermal calibration of 1 Oct 2026
THETA_N_END, THETA_P_END = 0.02741, 0.88946    # [D] at 2.5 V, C/20
TAU_C_POS_TRANSFERRED = (5.22e-4) ** 2 / 4.0e-11
TAU_C_POS_BOUND = 1400.0


def ocp_nmc811_chen2020(theta):
    """[L] NMC811 positive-electrode OCP, `nmc_LGM50_ocp_Chen2020` [S1, S2].

    Note the third and fourth terms nearly cancel (-17.7326 tanh(...) and
    +17.5842 tanh(...) with almost identical arguments): the fit encodes the
    steep low-stoichiometry rise as a small difference of two large tanh terms.
    Evaluate in double precision and do not "simplify" it.
    """
    s = np.clip(theta, 1.0e-6, 1.0 - 1.0e-6)
    return (-0.8090 * s
            + 4.4875
            - 0.0428 * np.tanh(18.5138 * (s - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (s - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (s - 0.3120)))


def ocp_graphite_chen2020(theta):
    """[L] Graphite negative-electrode OCP, `graphite_LGM50_ocp_Chen2020`
    [S1, S2].  The LG M50 anode is graphite with a few percent SiOx; [S4]
    reports the 4680 anode as graphite with no silicon, so this fit is
    transferred with that caveat -- the silicon contribution mainly tilts the
    high-stoichiometry end of the curve."""
    s = np.clip(theta, 1.0e-6, 1.0 - 1.0e-6)
    return (1.9793 * np.exp(np.clip(-39.3631 * s, -60.0, 60.0))
            + 0.2482
            - 0.0909 * np.tanh(29.8538 * (s - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (s - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (s - 0.6103)))


# ============================================================================ #
#  ELECTROLYTE: 1.0 M LiPF6 in EC:EMC 3:7                                      #
#  Nyman et al. 2008 [S3], as implemented in PyBaMM.  Published in SI with     #
#  ce in mol/m^3; converted here to the CGS units the solver uses.             #
# ============================================================================ #
def kappa_nyman2008(ce):
    """[L] Ionic conductivity, S/cm, with ce in mol/cm^3 [S3].

    SI form:  kappa[S/m] = 0.1297 c^3 - 2.5100 c^1.5 + 3.3290 c,  c = ce/1000
              with ce in mol/m^3.
    ce[mol/m^3] = 1e6 * ce[mol/cm^3], so c = 1000 * ce_cgs, and S/m -> S/cm
    divides by 100.  At 1 M this returns 9.49e-3 S/cm (0.949 S/m), about nine
    times the plasticised-gel value of Gu & Wang Eq. 19 -- which is why the
    electrolyte ohmic loss stops dominating the polarisation budget.
    """
    c = np.clip(ce, 1.0e-9, 5.0e-3) * 1.0e3           # -> mol/dm^3
    return (0.1297 * c ** 3 - 2.5100 * c ** 1.5 + 3.3290 * c) / 100.0


def De_nyman2008(ce):
    """[L] Salt diffusivity, cm^2/s, with ce in mol/cm^3 [S3].

    SI form:  De[m^2/s] = 8.794e-11 c^2 - 3.972e-10 c + 4.862e-10, c = ce/1000
    with ce in mol/m^3.  m^2/s -> cm^2/s multiplies by 1e4.  At 1 M this is
    1.77e-6 cm^2/s, about 2.4x the Gu & Wang gel value.
    """
    c = np.clip(ce, 1.0e-9, 5.0e-3) * 1.0e3           # -> mol/dm^3
    return (8.794e-11 * c ** 2 - 3.972e-10 * c + 4.862e-10) * 1.0e4


# ============================================================================ #
#  GEOMETRY                                                                    #
# ============================================================================ #
# ---- format-level, from teardown -------------------------------------------
WIND_LENGTH = _p("winding length H", "M", 326.7, "cm",
                 "[S6] Fig. 8 measures the cathode at 3267 mm unwound (the "
                 "anode is 3403 mm, 136 mm longer, and wraps it). The cathode "
                 "sets the active area, so it is the cathode length that "
                 "becomes the model's y extent: y=0 is the core, y=H the "
                 "outer wrap. [S4] reported ~3.3 m, consistent.")
ELECTRODE_HEIGHT = _p("electrode height", "M", 7.0, "cm",
                      "[S4] 70 mm coated width. The out-of-plane direction; "
                      "not resolved in x-y, it sets the active area and "
                      "R_series. Distinct from the mechanical roll height "
                      "below: the coating stops short of the foil edge, which "
                      "is where the collector flags are folded.")
ROLL_HEIGHT = _p("jelly-roll height", "M", 7.10, "cm",
                 "[S6] 71 mm measured after removing both collector disks. "
                 "This is the MECHANICAL height and it is what conducts heat "
                 "axially and carries thermal mass; the coated height above is "
                 "1 mm shorter and is what carries current. Conflating the two "
                 "overstates the active area by 1.4 %.")
DEPTH = _p("model depth", "M/E", 2.0 * ELECTRODE_HEIGHT, "cm",
           "2 x electrode height because the electrode is coated on BOTH faces "
           "and the model resolves one half-stack. Gives an active separator "
           "area of H x depth = 4620 cm^2.")

# ---- current collectors ------------------------------------------------------
T_CU_FULL = _p("Cu foil thickness (full)", "M", 8.0 * UM, "cm",
               "[S5] reports 6-8 um; [S4] notes 4.5 um foil entering "
               "production. 8 um chosen as the conservative (thicker, lower "
               "resistance is optimistic) end of the reported range.")
T_AL_FULL = _p("Al foil thickness (full)", "M", 10.0 * UM, "cm",
               "[S5] reports 9-10 um.")

# ---- coatings ----------------------------------------------------------------
# The cathode thickness is not a free choice: it is pinned by the reported areal
# capacity.  Solving  Q = eps_s * L * cs_max * (theta_max - theta0) * F  for L
# at Q = 5.2 mAh/cm^2 with the [L] cathode properties below gives L = 82 um,
# which sits right on the ~85 um of [S5].  That agreement is the main reason to
# believe the [L] transfer from the LG M50 is not wildly wrong.
T_CATHODE = _p("cathode coating thickness", "M", 85.0 * UM, "cm",
               "[S5] reports ~85 um, taken as measured rather than fitted. "
               "Inverting the capacity relation for L at the 5.2 mAh/cm^2 "
               "areal capacity of [S4] gives 82 um, within 4 % of it, and that "
               "independent agreement is the main evidence the [L] transfer "
               "from the LG M50 is sound. NOTE that the delivered capacity is "
               "NOT a further independent check: theta_max is calibrated "
               "against Ank's C/3 measurement, so the capacity agreement is "
               "arithmetic. See the UTILISATION NOTE.")
T_ANODE = _p("anode coating thickness", "M", 126.0 * UM, "cm",
             "[S6] Fig. 9: anode loading 19.5 mg/cm^2 per side at a coating "
             "density of 1.55 g/cm^3 gives 126 um per side, consistent with "
             "the 258-268 um double-sided thickness less the 8 um foil. "
             "Until 27 Sep 2026 this was an [E] 110 um chosen for N/P = 1.15; "
             "N/P is now an output of the loadings.")
# Anode solid fraction from the same loading: 19.5 mg/cm^2 over 126 um at the
# composite solid density of 2.20 g/cm^3 (RHO_S["anode"]); porosity follows.
# As in Chen2020, all anode solid is counted as active (binder not separated).
ANODE_LOADING = 19.5e-3                                  # g/cm^2 per side [M]
EPS_S_ANODE = ANODE_LOADING / (2.20 * T_ANODE)           # [D] 0.7035
T_SEP = _p("separator thickness", "L", 12.0 * UM, "cm",
           "[S1] LG M50 value. [S4] notes the 4680 separator is ceramic-coated "
           "on the cathode side, which this does not capture.")


# ============================================================================ #
#  LAYER STACK                                                                 #
# ============================================================================ #
# ---------------------------------------------------------------------------
#  UTILISATION NOTE -- why theta_max is calibrated and nothing else is
# ---------------------------------------------------------------------------
#  The delivered capacity comes out of four factors:
#
#      Q = eps_s * L * cs_max * (theta_max - theta0) * F * A
#
#  and against Ank et al.'s C/3 measurement of 22.078 Ah (sigma 186 mAh) the
#  model as first written gave 23.038 Ah -- high by 4.3 %, which is 5.2 sigma.
#  Something had to move. Which one is not a free choice:
#
#    A, the electrode area.  Ank MEASURE it. "The total length of the anode is
#      3403 mm, which is 136 mm longer than the cathode" puts the cathode at
#      3267 mm, which is exactly the WIND_LENGTH this module already carries.
#      Overruling a caliper reading to fix a model is not available.
#
#    cs_max.  A crystallographic property of NMC811, measured in [S1] Table VII.
#      Tuning it would also retract this module's own argument: it uses the
#      MEASURED cs_max variant precisely so that delivered capacity is a
#      PREDICTION. Fitting it afterwards would make that claim circular.
#
#    L and eps_s.  Measured coating thickness and porosity.
#
#    theta_max - theta0, the usable stoichiometry window.  Not measured, and
#      physically obliged to be narrower than the laboratory half-cell window:
#      a full cell cycled to a 2.5 V cut-off does not fully relithiate its
#      cathode. It was carrying the half-cell value 0.911 by default.
#
#  So the window is where the discrepancy belongs, and it takes a 4.17 % shrink
#  of the cathode span, 0.6449 -> 0.6180, i.e. theta_max 0.911 -> 0.8841.
#
#  WHAT THIS COSTS, STATED PLAINLY.  Delivered capacity is no longer a blind
#  prediction; it is matched to one measurement by one parameter, and the AREAL
#  capacity goes with it, because the two are the same number divided by the
#  same area (5.037 -> 4.827 mAh/cm^2 against Ank's 4.9). Neither can be quoted
#  as an independent prediction any more, and the paper must not try.
#
#  What DOES survive is the consistency check that motivated the measured
#  cs_max choice in the first place: with nothing adjusted at all, the measured
#  85 um coating and the measured cs_max deliver 23.038 Ah against a measured
#  22.078 Ah -- agreement to 4.3 %, from independent material and geometric
#  data that were never fitted to each other. That is a real result and it is
#  what the paper should claim. The calibration then absorbs the residual into
#  the single quantity that was never measured.
#
#  The calibration inherits the measurement's uncertainty. Ank's sigma of
#  186 mAh on 22.078 Ah is 0.84 %, so theta_max = 0.8841 +/- 0.0052; any
#  sensitivity study should carry that band rather than treating 0.8841 as
#  exact. The N/P ratio moves 1.149 -> 1.199 as a consequence, since the anode
#  window is untouched.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  PHASE INVENTORY -- solids and electrolyte counted once each
# ---------------------------------------------------------------------------
#  A porous electrode is two phases. Averaging them into one density and one
#  specific heat is fine, but the average has to be taken over the SAME
#  inventory for both, or mass and heat capacity describe different objects.
#  They previously did not: the densities here were solid-only while documented
#  as including electrolyte, so 34 g of pore electrolyte was missing from the
#  cell's mass, while the specific heats implied cp ~ 1.0 J/g/K for a graphite
#  electrode -- above graphite's 0.71, which is what you get if electrolyte IS
#  folded in. The effective properties are now COMPUTED from the phases below
#  and the porosity the electrochemistry already uses:
#
#      rho_eff    = rho_s (1 - eps_e) + rho_e eps_e
#      rho_cp_eff = rho_s cp_s (1 - eps_e) + rho_e cp_e eps_e
#
#  This raises the roll's thermal mass by 12.9 % (202.9 -> 229.1 J/K), which is
#  invisible in a voltage curve and fully visible in a temperature rise. See
#  stack4680.py for the closure audit against Ank et al.'s weighed 355.9 g.
RHO_ELY = 1.22       # g/cm^3  [L] LiPF6 in carbonate
CP_ELY = 1.518       # J/g/K   [L] Chen2020 electrolyte, 1518 J/kg/K
RHO_S = {"anode": 2.20, "cathode": 4.50, "separator": 0.91}   # [L] solid only
CP_S = {"anode": 0.75, "cathode": 0.84, "separator": 1.90}    # [L] solid only


def _eff(name, eps_e):
    """Volume-averaged density and rho*cp for a porous layer. [D]"""
    rs, cs = RHO_S[name], CP_S[name]
    rho = rs * (1.0 - eps_e) + RHO_ELY * eps_e
    rcp = rs * cs * (1.0 - eps_e) + RHO_ELY * CP_ELY * eps_e
    return rho, rcp / rho          # (rho_eff, cp_eff) so Layer's rho*cp is right


def layers_4680() -> List[Layer]:
    """Half-symmetric stack: Cu centreline -> anode -> separator -> cathode ->
    Al centreline.  Foil thicknesses are HALVED (see module docstring)."""
    return [
        # ---------------------------------------------------------- Cu (half)
        Layer(
            "Cu", CC_NEG,
            L=0.5 * T_CU_FULL,
            rho=8.96,          # [L] pure copper
            cp=0.385,          # [L] pure copper
            lam=4.00,          # [L] pure copper, 400 W/m/K
            sigma=5.96e5,      # [L] pure copper, 5.96e7 S/m
        ),
        # -------------------------------------------------------------- anode
        Layer(
            "anode", ANODE,
            L=T_ANODE,
            # --- thermal: [E] composite estimates, no 4680-specific source
            rho=_eff("anode", 1.0 - EPS_S_ANODE)[0],  # [D] 2.20 solid + 1.22 ely
            cp=_eff("anode", 1.0 - EPS_S_ANODE)[1],   # [D] at the [D] porosity
            lam=0.015,         # [E] 1.5 W/m/K through-plane composite
            # --- electrochemistry: [L] Chen2020 LG M50 negative electrode
            sigma=2.15,             # [L] 215 S/m
            eps_e=1.0 - EPS_S_ANODE,      # [D] 0.2965; Chen2020 [L] was 0.25
            eps_s_explicit=EPS_S_ANODE,   # [D] 0.7035 from the loading; was 0.75
            rs=5.86e-4,             # [L] 5.86 um
            cs_max=0.030555,        # [L] 30555 mol/m^3, [S1] Table VII MEASURED.
                                    #     See the CS_MAX NOTE below: the measured
                                    #     value is used for everything that sets
                                    #     capacity, because it is what makes the
                                    #     4680 thickness and Ah agree.
            theta0=THETA_N0,        # [E] WINDOW CALIBRATION (27 Sep 2026),
                                    #     see p1_window_calibration.py. Chen2020
                                    #     [L] gave 0.9014.
            theta_min=THETA_N_END,  # [D] end of the C/20 discharge to 2.5 V
            Ds=3.3e-10,             # [L] 3.3e-14 m^2/s
            m_ref=0.0648,           # [L] 6.48e-7 (A/m2)(m3/mol)^1.5 -> CGS x 1e5
            alpha_a=0.5, alpha_c=0.5,   # [L]
            Eact_i0=35.0e3,         # [E] typical graphite charge-transfer value;
                                    #     Chen2020 is isothermal and gives none
            Eact_Ds=30.0e3,         # [E] typical graphite solid-diffusion value
            dUdT=DUDT_NEG_AVG,      # [L] O'Regan 2022, window average. The
                                    #     solver uses the function of
                                    #     stoichiometry (Params.dUdT_neg_fn);
                                    #     this constant is the table value and
                                    #     the fallback when the function is off.
        ),
        # ---------------------------------------------------------- separator
        Layer(
            "separator", SEP,
            L=T_SEP,
            rho=_eff("separator", 0.47)[0],  # [D] 0.91 PP + 1.22 electrolyte
            cp=_eff("separator", 0.47)[1],   # [D] at eps_e = 0.47
            lam=0.0033,        # [E] 0.33 W/m/K, the stack's dominant resistance
            eps_e=0.47,        # [L] Chen2020
            eps_p=0.53,        # [L] remainder is polymer
        ),
        # ------------------------------------------------------------ cathode
        Layer(
            "cathode", CATHODE,
            L=T_CATHODE,
            # --- thermal: [E] composite estimates
            rho=_eff("cathode", 0.335)[0],   # [D] 4.50 solid + 1.22 electrolyte
            cp=_eff("cathode", 0.335)[1],    # [D] at eps_e = 0.335
            lam=0.015,         # [E] 1.5 W/m/K through-plane composite
            # --- electrochemistry: [L] Chen2020 LG M50 positive electrode
            sigma=0.0018,           # [L] 0.18 S/m
            eps_e=0.335,            # [L] porosity; [S5] reports 30-40 %, consistent
            eps_s_explicit=0.665,   # [L] active material volume fraction
            rs=5.22e-4,             # [L] 5.22 um
            cs_max=0.051555,        # [L] 51555 mol/m^3, [S1] Table VII MEASURED.
                                    #     The calibrated [S2] value is 63104; with
                                    #     that, an 85 um coating would give
                                    #     5.95 mAh/cm^2 and 27.5 Ah, well above
                                    #     every reported 4680 capacity. See the
                                    #     CS_MAX NOTE below.
            theta0=THETA_P0,        # [E] WINDOW CALIBRATION (27 Sep 2026);
                                    #     Chen2020 [L] gave 0.2661.
            theta_max=THETA_P_END,  # [D] end of the C/20 discharge to 2.5 V.
                                    #     Until 27 Sep this was 0.8841, chosen
                                    #     so the WINDOW held 22.078 Ah; it did
                                    #     not stop the discharge, which with the
                                    #     transferred theta0 ran to 26.1 Ah at
                                    #     C/3 (tau_c = 700 s), +18 %.
            Ds=(5.22e-4) ** 2 / TAU_C_POS,  # [E] set through the diffusion
                                    #     time tau_c = r_s^2/D_s (TAU_C_POS),
                                    #     not transferred; 3.89e-14 m^2/s
            m_ref=0.342,            # [L] 3.42e-6 (A/m2)(m3/mol)^1.5 -> CGS x 1e5
            alpha_a=0.5, alpha_c=0.5,   # [L]
            Eact_i0=17.8e3,         # [E] typical NMC charge-transfer value
            Eact_Ds=EACT_DS_POS_CAL,  # [E] calibrated 1 Oct 2026 (was 25 kJ/mol)
            dUdT=DUDT_POS_AVG,      # [L] O'Regan 2022, window average; the
                                    #     solver uses Params.dUdT_pos_fn
        ),
        # ---------------------------------------------------------- Al (half)
        Layer(
            "Al", CC_POS,
            L=0.5 * T_AL_FULL,
            rho=2.70,          # [L] pure aluminium
            cp=0.900,          # [L] pure aluminium
            lam=2.38,          # [L] pure aluminium, 238 W/m/K
            sigma=3.50e5,      # [L] pure aluminium, 3.5e7 S/m
        ),
    ]


# ============================================================================ #
#  SERIES RESISTANCE                                                           #
# ============================================================================ #
def foil_resistance(collection="tabless") -> float:
    """Ohmic path the 2-D domain does NOT resolve, in ohm.

    In a tabless cell current leaves each foil sideways, across the electrode
    height, into an end disc welded along the whole winding length.  For a
    uniformly distributed collection over a strip of height h, width L_w and
    thickness t, the effective resistance is h / (3 sigma t L_w) -- the factor 3
    is the standard result for current that builds up linearly along the path.

    Both foils carry the current of BOTH coated faces, so the full (not halved)
    thickness is used here.
    """
    h = ELECTRODE_HEIGHT
    R_al = h / (3.0 * 3.50e5 * T_AL_FULL * WIND_LENGTH)
    R_cu = h / (3.0 * 5.96e5 * T_CU_FULL * WIND_LENGTH)
    return R_al + R_cu


ROLL_R_CORE = _p("jelly-roll core radius", "M", 0.25, "cm",
                 "[S6] report NO winding mandrel: the core is an empty 5 mm "
                 "void. Previously estimated at 3 mm assuming a former, which "
                 "was 20 % too large. The inner thermal boundary is adiabatic "
                 "either way, but the void carries no heat capacity and the "
                 "spiral map starts further in.")
ROLL_R_OUT = _p("jelly-roll outer radius", "M", 2.225, "cm",
                "[S6] measure the roll at 44.5 mm diameter, inside a 46 mm "
                "can with a 0.5 mm wall.")
ROLL_PERIOD = _p("full winding period", "E", 2.0 * (T_ANODE + T_SEP + T_CATHODE)
                 + T_CU_FULL + T_AL_FULL, "cm",
                 "Derived, not independent: Al + 2 cathode + 2 separator + "
                 "2 anode + Cu. Sets the radius-vs-winding-length map.")
H_CAN = _p("can-surface heat transfer coefficient", "E", 5.0e-3, "W/cm^2/K",
           "50 W/m^2/K on the can outer surface, representative of a 4680 in a "
           "liquid-cooled pack with side-wall contact. No 4680-specific "
           "measurement found. This dominates the predicted temperature rise: "
           "at 1C the can-side convection is ~80 % of the total thermal "
           "resistance, so halving it nearly doubles the steady temperature "
           "rise. Sweep it before trusting any thermal number from this model.")

R_HARDWARE = _p("hardware series resistance", "E", 0.5e-3, "ohm",
                "Welds, end discs, can, terminals and contact resistance. Not "
                "resolved by any 2-D slab model. Chosen so that the total "
                "modelled DCIR lands inside the 5.4-7 mOhm of [S5]; the "
                "electrochemistry supplies the rest. Pure fitting parameter.")
_p("holder contact resistance, heat only", "E", 0.94e-3, "ohm",
   "I^2 R_c at the can node, outside the voltage path, during the charge; "
   "calibrated 1 Oct 2026 on the charge test of Ank et al. (2023). Equals "
   "0.5 mOhm per terminal, the value Ank et al. estimate for their holder. A "
   "property of the holder of that test, not of the cell (R_CONTACT_ANK).")


# ============================================================================ #
#  PARAMETER SET                                                               #
# ============================================================================ #
def params_4680(C_rate=1.0, collection="tabless", **kw) -> Params:
    """Assemble the 4680 Params.  Any field can be overridden by keyword."""
    p = Params(
        layers=layers_4680(),
        # ---- geometry: the unrolled winding
        H=WIND_LENGTH,
        depth=DEPTH,
        # ---- electrolyte, 1.0 M LiPF6 in EC:EMC 3:7
        ce0=1.0e-3,                 # [L] 1000 mol/m^3
        t_plus=0.2594,              # [L] Chen2020 / Landesfeind
        dlnf_dlnce=0.0,             # [L] Chen2020 uses a thermodynamic factor of 1
        kappa_fn=kappa_nyman2008,   # [L] replaces Gu & Wang Eq. 19
        De_fn=De_nyman2008,         # [L] replaces the constant De of Table II
        Eact_De=17.1e3,             # [E] typical carbonate value; [S3] is 25 C only
        Eact_kap=17.1e3,            # [E] as above
        # ---- chemistry
        ocp_neg=ocp_graphite_chen2020,
        ocp_pos=ocp_nmc811_chen2020,
        # ---- operating point
        C_rate=C_rate,
        T0=298.15, Ta=298.15,
        V_cut=2.5,                  # [L] standard NMC811/graphite lower cutoff
        # ---- current collection
        collection=collection,
        R_series=foil_resistance(collection) + R_HARDWARE,
        # ---- thermal topology.  y is the unrolled winding, NOT a Cartesian
        #      coordinate, so conduction along y is replaced by cylindrical
        #      conduction between successive wraps and the outermost wrap is
        #      coupled to ambient through the can.  See gw2d.Params docstring.
        thermal_topology="wound_radial",
        roll_r_core=ROLL_R_CORE,
        roll_r_out=ROLL_R_OUT,
        roll_height=ROLL_HEIGHT,   # mechanical height: what conducts heat
        roll_period=ROLL_PERIOD,
        h_can=H_CAN,
        h_tab=0.0,                  # no tab in the tabless configuration
        # ---- heat source (paper 1, 27 Sep 2026): the reaction heat with the
        #      OCP at the particle mean, which adds the heat of mixing the
        #      printed Gu & Wang form omits, and the entropic coefficients of
        #      O'Regan et al. (2022) as functions of stoichiometry.
        heat_ocp="mean",
        dUdT_neg_fn=dUdT_graphite_oregan2022,
        dUdT_pos_fn=dUdT_nmc811_oregan2022,
    )
    return replace(p, **kw) if kw else p


def params_4680_two_node(C_rate=1.0, **kw) -> Params:
    """Paper 1's validation model for the NMC cell: the DFN with a two-node
    thermal balance (roll + surface), convecting through the whole can
    exterior. Use with ny=1 -- with one roll temperature every row is the same."""
    import parts4680 as _PT
    r = _PT.CAN_OD / 2.0
    A_ext = 2.0 * np.pi * r * _PT.CAN_H + 2.0 * np.pi * r ** 2      # cm^2
    return params_4680(C_rate=C_rate, two_node=True, two_node_A_ext=A_ext, **kw)


# Whole-cell heat capacity for the two-node model of Ank et al.'s fixture.
# m = 355.9 g [M] (Ank et al. 2023), c_p = 884.6 J/kg/K [L] (BPX transfer; its
# +-30 % band is the main thermal uncertainty). hA of the fixture [M]:
# m c_p / tau with tau = 1735 s from the measured cooling tails.
CELL_MASS_G = 355.9
CELL_CP = 0.8846            # J/g/K
HA_ANK_FIXTURE = 0.18       # W/K


def params_4680_ank(C_rate=1.0, hA=HA_ANK_FIXTURE, P_can_ext=0.0, nx=None,
                    **kw) -> Params:
    """Paper 1's model of Ank et al.'s thermal test (author, 27 Sep 2026):
    DFN + two-node, the roll node carrying its own layers' heat capacity and
    the can node the rest of m c_p (can, end discs, tabs, free electrolyte),
    hA on the can node, and the I^2 R_series heat in the balance with the foil
    share in the roll and the hardware share in the can node."""
    p = params_4680_two_node(C_rate=C_rate, **kw)
    m = GuWang2D(p, nx_per_layer=nx or (1, 20, 8, 20, 1), ny=1)
    C_roll = float((m.rho_cp * m.m.V).sum())
    return replace(p, two_node_C_can=CELL_MASS_G * CELL_CP - C_roll,
                   h_can=hA / p.two_node_A_ext, series_heat=True,
                   series_R_roll=foil_resistance(p.collection),
                   P_can_ext=P_can_ext)


def cell_4680(C_rate=1.0, collection="tabless",
              nx_per_layer=(1, 14, 6, 12, 1), ny=16, **kw) -> GuWang2D:
    return GuWang2D(params_4680(C_rate=C_rate, collection=collection, **kw),
                    nx_per_layer=nx_per_layer, ny=ny)


# ============================================================================ #
#  COMPLETE PARAMETER REGISTRY                                                 #
# ============================================================================ #
# The module-level PROV registry above covers the format-level geometry and the
# thermal boundary condition, because those are the numbers defined as module
# constants.  The bulk of the parameter set lives inside the Layer objects, and
# until now its provenance existed only as inline comments -- readable in the
# source, invisible in the released metadata and invisible in the paper.
#
# LAYER_PROV closes that gap.  It carries only the SYMBOL, TAG, UNIT and SOURCE
# NOTE for each layer field; the VALUE is always read back from the live Layer
# object by `parameter_rows()`, never duplicated here.  A registry that stored
# its own copy of the value would drift from the model the first time someone
# edited one and not the other, which is exactly the failure this whole package
# is built to make impossible.
#
# `audit_registry()` asserts the converse -- that no meaningful field of any
# layer is missing from the registry -- so adding a parameter without
# documenting its provenance fails loudly instead of quietly shipping.
#
# Key: (layer name, field name) -> (symbol, tag, unit, note)
LAYER_PROV: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {
    # ------------------------------------------------------------------ Cu --
    ("Cu", "L"): ("$L_{\\mathrm{Cu}}/2$", "M", "um",
                  "Half of the 8 um foil [S5]; the model resolves one "
                  "half-stack about the foil centreline."),
    ("Cu", "rho"): ("$\\rho$", "L", "g/cm^3", "Pure copper."),
    ("Cu", "cp"): ("$c_p$", "L", "J/g/K", "Pure copper."),
    ("Cu", "lam"): ("$\\lambda$", "L", "W/cm/K", "Pure copper, 400 W/m/K."),
    ("Cu", "sigma"): ("$\\sigma$", "L", "S/cm", "Pure copper, 5.96e7 S/m."),
    # --------------------------------------------------------------- anode --
    ("anode", "L"): ("$L_-$", "M", "um",
                     "[S6] loading 19.5 mg/cm^2 per side at 1.55 g/cm^3 gives "
                     "126 um, consistent with the measured double-sided "
                     "thickness. Was an [E] 110 um until 27 Sep 2026."),
    ("anode", "rho"): ("$\\rho$", "E", "g/cm^3",
                       "Graphite composite including binder and imbibed "
                       "electrolyte. No 4680-specific source."),
    ("anode", "cp"): ("$c_p$", "E", "J/g/K", "Typical graphite electrode."),
    ("anode", "lam"): ("$\\lambda$", "E", "W/cm/K",
                       "1.5 W/m/K through-plane composite."),
    ("anode", "sigma"): ("$\\sigma$", "L", "S/cm", "[S1] 215 S/m."),
    ("anode", "eps_e"): ("$\\varepsilon_e$", "D", "-",
                         "1 - eps_s, as in [S1], which counts all anode "
                         "solid as active. [S1] gave 0.25."),
    ("anode", "eps_s_explicit"): ("$\\varepsilon_s$", "D", "-",
                                  "[S6] loading 19.5 mg/cm^2 over the 126 um "
                                  "coating at the composite solid density of "
                                  "2.20 g/cm^3. [S1] gave 0.75."),
    ("anode", "rs"): ("$r_s$", "L", "um", "[S1] 5.86 um."),
    ("anode", "cs_max"): ("$c_{s,\\max}$", "L", "mol/cm^3",
                          "[S1] Table VII, 30555 mol/m^3, MEASURED variant. "
                          "See the note on the two cs_max variants in the provenance section."),
    ("anode", "theta0"): ("$\\theta_0$", "E", "-",
                          "Window calibration (27 Sep 2026): solved with the other electrode's value so that OCV(100 %) = 4.2 V and a C/20 discharge to 2.5 V delivers 22.31 Ah (Ank et al. 2023), with the Chen2020 OCP functions and the loadings kept, giving 0.701; left free in the thermal calibration of 1 Oct 2026 (paper 1 Section 5.1), which returned 0.700. Chen2020 [S1] gave 0.9014."),
    ("anode", "theta_min"): ("$\\theta_{\\min}$", "D", "-",
                             "End of the C/20 discharge to 2.5 V after the "
                             "window calibration."),
    ("anode", "Ds"): ("$D_s$", "L", "cm^2/s", "[S2] 3.3e-14 m^2/s."),
    ("anode", "m_ref"): ("$m_{\\mathrm{ref}}$", "L",
                         "(A/cm^2)(cm^3/mol)^1.5",
                         "[S2] 6.48e-7 in SI units."),
    ("anode", "alpha_a"): ("$\\alpha_a$", "L", "-", "Symmetric kinetics."),
    ("anode", "alpha_c"): ("$\\alpha_c$", "L", "-", "Symmetric kinetics."),
    ("anode", "Eact_i0"): ("$E_{a,i_0}$", "E", "J/mol",
                           "Typical graphite charge-transfer activation "
                           "energy; [S1] is isothermal and reports none."),
    ("anode", "Eact_Ds"): ("$E_{a,D_s}$", "E", "J/mol",
                           "Typical graphite solid-diffusion value."),
    ("anode", "dUdT"): ("$\\partial U/\\partial T$", "L", "V/K",
                        "O'Regan et al. (2022), measured on the LG M50 "
                        "electrodes of [S1], as a function of stoichiometry. "
                        "The value shown is its average over the working "
                        "window."),
    # ----------------------------------------------------------- separator --
    ("separator", "L"): ("$L_{\\mathrm{sep}}$", "L", "um",
                         "[S1] LG M50 value. [S4] notes the 4680 separator is "
                         "ceramic-coated on the cathode side, not captured."),
    ("separator", "rho"): ("$\\rho$", "E", "g/cm^3",
                           "Polyethylene wetted with carbonate."),
    ("separator", "cp"): ("$c_p$", "E", "J/g/K", "Wetted polyethylene."),
    ("separator", "lam"): ("$\\lambda$", "E", "W/cm/K",
                           "0.33 W/m/K. The dominant through-stack thermal "
                           "resistance of the layer stack."),
    ("separator", "eps_e"): ("$\\varepsilon_e$", "L", "-", "[S1] porosity."),
    ("separator", "eps_p"): ("$\\varepsilon_p$", "L", "-",
                             "Remainder is polymer."),
    # ------------------------------------------------------------- cathode --
    ("cathode", "L"): ("$L_+$", "M", "um",
                       "[S5] ~85 um, taken as measured rather than fitted. "
                       "The delivered capacity is NOT thereby a prediction: "
                       "the accessible stoichiometric window was calibrated "
                       "to the measured capacity instead. See the "
                       "UTILISATION NOTE."),
    ("cathode", "rho"): ("$\\rho$", "E", "g/cm^3",
                         "NMC811 composite including binder and electrolyte."),
    ("cathode", "cp"): ("$c_p$", "E", "J/g/K", "Typical NMC electrode."),
    ("cathode", "lam"): ("$\\lambda$", "E", "W/cm/K",
                         "1.5 W/m/K through-plane composite."),
    ("cathode", "sigma"): ("$\\sigma$", "L", "S/cm", "[S1] 0.18 S/m."),
    ("cathode", "eps_e"): ("$\\varepsilon_e$", "L", "-",
                           "[S1] porosity; [S5] reports 30--40 %, consistent."),
    ("cathode", "eps_s_explicit"): ("$\\varepsilon_s$", "L", "-",
                                    "[S1] active-material volume fraction."),
    ("cathode", "rs"): ("$r_s$", "L", "um", "[S1] 5.22 um."),
    ("cathode", "cs_max"): ("$c_{s,\\max}$", "L", "mol/cm^3",
                            "[S1] Table VII, 51555 mol/m^3, MEASURED variant. "
                            "The calibrated [S2] value of 63104 would give "
                            "27.5 Ah, above every reported 4680 capacity."),
    ("cathode", "theta0"): ("$\\theta_0$", "E", "-",
                            "Window calibration (27 Sep 2026): solved with the other electrode's value so that OCV(100 %) = 4.2 V and a C/20 discharge to 2.5 V delivers 22.31 Ah (Ank et al. 2023), with the Chen2020 OCP functions and the loadings kept. Chen2020 [S1] gave 0.2661."),
    ("cathode", "theta_max"): ("$\\theta_{\\max}$", "D", "-",
                               "End of the C/20 discharge to 2.5 V after the "
                               "window calibration. It replaces 0.8841, which "
                               "set the window to 22.078 Ah but did not stop "
                               "the discharge: with the transferred theta0 the "
                               "model delivered 26.1 Ah at C/3, +18 %. Check "
                               "not used in the calibration: C/3 now gives "
                               "22.083 Ah (+0.02 %)."),
    ("cathode", "Ds"): ("$D_s$", "E", "cm^2/s",
                        "Set through the diffusion time tau_c = r_s^2/D_s "
                        "with r_s [L]. Calibrated 1 Oct 2026 on the 1C/2C "
                        "charge test of Ank et al. (2023) (paper 1 Section "
                        "5.1): tau_c = 8031 s (3.39e-15 m^2/s), within x1.2 "
                        "of the transferred 6812 s of [S2] (4.0e-15 m^2/s). "
                        "The transferred set of the paper used 700 s, chosen "
                        "below the resistance-based heat of About:Energy."),
    ("cathode", "m_ref"): ("$m_{\\mathrm{ref}}$", "L",
                           "(A/cm^2)(cm^3/mol)^1.5",
                           "[S2] 3.42e-6 in SI units."),
    ("cathode", "alpha_a"): ("$\\alpha_a$", "L", "-", "Symmetric kinetics."),
    ("cathode", "alpha_c"): ("$\\alpha_c$", "L", "-", "Symmetric kinetics."),
    ("cathode", "Eact_i0"): ("$E_{a,i_0}$", "E", "J/mol",
                             "Typical NMC charge-transfer value."),
    ("cathode", "Eact_Ds"): ("$E_{a,D_s}$", "E", "J/mol",
                             "Calibrated 1 Oct 2026 with tau_c on the charge "
                             "test of Ank et al. (2023): 11.0 kJ/mol "
                             "(transferred set: 25 kJ/mol, a typical NMC value)."),
    ("cathode", "dUdT"): ("$\\partial U/\\partial T$", "L", "V/K",
                          "O'Regan et al. (2022), LG M50 NMC811, as a "
                          "function of stoichiometry; window average shown."),
    # ------------------------------------------------------------------ Al --
    ("Al", "L"): ("$L_{\\mathrm{Al}}/2$", "M", "um",
                  "Half of the 10 um foil [S5]."),
    ("Al", "rho"): ("$\\rho$", "L", "g/cm^3", "Pure aluminium."),
    ("Al", "cp"): ("$c_p$", "L", "J/g/K", "Pure aluminium."),
    ("Al", "lam"): ("$\\lambda$", "L", "W/cm/K",
                    "Pure aluminium, 238 W/m/K."),
    ("Al", "sigma"): ("$\\sigma$", "L", "S/cm", "Pure aluminium, 3.5e7 S/m."),
}

# Params-level fields that are not module constants and so are not in PROV.
# Same contract: value read live, everything else declared here.
PARAM_PROV: Dict[str, Tuple[str, str, str, str]] = {
    "ce0": ("$c_{e,0}$", "L", "mol/cm^3",
            "1.0 M LiPF6 in EC:EMC 3:7, the electrolyte of [S1] and [S3]."),
    "t_plus": ("$t_+^0$", "L", "-", "[S1] transference number."),
    "dlnf_dlnce": ("$\\mathrm{d}\\ln f/\\mathrm{d}\\ln c_e$", "L", "-",
                   "[S1] uses a thermodynamic factor of unity, so this is "
                   "zero."),
    "Eact_De": ("$E_{a,D_e}$", "E", "J/mol",
                "Typical carbonate value; [S3] is reported at 25 C only and "
                "gives no activation energy."),
    "Eact_kap": ("$E_{a,\\kappa}$", "E", "J/mol", "As for $E_{a,D_e}$."),
    "T0": ("$T_0$", "M", "K", "Uniform initial temperature, 25 C."),
    "Ta": ("$T_a$", "M", "K", "Ambient, 25 C."),
    "V_cut": ("$V_{\\mathrm{cut}}$", "L", "V",
              "Standard NMC811/graphite lower cut-off."),
}

# Correlations, which are functions rather than numbers.
# --------------------------------------------------------------------------- #
#  CORRELATIONS: coefficients, reference values, and a check that the formula
#  printed in the paper is the formula the solver evaluates
# --------------------------------------------------------------------------- #
# The four constitutive relations are functions, not numbers, so a parameter
# table can only give a reference value; the closed forms belong in the text.
# But an equation transcribed into LaTeX by hand is exactly the kind of thing
# that silently stops matching the code.
#
# So the coefficients are declared once, here. `rebuild_*` reconstructs each
# correlation from them, `verify_correlations()` asserts the reconstruction
# reproduces the module's own function to round-off across the whole physical
# range, and build_tex.py typesets the equations from the same declarations.
# The functions above are left untouched -- they are the verified physics, and
# this machinery checks them rather than replacing them.
OCP_POS_COEF = dict(lin=-0.8090, const=4.4875,
                    tanh=((-0.0428, 18.5138, 0.5542),
                          (-17.7326, 15.7890, 0.3117),
                          (+17.5842, 15.9308, 0.3120)))
OCP_NEG_COEF = dict(exp=(1.9793, -39.3631), const=0.2482,
                    tanh=((-0.0909, 29.8538, 0.1234),
                          (-0.04478, 14.9159, 0.2769),
                          (-0.0205, 30.4444, 0.6103)))
# Nyman 2008 as published, in SI, with c = ce[mol/m^3]/1000
KAPPA_COEF = (0.1297, -2.5100, 3.3290)        # c^3, c^1.5, c  -> S/m
DE_COEF = (8.794e-11, -3.972e-10, 4.862e-10)  # c^2, c, 1      -> m^2/s


def rebuild_ocp_pos(theta):
    s = np.clip(theta, 1.0e-6, 1.0 - 1.0e-6)
    c = OCP_POS_COEF
    out = c["lin"] * s + c["const"]
    for a, b, d in c["tanh"]:
        out = out + a * np.tanh(b * (s - d))
    return out


def rebuild_ocp_neg(theta):
    s = np.clip(theta, 1.0e-6, 1.0 - 1.0e-6)
    c = OCP_NEG_COEF
    a0, k0 = c["exp"]
    out = a0 * np.exp(np.clip(k0 * s, -60.0, 60.0)) + c["const"]
    for a, b, d in c["tanh"]:
        out = out + a * np.tanh(b * (s - d))
    return out


def rebuild_kappa(ce):
    c = np.clip(ce, 1.0e-9, 5.0e-3) * 1.0e3
    a3, a15, a1 = KAPPA_COEF
    return (a3 * c ** 3 + a15 * c ** 1.5 + a1 * c) / 100.0


def rebuild_De(ce):
    c = np.clip(ce, 1.0e-9, 5.0e-3) * 1.0e3
    a2, a1, a0 = DE_COEF
    return (a2 * c ** 2 + a1 * c + a0) * 1.0e4


def verify_correlations() -> Dict[str, float]:
    """Max relative difference between each declared form and the function the
    solver actually calls. Should be at round-off; anything larger means the
    equation in the paper has drifted from the code."""
    th = np.linspace(0.01, 0.99, 2001)
    ce = np.linspace(1.0e-5, 4.0e-3, 2001)

    def rel(a, b):
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))

    return {
        "U_pos": rel(rebuild_ocp_pos(th), ocp_nmc811_chen2020(th)),
        "U_neg": rel(rebuild_ocp_neg(th), ocp_graphite_chen2020(th)),
        "kappa": rel(rebuild_kappa(ce), kappa_nyman2008(ce)),
        "De": rel(rebuild_De(ce), De_nyman2008(ce)),
    }


# Reference states at which each correlation is quoted in the parameter table.
# A function has no single value, but a reader still needs a magnitude, and a
# quoted number is only meaningful if the state it was evaluated at is given.
def correlation_rows() -> List[Dict[str, object]]:
    p_pos = next(l for l in layers_4680() if l.name == "cathode")
    p_neg = next(l for l in layers_4680() if l.name == "anode")
    ce0 = 1.0e-3
    return [
        dict(symbol="$U_+(\\theta)$", tag="L", unit="V",
             value=float(ocp_nmc811_chen2020(p_pos.theta0)),
             ref=f"at $\\theta = {p_pos.theta0:.4f}$, i.e. 100 \\% SOC",
             note="NMC811 open-circuit potential, `nmc_LGM50_ocp_Chen2020` "
                  "[S1, S2], reproduced verbatim. Full form in Eq. (\\ref{eq:ocppos})."),
        dict(symbol="$U_-(\\theta)$", tag="L", unit="V",
             value=float(ocp_graphite_chen2020(p_neg.theta0)),
             ref=f"at $\\theta = {p_neg.theta0:.4f}$, i.e. 100 \\% SOC",
             note="Graphite open-circuit potential, "
                  "`graphite_LGM50_ocp_Chen2020` [S1, S2]. The LG M50 anode "
                  "carries a few per cent SiOx whereas [S4] reports the 4680 "
                  "anode as silicon-free; the fit is transferred with that "
                  "caveat. Full form in Eq. (\\ref{eq:ocpneg})."),
        dict(symbol="$\\kappa(c_e)$", tag="L", unit="S/cm",
             value=float(kappa_nyman2008(ce0)),
             ref="at $c_e = 1.0$ M",
             note="Electrolyte ionic conductivity [S3]. About nine times the "
                  "plasticised-gel value of Gu & Wang, which is why the "
                  "electrolyte ohmic loss stops dominating the polarisation "
                  "budget. Full form in Eq. (\\ref{eq:kappa})."),
        dict(symbol="$D_e(c_e)$", tag="L", unit="cm^2/s",
             value=float(De_nyman2008(ce0)),
             ref="at $c_e = 1.0$ M",
             note="Salt diffusivity [S3], about 2.4 times the Gu & Wang gel "
                  "value. Full form in Eq. (\\ref{eq:de})."),
    ]

# Fields deliberately left at their dataclass defaults, and why. Listing them
# is part of the audit: a reader should be able to tell "not used" from
# "forgotten".
UNUSED_FIELDS: Dict[str, str] = {
    "eps_f": "no separate filler phase; binder and carbon are lumped into the "
             "active fraction, as [S1] reports them",
    "i0": "superseded by m_ref, the form modern parameter sets report",
    "Eact_sig": "solid conductivity held temperature-independent",
    "theta_max": "full utilisation assumed where not set explicitly",
    "theta_min": "full utilisation assumed where not set explicitly",
    "eps_p": "only meaningful for the separator",
}


def parameter_rows(p: Params = None) -> List[Dict[str, object]]:
    """Every parameter, with its live value and its declared provenance.

    Values come from the Params/Layer objects the solver actually runs with, so
    this cannot disagree with the model. Units are converted to the reporting
    unit declared in the registry (cm -> um where that is what a reader wants).
    """
    p = p or params_4680()
    rows = []

    for L in p.layers:
        for field, (sym, tag, unit, note) in LAYER_PROV.items():
            if field[0] != L.name:
                continue
            val = getattr(L, field[1])
            if unit == "um":                      # the registry reports um
                val = val * 1e4
            rows.append(dict(group=L.name, symbol=sym, field=field[1],
                             value=val, unit=unit, tag=tag, note=note))

    for name, (sym, tag, unit, note) in PARAM_PROV.items():
        rows.append(dict(group="cell", symbol=sym, field=name,
                         value=getattr(p, name), unit=unit, tag=tag, note=note))

    for name, (tag, val, unit, note) in PROV.items():
        rows.append(dict(group="format", symbol="", field=name,
                         value=val, unit=unit, tag=tag, note=note))

    for c in correlation_rows():
        rows.append(dict(group="correlation", symbol=c["symbol"],
                         field=c["symbol"], value=c["value"], unit=c["unit"],
                         tag=c["tag"], note=c["note"], ref=c["ref"]))
    return rows


def audit_registry(p: Params = None) -> List[str]:
    """Assert the registry covers every meaningful field of every layer.

    Returns the list of undocumented fields, which should be empty. This is the
    check that keeps the provenance record honest as the model changes: a new
    parameter with no declared source shows up here rather than shipping
    silently.
    """
    p = p or params_4680()
    missing = []
    for L in p.layers:
        electrode = L.kind in (ANODE, CATHODE)
        for f in L.__dataclass_fields__:
            if f in ("name", "kind") or f in UNUSED_FIELDS:
                continue
            # the kinetic symmetry factors carry dataclass defaults on every
            # layer but are meaningless anywhere but an electrode
            if not electrode and f in ("alpha_a", "alpha_c"):
                continue
            val = getattr(L, f)
            if isinstance(val, float) and val == 0.0:
                continue                          # genuinely not used here
            if (L.name, f) not in LAYER_PROV:
                missing.append(f"{L.name}.{f} = {val!r}")
    return missing


def tag_counts(p: Params = None) -> Dict[str, int]:
    """How many parameters carry each tag. Quoted in the paper, so that the
    weight of the [E] category is stated rather than left to be counted."""
    out: Dict[str, int] = {}
    for r in parameter_rows(p):
        t = str(r["tag"]).split("/")[0]
        out[t] = out.get(t, 0) + 1
    return out


# ============================================================================ #
#  REPORTING                                                                   #
# ============================================================================ #
def _winding_period(p: Params) -> float:
    """Full winding period of ONE turn, cm, from the Params' own layers. [D]

    One turn contains one Cu foil, one Al foil and two of each coating, so the
    foils count once at full thickness -- and the layer objects hold HALF of
    each foil, about its centreline, hence the factor 2.

    Computed from p rather than from this module's constants because
    derived_metrics is called on BOTH cells' Params. It used to read T_ANODE,
    T_SEP, T_CATHODE, T_CU_FULL and T_AL_FULL directly, which are the NMC
    cell's, so an LFP Params silently got the NMC period back -- 432 um where
    its own geometry gives 297.
    """
    stack = sum(L.L for L in p.layers if L.name in ("anode", "separator",
                                                    "cathode"))
    foils = sum(L.L for L in p.layers if L.name in ("Cu", "Al"))
    return 2.0 * stack + 2.0 * foils


def _foil_R(p: Params) -> float:
    """Foil collection resistance, ohm, from the Params' own geometry. [D]

    Same formula as foil_resistance() -- h/(3 sigma t L_w) per foil -- and the
    same reason for existing as _winding_period: foil_resistance() reads this
    module's NMC constants and so returned the NMC value for the LFP cell.
    Verified against foil_resistance("tabless") for the NMC Params.
    """
    h = p.depth / 2.0                    # model depth counts both coated faces
    R = 0.0
    for name in ("Cu", "Al"):
        L = p.layer_by_name(name) if hasattr(p, "layer_by_name") else next(
            x for x in p.layers if x.name == name)
        R += h / (3.0 * L.sigma * (2.0 * L.L) * p.H)
    return R


def derived_metrics(p: Params) -> Dict[str, float]:
    a, c = p.layer(ANODE), p.layer(CATHODE)
    return {
        "half-stack thickness [um]": p.L_cell * 1e4,
        "full winding period [um]": _winding_period(p) * 1e4,
        "active separator area [cm^2]": p.separator_area,
        "anode areal capacity [mAh/cm^2]": p.electrode_capacity(ANODE) / 3.6,
        "cathode areal capacity [mAh/cm^2]": p.electrode_capacity(CATHODE) / 3.6,
        "cell areal capacity [mAh/cm^2]": p.areal_capacity() / 3.6,
        "N/P ratio [-]": p.np_ratio,
        "cell capacity [Ah]": p.capacity_Ah,
        "1C current [A]": p.capacity_Ah,
        "OCV at 100% SOC [V]": float(p.U_pos(c.theta0) - p.U_neg(a.theta0)),
        "a_s anode [1/cm]": a.a_s,
        "a_s cathode [1/cm]": c.a_s,
        "kappa at ce0 [S/cm]": float(kappa_nyman2008(p.ce0)),
        "De at ce0 [cm^2/s]": float(De_nyman2008(p.ce0)),
        "foil R, tabless [mohm]": _foil_R(p) * 1e3,
        "R_series total [mohm]": p.R_series * 1e3,
        "lambda_radial [W/m/K]": _lam_radial(p) * 100.0,
        "lambda_winding [W/m/K]": _lam_parallel(p) * 100.0,
        "jelly-roll mass estimate [g]": _roll_mass(p),
        "thermal mass [J/K]": _thermal_mass(p),
    }


def _lam_radial(p: Params) -> float:
    """Series (through-stack) effective conductivity, W/cm/K.  This is what
    governs radial heat flow out of the jelly roll."""
    return p.L_cell / sum(L.L / L.lam for L in p.layers)


def _lam_parallel(p: Params) -> float:
    """Parallel (in-plane) effective conductivity, W/cm/K -- dominated by the
    metal foils.  Reported for context; the wound_radial topology does not use
    it, because in a wound cell this direction runs along the winding, not
    towards any cooled surface."""
    return sum(L.L * L.lam for L in p.layers) / p.L_cell


def _roll_mass(p: Params) -> float:
    return sum(L.L * L.rho for L in p.layers) * p.separator_area


def _thermal_mass(p: Params) -> float:
    return sum(L.L * L.rho_cp for L in p.layers) * p.separator_area


def jelly_roll_consistency() -> Dict[str, float]:
    """Does a stack of this period, wound into Ank's measured roll, produce
    Ank's measured electrode length?

    An independent audit: none of the coating thicknesses were chosen to
    satisfy it, and Ank et al. measure BOTH ends of it -- the roll (44.5 mm
    outer, 5 mm void, explicitly no mandrel) and the electrode (cathode 3267 mm,
    from "the total length of the anode is 3403 mm, which is 136 mm longer than
    the cathode").

    An Archimedean spiral of constant pitch p from r_core to r_out has length
    L = pi (r_out^2 - r_core^2) / p exactly, which is better than counting turns
    times a mean circumference and costs nothing.

    THE RESIDUAL, AND WHY IT IS NOT A PARAMETER ERROR
    -------------------------------------------------
    The nominal 432 um period gives 355.5 cm against a measured 326.7 cm: 8.8 %
    long, equivalently a mean period of 470 um rather than 432. That looks like
    a missing 19 um per half-stack until Ank's Fig. 9 is read, where they
    measure thickness ALONG the electrode and find it strongly non-uniform --
    highest at the core, falling, then rising again beyond 1 m -- while "the
    electrode loadings weighed from the electrode samples remain almost constant
    throughout their length". They attribute the non-uniformity to differential
    compression in the winding and to volume changes in a cell that had already
    been cycled, with no mandrel to restrain the core.

    So the as-wound period is not the nominal one, and a mean above nominal is
    exactly what their measurement shows. The residual is a property of the
    wound state, not an error in a coating thickness.

    It also does not propagate into capacity, which is built from the MEASURED
    loading and the MEASURED area, never from thickness times a spiral. That
    independence is the reason this check is worth publishing rather than
    quietly satisfying.
    """
    d_out, d_core = 4.45, 0.50      # cm [M] Ank et al., no mandrel
    L_meas = WIND_LENGTH            # cm [M] cathode, 3267 mm
    period = 2.0 * (T_ANODE + T_SEP + T_CATHODE) + T_CU_FULL + T_AL_FULL
    r_o, r_c = d_out / 2.0, d_core / 2.0
    length = np.pi * (r_o ** 2 - r_c ** 2) / period
    period_implied = np.pi * (r_o ** 2 - r_c ** 2) / L_meas
    return {"winding period, nominal [um]": period * 1e4,
            "winding period implied by the measured length [um]":
                period_implied * 1e4,
            "turns, nominal [-]": (r_o - r_c) / period,
            "implied winding length [cm]": length,
            "measured winding length [cm]": L_meas,
            "discrepancy [%]": 100.0 * (length - L_meas) / L_meas}

def provenance_report() -> str:
    """Full audit trail, grouped by tag."""
    p = params_4680()
    out = ["=" * 78,
           "4680 PARAMETER PROVENANCE",
           "=" * 78,
           "  [M] measured/reported for the 4680   [L] literature, same chemistry",
           "  [E] EXTRAPOLATED BY THIS MODEL -- treat as a fitting parameter",
           ""]

    # module-level registry
    for tag in ("M", "M/E", "L", "E"):
        rows = [(k, v) for k, v in PROV.items() if v[0] == tag]
        if not rows:
            continue
        out.append(f"--- tag [{tag}] " + "-" * (62 - len(tag)))
        for k, (_, val, unit, note) in rows:
            out.append(f"  {k:38s} {val:>12.6g} {unit}")
            for line in _wrap(note, 70):
                out.append(f"      {line}")
        out.append("")

    # per-layer table with inline tags
    out.append("--- layer stack (half-symmetric; foils are halved) " + "-" * 26)
    out.append(f"  {'layer':10s} {'L [um]':>9s} {'eps_e':>7s} {'eps_s':>7s} "
               f"{'rs [um]':>9s} {'sigma':>10s} {'lam':>8s}")
    for L in p.layers:
        out.append(f"  {L.name:10s} {L.L*1e4:9.2f} {L.eps_e:7.3f} {L.eps_s:7.3f} "
                   f"{L.rs*1e4:9.2f} {L.sigma:10.4g} {L.lam:8.4g}")
    out.append("")

    out.append("--- derived cell metrics " + "-" * 51)
    for k, v in derived_metrics(p).items():
        out.append(f"  {k:38s} {v:>14.6g}")
    out.append("")

    out.append("--- jelly-roll geometric consistency check " + "-" * 34)
    for k, v in jelly_roll_consistency().items():
        out.append(f"  {k:38s} {v:>14.6g}")
    out.append("  A stack of the chosen thickness wound into a 46 mm can should")
    out.append("  reproduce the measured 3.3 m winding. Agreement to within a few")
    out.append("  tens of percent is the most that can be expected, since the roll")
    out.append("  OD and mandrel diameter are themselves [E].")
    out.append("")

    out.append("--- registry completeness " + "-" * 51)
    miss = audit_registry(p)
    counts = tag_counts(p)
    total = sum(counts.values())
    out.append(f"  {total} parameters registered: "
               + ", ".join(f"{n} [{t}]" for t, n in sorted(counts.items())))
    if miss:
        out.append("  UNDOCUMENTED FIELDS -- provenance record is incomplete:")
        for m in miss:
            out.append(f"    {m}")
    else:
        out.append("  every meaningful layer field carries a declared source.")
    out.append("")

    out.append("--- MODEL LIMITS ".ljust(78, "-"))
    for line in _MODEL_LIMITS.strip().splitlines():
        out.append("  " + line)
    return "\n".join(out)


_MODEL_LIMITS = """
Things this parameterisation does NOT capture, in rough order of importance:

1. Reversible (entropic) heat uses O'Regan et al. (2022)'s entropic
   coefficients, measured on the LG M50 electrodes the rest of the set
   transfers, not on a 4680. They are functions of stoichiometry and change
   sign with SOC.
2. The out-of-plane (electrode height) direction is not resolved, so the model
   cannot show the axial temperature gradient between the two end discs.
3. Solid diffusion uses a two-parameter closure (parabolic, Gu & Wang Eq.
   14/15, or the quartic profile of Subramanian et al. 2005), not a resolved
   particle. The positive electrode's diffusion time tau_c = r_s^2/D_s is [E]:
   only an upper bound (about 1400 s) is set by the measured heat and DC
   resistance, and 700 s is a working value inside it.
4. Chen2020's Ds and m_ref are effective values fitted to an LG M50 at moderate
   rate; they carry no temperature dependence in the source, so every
   activation energy here is [E].
5. Silicon. [S4] reports the 4680 anode as graphite with no silicon while the
   Chen2020 anode is graphite-SiOx, so the transferred OCP is slightly wrong at
   the lithiated end.
6. Degradation, SEI growth and plating are absent, so N/P margin cannot be
   checked against plating onset -- only stated as a design number.
"""


def compare_tab_vs_tabless(C_rate=1.0, nx_per_layer=(1, 12, 5, 10, 1), ny=10,
                           dod_max=35.0, verbose=False) -> Dict[str, dict]:
    """Run the identical cell with the two collection schemes.

    Only the electrical boundary condition changes.  R_series is held at the
    same value in both cases so that the comparison isolates exactly one thing:
    the ohmic penalty of forcing the whole cell current down 3.3 m of foil to
    reach a tab, which is what Gu & Wang's 2-D analysis is about and what the
    tabless design exists to remove.

    Analytic expectation for the tabbed case, current entering at one end of a
    strip of length L_w, height h, thickness t:  R = L_w / (3 sigma t h).
    """
    out = {}
    for mode in ("tabless", "tab"):
        m = cell_4680(C_rate=C_rate, collection=mode,
                      nx_per_layer=nx_per_layer, ny=ny)
        # Only the first part of the discharge is needed: the tabbed variant's
        # penalty is an ohmic offset that is fully developed within a few
        # percent DOD, and running it to the cutoff is expensive -- its 0.5 V
        # foil drop makes every Newton step fight the potential step limiter.
        h = m.run(dt0=0.05, dt_max=30.0, dod_max=dod_max, verbose=verbose)
        out[mode] = {"model": m, "history": h}

    R_al = WIND_LENGTH / (3.0 * 3.50e5 * T_AL_FULL * ELECTRODE_HEIGHT)
    R_cu = WIND_LENGTH / (3.0 * 5.96e5 * T_CU_FULL * ELECTRODE_HEIGHT)
    out["analytic_tab_foil_resistance_mohm"] = (R_al + R_cu) * 1e3
    out["analytic_tabless_foil_resistance_mohm"] = foil_resistance() * 1e3
    return out


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


# ============================================================================ #
if __name__ == "__main__":
    print(provenance_report())

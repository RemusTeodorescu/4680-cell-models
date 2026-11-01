"""BYD FC4680 LFP/graphite parameter set for the coupled DFN+T model.

    python byd4680_params.py            # provenance report and derived metrics

WHAT THIS IS
------------
The electrochemical half of the LFP cell's parameter set, assembled to the same
standard as gw4680's NMC set: every value carries a provenance tag, every tag
carries a reason, and the tables in the paper are generated from this registry
rather than transcribed.

byd_parts.py and byd_stack.py already carry the geometry, the phase inventory
and the thermal network. This module adds what those two do not: the
electrochemistry, the transport, and the assembly into a Params object the
gw2d solver can run.

WHERE THE ELECTROCHEMISTRY COMES FROM, AND WHY THAT IS DEFENSIBLE
----------------------------------------------------------------
From About:Energy's open BPX parameterisation of an LFP/graphite 2 Ah 18650
[S2], released CC-BY-SA 4.0 and built from cell teardown plus electrochemical
measurement on harvested electrodes.

This is the same operation the NMC set performs. That set transfers its
electrochemistry from an LG M50 21700 [S1] on the assumption that an
NMC811/graphite electrode behaves like another NMC811/graphite electrode, and
tags every such value [L]. Transferring LFP/graphite electrochemistry from a
2 Ah 18650 to a 15.4 Ah 4680 is that operation again, with the same tag and the
same caveat: these are material properties, not cell properties.

The distinction that makes it work is the one the paper's scope line already
draws. GEOMETRY belongs to the cell and is measured on THIS cell [M]. MATERIAL
properties belong to the chemistry and are transferred [L]. The three rows
where that distinction bites hardest are marked CELL-SPECIFIC below: cathode
porosity, cathode coating thickness, anode coating thickness. About:Energy's
18650 has a cathode porosity of 0.204; this cell's weighed mass implies 0.419.
Transferring that number would make this a model of their cell in a 4680 can.

WHAT THE DFN CANNOT DO FOR LFP, STATED UP FRONT
-----------------------------------------------
LiFePO4 discharges through a two-phase transformation, not a solid solution.
The DFN's single-particle Fickian diffusion does not describe that, and three
consequences follow that this module does not hide:

  1. Ds below is an EFFECTIVE quantity. It absorbs phase-boundary kinetics and
     is not a diffusivity in the sense the NMC set's Ds is. It is tagged [L]
     with this note attached, and no physical meaning is claimed for it.
  2. The model cannot represent charge/discharge hysteresis. Real LFP cells
     show tens of millivolts of it. Test T2 of the lab campaign measures it;
     the model will not reproduce it.
  3. About:Energy report their own set degrading "in voltage and capacity
     prediction at higher rates in the limit of low state-of-charge". Liu's
     88.1 % retention at 2C is the test of exactly that corner, and a miss
     there is reported as the model's limit rather than tuned away.

None of this prevents the model from being useful within its limits, which is
the position About:Energy themselves take in publishing the set at all.

LICENCE
-------
[S2] is CC-BY-SA 4.0. Share-alike propagates to derivative WORKS; their JSON
file is a work, individual parameter values are facts. This module therefore
TRANSCRIBES values with [L] tags citing About:Energy and does not redistribute
their file. Our own data stays CC BY 4.0.

SOURCES
    [S1] Chen et al., J. Electrochem. Soc. 167 080534 (2020). LG M50 21700.
         Used here only for the electrolyte, which is the same carbonate
         family, and for values About:Energy do not report.
    [S2] About:Energy, BPX parameterisation of an LFP|graphite 2 Ah 18650
         cylindrical cell. https://github.com/About-Energy-OpenSource/
         About-Energy-BPX-Parameterisation, CC-BY-SA 4.0.
    [S3] Nyman et al., Electrochim. Acta 53 6356 (2008). Electrolyte
         transport correlations.
    [S4] Liu, Yao, Yang, Gonsoulin, Larson, Condon, Attia and Meng, "Design
         and Performance of the BYD LFP/Graphite 4680 Cylindrical Cell",
         J. Electrochem. Soc. 172 110536 (2025). CC BY 4.0. Teardown.
    [S5] Batemo Cell Explorer, BYD FC4680, public page. Measured C/10
         capacity and current ratings.

TAGS
    [M] measured on this cell, or reported for it by a teardown
    [L] literature value for the same chemistry, transferred
    [D] derived here from tagged quantities by a stated relation
    [E] estimated by the present authors, with the reasoning recorded
"""

from __future__ import annotations

import numpy as np

import byd_parts as BP
import byd_stack as BS
from gw2d import ANODE, CATHODE, CC_NEG, CC_POS, SEP, Layer, Params
from gw4680 import De_nyman2008, kappa_nyman2008

UM = 1.0e-4                      # cm per micron

# ---------------------------------------------------------------------------
#  PROVENANCE REGISTRY
# ---------------------------------------------------------------------------
PROV: dict = {}


def _p(name, tag, value, unit, note):
    """Register a parameter and return its value."""
    assert tag in ("M", "L", "D", "E"), tag
    PROV[name] = dict(tag=tag, value=value, unit=unit, note=note)
    return value


# ---------------------------------------------------------------------------
#  OPEN-CIRCUIT POTENTIALS
#
#  Transcribed from [S2]'s BPX expressions. Both are functions of the
#  electrode's own stoichiometry.
# ---------------------------------------------------------------------------
def ocp_lfp_raw(theta):
    """[L] LiFePO4 positive-electrode OCP [S2].

    The plateau is the defining feature: |dU/dtheta| is about 0.015 V across
    the working window, against O(1) V for NMC811. The two exponential terms
    are the end-of-window rises; the linear term is the plateau's slight tilt.
    """
    x = np.asarray(theta, dtype=float)
    return (3.41285712e+00
            - 1.49721852e-02 * x
            + 3.54866018e+14 * np.exp(-3.95729493e+02 * x)
            - 1.45998465e+00 * np.exp(-1.10108622e+02 * (1.0 - x)))


def ocp_graphite_raw(theta):
    """[L] Graphite negative-electrode OCP [S2].

    A different graphite fit from the Chen2020 one the NMC set uses. Kept
    from [S2] rather than swapped for Chen's, because [S2]'s stoichiometry
    window, cs_max and kinetics were fitted against THIS curve and the four
    are not separable.
    """
    x = np.asarray(theta, dtype=float)
    return (5.29210878e+01 * np.exp(-1.72699386e+02 * x)
            - 1.17963399e+03
            + 1.20956356e+03 * np.tanh(6.72033948e+01 * (x + 2.44746396e-02))
            + 4.52430314e-02 * np.tanh(-1.47542326e+01 * (x - 1.62746053e-01))
            + 2.01855800e+01 * np.tanh(-2.46666302e+01 * (x - 1.12986136e+00))
            + 2.01708039e-02 * np.tanh(-1.19900231e+01 * (x - 5.49773440e-01))
            + 4.99616805e+01 * np.tanh(-6.11370883e+01 * (x + 4.69382558e-03)))



# ---------------------------------------------------------------------------
#  WINDOW GUARD -- why the OCPs are clamped, and why that is not a fudge
#
#  [S2]'s OCP expressions are FITS, valid on the stoichiometry window they were
#  fitted over and nowhere else. Outside it they do not degrade gracefully;
#  they diverge. The cathode fit carries a 3.55e14 * exp(-395.7 x) term, so
#
#      theta = 0.0875 (the window edge)        3.74 V
#      theta = 0.0500                      905 000    V
#      theta = 0.0000                        3.55e14  V
#
#  and the graphite fit reaches 904 V at theta = -0.02. The NMC set's Chen2020
#  OCPs are tanh-based and bounded everywhere, which is why this problem is
#  peculiar to this cell and why it was not anticipated.
#
#  THIS IS WHAT STOPPED THE LFP MODEL RUNNING. Newton does not respect physical
#  bounds between iterations: one overshoot of the surface stoichiometry below
#  0.08 puts a megavolt into the residual, the iteration is destroyed, and the
#  reported symptom is a cell that "discharges" to its 2.0 V cut-off at 25 %
#  depth of discharge with a 482 K hot spot. Both numbers are artefacts. The
#  underlying OCV curve, tabulated over the window, is a textbook LFP plateau.
#
#  The guard evaluates the fit inside the window and extrapolates LINEARLY
#  outside it, using the fit's own slope at the edge. Linear rather than a flat
#  clamp because Newton needs a gradient to be pushed back by; a flat clamp
#  removes the restoring force exactly where it is needed. Inside the window --
#  which is the whole of the cell's operating range -- nothing is altered, and
#  ocp_*_raw() remain available so that the unmodified fit can still be plotted
#  and audited.
# ---------------------------------------------------------------------------
def _window_guard(fn, lo, hi, h=1e-6):
    """[D] Evaluate a fit on [lo, hi]; extrapolate linearly beyond it."""
    def guarded(theta):
        x = np.asarray(theta, dtype=float)
        xc = np.clip(x, lo, hi)
        y = fn(xc)
        below, above = x < lo, x > hi
        if np.any(below):
            slope = (fn(lo + h) - fn(lo)) / h
            y = np.where(below, fn(lo) + slope * (x - lo), y)
        if np.any(above):
            slope = (fn(hi) - fn(hi - h)) / h
            y = np.where(above, fn(hi) + slope * (x - hi), y)
        return y if y.ndim else float(y)
    guarded.__name__ = fn.__name__ + "_guarded"
    guarded.__doc__ = (fn.__doc__ or "") + f"\n\nWindow-guarded on [{lo}, {hi}]."
    return guarded


# The guarded fits, bound to [S2]'s own window. These are the names the rest of
# the module and the solver use; ocp_*_raw stay available for plotting the
# unmodified fit. The window numbers are repeated here rather than referenced
# because the THETA_* registry entries are defined further down the file.
_ocp_lfp_guarded = _window_guard(ocp_lfp_raw, 0.0875, 0.95038)
ocp_graphite_aboutenergy = _window_guard(ocp_graphite_raw, 0.0016261, 0.82258)


# ---------------------------------------------------------------------------
#  END-OF-LITHIATION DROP (calibrated, paper 1 v2 Section 5.2, 1 Oct 2026)
#
#  The guard above extrapolates the [S2] fit flat beyond theta = 0.950 and so
#  removes the drop that ends the discharge of LiFePO4 (the raw fit falls from
#  3.39 V at 0.95 to 1.94 V at 1.0). With the guard alone the model ended the
#  discharge on the anode and missed the 2C knee and peak of the bench test.
#  The drop is restored as a smooth term -A ((theta - 1 + w)/w)^n for
#  theta > 1 - w, with A = 1.4 V fixed by the fit's value at theta = 1 and the
#  width w and exponent n calibrated on the bench test. An effective
#  description [E] of a drop whose shape, in a real electrode, follows from the
#  distribution of particle states.
# ---------------------------------------------------------------------------
OCP_DROP_A = 1.4          # V   [D] the raw fit's drop between theta 0.95 and 1
OCP_DROP_W = 0.131        # -   [E] calibrated 1 Oct 2026
OCP_DROP_N = 2.58         # -   [E] calibrated 1 Oct 2026


def _with_drop(fn, A=OCP_DROP_A, w=OCP_DROP_W, n=OCP_DROP_N):
    tw = 1.0 - w

    def dropped(theta):
        x = np.asarray(theta, dtype=float)
        y = np.asarray(fn(x), dtype=float) - A * np.clip((x - tw) / w, 0.0, None) ** n
        return y if y.ndim else float(y)
    dropped.__name__ = fn.__name__ + "_drop"
    dropped.__doc__ = (fn.__doc__ or "") + f"\n\nEnd-of-lithiation drop A={A} V, w={w}, n={n}."
    return dropped


ocp_lfp_aboutenergy = _with_drop(_ocp_lfp_guarded)


# ---------------------------------------------------------------------------
#  ENTROPIC COEFFICIENTS
#
#  The NMC set has NONE of these -- it sets dUdT = 0 "for want of data", and
#  validate_lumped found 0.44 W of measured heat generation the NMC model
#  cannot account for, which is almost certainly this term. [S2] carry both,
#  so the LFP set is BETTER equipped than the NMC one here.
#
#  gw2d takes a single scalar dUdT per electrode. The full curves are averaged
#  over the working window, which is a real approximation and is noted as such.
#  Resolving it would mean making dUdT a callable in gw2d -- worth doing, and
#  listed as an open item rather than done silently here.
# ---------------------------------------------------------------------------
_ENT_POS_X = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                       0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                       1.0])
_ENT_POS_Y = np.array([1.0000e-04, 4.7145e-05, 3.7666e-05, 2.0299e-05,
                       5.9833e-06, -4.6859e-06, -1.3966e-05, -2.3528e-05,
                       -3.1041e-05, -3.4137e-05, -3.4772e-05, -3.5300e-05,
                       -3.6153e-05, -3.7454e-05, -3.9224e-05, -4.2114e-05,
                       -4.7143e-05, -5.4213e-05, -6.2500e-05, -7.1429e-05,
                       -8.0000e-05])


def dudt_lfp(theta):
    """[L] LiFePO4 entropic coefficient, V/K, interpolated from [S2]."""
    return np.interp(np.asarray(theta, dtype=float), _ENT_POS_X, _ENT_POS_Y)


def dudt_graphite(theta):
    """[L] Graphite entropic coefficient, V/K [S2]."""
    x = np.asarray(theta, dtype=float)
    return (-0.1112 * x + 0.02914
            + 0.3561 * np.exp(-((x - 0.08309) ** 2) / 0.004616)) / 1000.0


# ---------------------------------------------------------------------------
#  GEOMETRY -- from this cell, not transferred
# ---------------------------------------------------------------------------
T_CU = _p("Cu foil thickness (full)", "M", BP.T_CU, "cm",
          "[S4] ~10 um. The model resolves one half-stack about the foil "
          "centreline, so the Layer below carries half of this.")
T_AL = _p("Al foil thickness (full)", "M", BP.T_AL, "cm",
          "[S4] ~10 um.")
T_CATHODE = _p("cathode coating thickness", "D", BS.T_CATH_1S, "cm",
               "CELL-SPECIFIC. Derived from [S4]'s ~150 um through-sheet "
               "measurement less the 10 um Al foil, halved: 70 um per side. "
               "[S2]'s 18650 has 64.3 um -- close, but this cell's value is "
               "measured and is used.")
T_ANODE = _p("anode coating thickness", "M", BS.T_AN_1S, "cm",
             "CELL-SPECIFIC. [S4] ~125 um through-sheet less 10 um Cu, "
             "halved: 54.5 um per side. [S2]'s 18650 has 44.4 um.")
T_SEP = _p("separator thickness", "D", 13.73 * UM, "cm",
           "TRIANGULATED, not assumed -- see byd_triangulate. [S4] do not "
           "report it, and it cannot be taken from [S2] (20 um) because at "
           "that thickness the wound roll does not fit inside the can. It is "
           "determined instead by the mass, capacity and volume closures "
           "solved simultaneously, which give 13.7 um. Superseded values: "
           "12 um assumed by analogy with the NMC cell, and [S2]'s 20 um. "
           "revision of byd_parts assumed 12 um by analogy with the NMC "
           "cell's LG M50 separator; [S2]'s value is for an LFP cell and is "
           "preferred. The winding-period closure in byd_parts must be "
           "re-run against this value.")

WIND_LENGTH = _p("winding length H", "M", BP.CATH_L, "cm",
                 "[S4] 5370 mm cathode. The unrolled winding coordinate.")
ELEC_HEIGHT = _p("electrode height", "M", BP.CATH_H, "cm",
                 "[S4] 69 mm cathode; the anode is 71 mm.")
DEPTH = _p("model depth", "D", 2.0 * BP.CATH_H, "cm",
           "2 x electrode height, because the electrode is coated on BOTH "
           "faces and the model resolves one half-stack.")

# ---------------------------------------------------------------------------
#  POROSITY -- the row that must not be transferred
# ---------------------------------------------------------------------------
EPS_CATH = _p("cathode porosity", "D", 0.3310, "-",
              "CELL-SPECIFIC, and TRIANGULATED -- see byd_triangulate.\n"
              "    Two independent routes to this number disagreed: the mass "
              "closure gave 0.419 and the measured areal capacity gave 0.331. "
              "The disagreement was not about the electrode. The mass route "
              "needs the can wall thickness, which [S4] do not report and "
              "which was being assumed at 0.50 mm, inherited from the NMC "
              "cell. Adding the volume closure -- the wound roll must fit the "
              "bore -- makes the system square in three unknowns (porosity, "
              "can wall, separator) and it has a unique solution. Across "
              "every plausible perturbation of the inputs the porosity stays "
              "in 0.29-0.37 and never reaches 0.419. [S2]'s 18650 has 0.204; "
              "transferring that would model their cell in our can.")
CAN_WALL = _p("can wall thickness", "D", 0.04182, "cm",
              "TRIANGULATED -- see byd_triangulate. 0.418 mm (0.403 mm "
              "before the anode porosity was propagated into the mass "
              "closure, 27 Sep 2026). [S4] do not "
              "report it and it is the one geometric quantity no public "
              "source gives for this cell. Battery Design quote 0.55 mm for a "
              "STRUCTURAL nickel-plated steel 4680 can; this cell is not "
              "structural, so a thinner wall is expected, and the three "
              "closures agree on 0.42 mm. Falsifiable without sectioning: the "
              "implied can mass is 50 g against 61 g at 0.55 mm.")

EPS_AN = _p("anode porosity", "D", 0.3190, "-",
            "DERIVED from [S4]'s measured anode areal capacity, 2.34 "
            "mAh/cm^2, inverted through Q'' = eps_s L cs_max dtheta F. "
            "Superseded a 0.28 estimate, which over-predicted that capacity "
            "by 6.2 %.\n"
            "    This is the LFP set's mutual-corroboration result and it is "
            "worth stating in the paper. The anode enters NONE of the three "
            "closures that determine the cathode porosity, the can wall and "
            "the separator, so this is an independent route on an independent "
            "electrode -- and it gives 0.319 against the cathode's "
            "triangulated 0.331. Two unrelated determinations landing within "
            "0.012 is what a pair of electrodes calendered on one line should "
            "look like, and neither was tuned toward the other.\n"
            "    [S2]'s 18650 has 0.207 for its anode. Again the difference "
            "is the cell, not the chemistry.")
EPS_SEP = _p("separator porosity", "L", 0.47, "-",
             "[S2] 0.47, identical to the NMC set's assumption.")

# ---------------------------------------------------------------------------
#  ELECTROLYTE
# ---------------------------------------------------------------------------
CE0 = _p("ce0", "L", 1.0e-3, "mol/cm^3", "[S2] 1000 mol/m^3.")
T_PLUS = _p("t_plus", "L", 0.259, "-",
            "[S2] 0.259. The NMC set carries 0.2594 from [S1] -- the same "
            "number to three figures, as expected for the same electrolyte "
            "family.")
DLNF = _p("dlnf_dlnce", "L", 0.0, "-",
          "Thermodynamic factor of 1, as [S1]. [S2] do not report it.")
EACT_DE = _p("Eact_De", "L", 17.1e3, "J/mol",
             "[S2] 17.1 kJ/mol. This was [E] in the NMC set -- an estimate "
             "for want of a source. [S2] supply it, so it is [L] here.")
EACT_KAP = _p("Eact_kap", "L", 17.1e3, "J/mol", "[S2] 17.1 kJ/mol.")

# ---------------------------------------------------------------------------
#  OPERATING POINT
# ---------------------------------------------------------------------------
V_CUT = _p("V_cut", "M", 2.0, "V", "[S4], [S5]. LFP cut-off, not NMC's 2.5.")
T0 = _p("T0", "M", 298.15, "K", "25 C.")
TA = _p("Ta", "M", 298.15, "K", "25 C ambient.")
H_CAN = _p("can-surface heat transfer coefficient", "M", 22.4e-4, "W/cm^2/K",
           "22.4 W/m^2/K over the whole can: hA = 0.337 W/K measured in the "
           "bench fixture (fan chamber, toggle clamp) from the energy balance "
           "over a full 1C cycle, bench test of 29 Sep 2026 (paper 1 Section "
           "5.2). A property of that fixture; the still-air estimate used "
           "before the test was 10 W/m^2/K.")
R_HARDWARE = _p("hardware series resistance", "E", 0.5e-3, "ohm",
                "Terminal and weld contacts, in the voltage path.")
R_CONTACT_HEAT = _p("holder contact resistance, heat only", "E", 1.34e-3, "ohm",
                    "I^2 R_c at the can node, outside the voltage path; "
                    "calibrated on the bench test (paper 1 Section 5.2, "
                    "1 Oct 2026). A property of the bench clamp (four spring "
                    "contacts and one pointed contact), not of the cell; zero "
                    "in the plain parameter set.")

# ---------------------------------------------------------------------------
#  CALIBRATED UTILISATION WINDOW
#
#  The NMC set calibrates theta_max to Ank's measured capacity and states
#  plainly that capacity is therefore NOT a prediction. The same move is made
#  here against Liu's 15.41 Ah, for the same reason and with the same
#  disclosure. calibrate_theta_max() below solves for it rather than hard-
#  coding a number, so it moves if the geometry does.
# ---------------------------------------------------------------------------
# 100 % SOC is DEFINED by the bench-test protocol (author, 27 Sep 2026): the
# state reached by a CC charge to 3.65 V followed by CV to C/20 (the test
# plan of 21 Sep 2026, lab/fc4680_test_plan.pdf, as run). The transferred [S2]
# stoichiometries put OCV(100 %) at 3.649 V, within the overpotential of the
# C/20 taper, so they need no shift. (An 18 Sep plan used 3.60 V; superseded.)
CHARGE_CUTOFF_V = 3.65        # [M] test protocol
THETA0_POS = _p("cathode theta0", "L", 0.0875, "-",
                "[S2] minimum stoichiometry: the fully charged state. With the "
                "anode value it gives OCV(100 %) = 3.649 V, consistent with "
                "the test protocol's full charge (CC to 3.65 V, CV to C/20).")
THETA0_NEG = _p("anode theta0", "E", 0.796, "-",
                "Calibrated on the bench test (paper 1 Section 5.2, 1 Oct "
                "2026); corresponds to a loss of cyclable lithium relative to "
                "the transferred window. [S2] maximum stoichiometry, the "
                "transferred value, was 0.82258.")
THETA_MIN_NEG = _p("anode theta_min", "L", 0.0016261, "-", "[S2].")

CS_MAX_POS = _p("cathode cs_max", "L", 21200e-6, "mol/cm^3",
                "[S2] 21200 mol/m^3. Theoretical LiFePO4 is 22806; [S2] use "
                "a practical value reflecting inactive material.")
CS_MAX_NEG = _p("anode cs_max", "L", 31400e-6, "mol/cm^3",
                "[S2] 31400 mol/m^3. The NMC set carries 33133 from [S1] for "
                "the same graphite -- a 5 % difference between two fits of "
                "the same material, which is the size of the uncertainty "
                "such transfers carry.")

RS_POS = _p("cathode particle radius", "L", 5.0e-7 * 100, "cm",
            "[S2] 500 nm. LFP particles are an order of magnitude smaller "
            "than NMC's, which is how the chemistry compensates for its low "
            "conductivity.")
RS_NEG = _p("anode particle radius", "L", 4.8e-6 * 100, "cm", "[S2] 4.8 um.")

DS_POS = _p("cathode Ds", "E", 6.873e-17 * 0.767 * 1e4, "cm^2/s",
            "Calibrated on the bench test (paper 1 Section 5.2, 1 Oct 2026): "
            "0.767 x the transferred [S2] 6.873e-17 m^2/s, i.e. 5.27e-17 "
            "m^2/s, tau_c = r_s^2/D_s = 4740 s. EFFECTIVE, NOT A "
            "DIFFUSIVITY: LiFePO4 discharges through a moving phase boundary, "
            "and this number absorbs that kinetics into a Fickian form. No "
            "physical meaning is claimed for it; see the module docstring.")
DS_NEG = _p("anode Ds", "L", 9.6e-15 * 1e4, "cm^2/s", "[S2] 9.6e-15 m^2/s.")

# Stored as About:Energy report them -- BPX reaction rate constants in
# mol/m^2/s -- and converted to the solver's m_ref by _k_to_mref, which is now
# verified against PyBaMM rather than argued for. The tag is [L] because the
# measurement is theirs; the conversion is [D] and is documented where it lives.
M_REF_POS = _p("cathode m_ref", "L", 9.736e-7, "mol/m^2/s",
               "[S2] BPX reaction rate constant 9.736e-7 mol/m^2/s, converted "
               "by m_ref = k F / (cs_max sqrt(ce0)); see _k_to_mref.")
M_REF_NEG = _p("anode m_ref", "L", 6.872e-6, "mol/m^2/s",
               "[S2] 6.872e-6 mol/m^2/s, same conversion.")

EACT_I0_POS = _p("cathode Eact_i0", "L", 35.0e3, "J/mol",
                 "[S2] 35 kJ/mol. [E] in the NMC set; measured here by "
                 "[S2] and checkable by lab test T3/T4.")
EACT_DS_POS = _p("cathode Eact_Ds", "E", 18.8e3, "J/mol",
                 "Calibrated on the bench test (paper 1 Section 5.2, 1 Oct "
                 "2026): with the transferred [S2] 80 kJ/mol no single "
                 "diffusivity places the knee at both rates.")
EACT_I0_NEG = _p("anode Eact_i0", "L", 55.0e3, "J/mol", "[S2] 55 kJ/mol.")
EACT_DS_NEG = _p("anode Eact_Ds", "L", 30.0e3, "J/mol", "[S2] 30 kJ/mol.")

SIGMA_POS = _p("cathode sigma", "L", 0.8e-2, "S/cm",
               "[S2] 0.8 S/m. Carbon-coated LFP; the NMC set's cathode is "
               "0.18 S/m, so this electrode is the better conductor of the "
               "two despite LFP's reputation.")
SIGMA_NEG = _p("anode sigma", "L", 7.46e-2, "S/cm", "[S2] 7.46 S/m.")


def _k_to_mref(k, cs_max, ce0=1.0e-3):
    """[D] A BPX reaction rate constant as gw2d's m_ref. RESOLVED 18 Sep 2026.

    WHAT WAS WRONG, AND FOR HOW LONG
    --------------------------------
    This function used to return F * k * 1e5 on a dimensional argument, and
    carried a docstring saying so and admitting it failed a sanity check by
    six orders of magnitude. The kinetics rows were marked OPEN on that basis
    and the LFP model was not to be run quantitatively.

    The error was in reading BPX's own definition. BPX does NOT write the
    exchange current density with raw concentrations. It NORMALISES them:

        BPX          j0 = F k sqrt( (ce/ce0) (c/cs_max) (1 - c/cs_max) )
        Gu and Wang  i0 = m_ref sqrt( ce c (cs_max - c) )          [this model]
        PyBaMM       j0 = k' sqrt( ce c (cs_max - c) )   -- same form as ours

    Matching them term by term leaves no freedom:

        m_ref = k F / (cs_max * sqrt(ce0))

    The normalisation contributes exactly the cs_max * sqrt(ce0) that was
    missing. For the graphite anode that denominator is 9.9e5 -- which IS the
    "six orders of magnitude". The old factor was not a little wrong; it had
    dropped a whole group of terms, and the sanity check was detecting a real
    error rather than a unit slip.

    HOW IT WAS CHECKED, RATHER THAN ASSERTED
    ----------------------------------------
    crosscheck_pybamm.verify_kinetics() writes a schema-valid BPX file
    carrying these rate constants, loads it with
    pybamm.ParameterValues.create_from_bpx, evaluates PyBaMM's own
    exchange-current function at a stated state, and compares it against
    m_ref * sqrt(ce c (cs_max - c)) with the m_ref this function returns. The
    two agree to machine precision for both electrodes. That is not a
    plausibility argument; it is the same number computed twice through
    independent code.

    AND AN INDEPENDENT CORROBORATION
    --------------------------------
    The conversion was derived from the cathode and the anode together, but
    nothing tied either to an outside value. The LFP cell's graphite anode
    comes out at m_ref = 6.68e-2 in these units, against the NMC cell's
    graphite at 6.48e-2 taken from Chen2020 -- two unrelated parameterisations
    of the same material, 3 % apart, neither tuned toward the other. The
    cathode lands 24x below the NMC cathode, which is the expected direction
    and size for LiFePO4's intrinsic kinetics.

    UNITS
    -----
    k is BPX's constant in mol m^-2 s^-1; cs_max and ce0 are this module's
    stored CGS concentrations in mol/cm^3. The 1e5 converts the resulting SI
    m_ref, in A m^-2 (m^3/mol)^1.5, into the (A/cm^2)(cm^3/mol)^1.5 the solver
    holds -- the same factor housestyle uses in the other direction to print
    the row.
    """
    F = 96485.33212
    m_si = k * F / ((cs_max * 1e6) * (ce0 * 1e6) ** 0.5)
    return m_si * 1e5


# ---------------------------------------------------------------------------
#  THE LAYER STACK
# ---------------------------------------------------------------------------
def layers_byd4680():
    """[D] The half-symmetric stack, Cu centreline outward.

    Thermal properties come from byd_stack's phase-consistent averaging --
    solids and electrolyte counted once each -- exactly as the NMC set does
    after its rebuild.
    """
    def eff(name, eps):
        """[D] Effective density and SPECIFIC HEAT for a porous layer.

        The division by rho is not cosmetic and was missing. byd_stack._eff
        returns (rho_eff, (rho c_p)_eff) -- a density and a VOLUMETRIC heat
        capacity -- because byd_stack's own inventory() wants the product and
        multiplies it by a volume. gw4680._eff returns (rho_eff, c_p,eff),
        dividing before it hands the value to a Layer, with a comment saying
        why. This function took byd_stack's second return value and passed it
        into Layer's c_p field, so every LFP layer carried a specific heat
        inflated by its own density: 2.66 J/g/K for the cathode where the
        composite is about 0.99, and a roll thermal mass of 542 J/K where the
        inventory that the mass closure audits says 243.

        The consequence was confined to the LFP set and pointed one way: a
        thermal mass more than twice too large under-predicts every
        temperature rise by the same factor. It is the same failure the NMC
        set had before its rebuild -- density and heat capacity averaged over
        different inventories -- in a different disguise, and it is worth
        noting that neither mass closure could catch it, because both close on
        MASS and this is an error in heat capacity alone.
        """
        rho, rho_cp = BS._eff(name, eps)
        return rho, rho_cp / rho

    rho_a, cp_a = eff("anode", EPS_AN)
    rho_s, cp_s = eff("separator", EPS_SEP)
    rho_c, cp_c = eff("cathode", EPS_CATH)

    return [
        Layer(name="Cu", kind=CC_NEG, L=T_CU / 2.0,
              rho=8.96, cp=0.385, lam=4.01, sigma=5.96e5),
        Layer(
            name="anode", kind=ANODE, L=T_ANODE,
            rho=rho_a, cp=cp_a,
            lam=1.04e-2,               # [E] 1.04 W/m/K, as the NMC set
            sigma=SIGMA_NEG,
            eps_e=EPS_AN,
            eps_s_explicit=1.0 - EPS_AN - 0.06,   # [E] 6 % binder + carbon
            rs=RS_NEG, cs_max=CS_MAX_NEG,
            theta0=THETA0_NEG, theta_min=THETA_MIN_NEG,
            Ds=DS_NEG, m_ref=_k_to_mref(M_REF_NEG, CS_MAX_NEG),
            alpha_a=0.5, alpha_c=0.5,
            Eact_i0=EACT_I0_NEG, Eact_Ds=EACT_DS_NEG,
            dUdT=float(np.mean(dudt_graphite(
                np.linspace(THETA_MIN_NEG, THETA0_NEG, 101)))),
        ),
        Layer(name="separator", kind=SEP, L=T_SEP,
              rho=rho_s, cp=cp_s, lam=0.33e-2, eps_e=EPS_SEP,
              eps_p=1.0 - EPS_SEP),
        Layer(
            name="cathode", kind=CATHODE, L=T_CATHODE,
            rho=rho_c, cp=cp_c,
            lam=1.5e-2,                # [E] 1.5 W/m/K, as the NMC set
            sigma=SIGMA_POS,
            eps_e=EPS_CATH,
            eps_s_explicit=1.0 - EPS_CATH - 0.06,
            rs=RS_POS, cs_max=CS_MAX_POS,
            theta0=THETA0_POS,
            theta_max=0.95038,         # [L] [S2]; CALIBRATED below
            Ds=DS_POS, m_ref=_k_to_mref(M_REF_POS, CS_MAX_POS),
            alpha_a=0.5, alpha_c=0.5,
            Eact_i0=EACT_I0_POS, Eact_Ds=EACT_DS_POS,
            dUdT=float(np.mean(dudt_lfp(np.linspace(0.0875, 0.95, 101)))),
        ),
        Layer(name="Al", kind=CC_POS, L=T_AL / 2.0,
              rho=2.70, cp=0.900, lam=2.37, sigma=3.77e5),
    ]


def params_byd4680(C_rate=1.0, **kw) -> Params:
    """[D] Assemble the LFP Params. Any field overridable by keyword."""
    p = Params(
        layers=layers_byd4680(),
        H=WIND_LENGTH,
        depth=DEPTH,
        ce0=CE0, t_plus=T_PLUS, dlnf_dlnce=DLNF,
        kappa_fn=kappa_nyman2008,
        De_fn=De_nyman2008,
        Eact_De=EACT_DE, Eact_kap=EACT_KAP,
        ocp_neg=ocp_graphite_aboutenergy,
        ocp_pos=ocp_lfp_aboutenergy,
        C_rate=C_rate,
        T0=T0, Ta=TA,
        V_cut=V_CUT,
        # Smooth floor of 2e-4 cs_max on the concentration factors of the
        # exchange-current density (gw2d.Params.kin_reg): numerical, lets the
        # solver follow the voltage below the knee to 2.0 V; no effect above
        # the knee (paper 1 v2 Section 5.2, change (d)).
        kin_reg=2e-4,
        R_series=BP.can_wall_resistance() + BP.build().R_collect_mean
                 + R_HARDWARE,
        # ---- current collection and thermal topology, as the NMC set.
        # Until 26 Sep 2026 these were left at gw2d's Gu & Wang defaults:
        # collection="tab" (all current leaves through ONE tab at the end of
        # the 5.37 m winding) and a Cartesian y with an isothermal tab
        # (h_tab = inf). The foil path then polarised the far end of the
        # winding, the reaction crowded towards the tab, and the coupled
        # solve lost its time step at 9-58 % DOD -- the failure that was
        # attributed to the flat LFP plateau. It was not: with the collection
        # below the same parameters discharge to cut-off, as PyBaMM's DFN
        # does on the exported BPX set (lfp_pybamm_check.py). The collection
        # resistance is already in R_series (R_collect_mean), so 'tabless'
        # also removes a double count.
        collection="tabless",
        thermal_topology="wound_radial",
        roll_r_core=_ROLL_CORE_R,
        roll_r_out=_roll_outer_radius(),
        roll_height=_ROLL_H,
        roll_period=_winding_period(),
        h_can=H_CAN,
        h_tab=0.0,
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def separator_area():
    """[D] Double-sided coated area, cm^2."""
    return WIND_LENGTH * DEPTH


def capacity_Ah(p=None):
    """[D] Charge in the accessible stoichiometric window."""
    p = params_byd4680() if p is None else p
    F = 96485.33212
    c = [L for L in p.layers if L.kind == CATHODE][0]
    dtheta = c.theta_max - c.theta0
    return (c.eps_s * c.L * c.cs_max * dtheta * F
            * separator_area() / 3600.0)


def capacity_closure():
    """[D] vs [M] -- do mass and capacity describe the same electrode?

    THE AUDIT THIS MODULE EXISTS TO REPORT. It now passes BY
    CONSTRUCTION, and that is a weaker statement than it looks -- read on.

    The cathode can be characterised two independent ways:

      (a) from the WEIGHED CELL. byd_stack solves the porosity that makes the
          phase inventory reproduce the measured 332 g: eps_e = 0.419.
      (b) from the MEASURED AREAL CAPACITY. Liu report 2.09 mAh/cm^2, and
          Q = eps_s L cs_max (theta_max - theta0) F inverts for eps_s, giving
          eps_e = 0.331 at About:Energy's cs_max and stoichiometry window.

    The two disagree by 0.088 in absolute porosity -- the capacity route wants
    a denser electrode than the mass route allows -- which is 14.4 % in areal
    capacity. Neither route used the other's datum, so this is a genuine
    closure audit and it is the LFP analogue of the 4.3 % capacity discrepancy
    that the NMC set resolved by calibrating its stoichiometry window.

    HERE THAT RESOLUTION IS NOT AVAILABLE. Closing 14.4 % with theta_max alone
    would drive it past 1.0, which is unphysical -- the window cannot exceed
    the material. So the discrepancy is reported rather than absorbed, and the
    three candidate causes are named:

      1. cs_max. About:Energy use 21200 mol/m^3, a practical value below
         LiFePO4's theoretical 22806. At the theoretical value the gap halves
         to 0.045 in porosity. A denser, better-utilised cathode than their
         18650 is plausible for a cell designed for energy.
      2. The composite solid density byd_stack assumes, 3.40 g/cm^3. The mass
         closure inverts through it, so an error here propagates directly to
         the porosity. It is [L], not measured on this cell.
      3. Liu's 2.09 mAh/cm^2 may be a nominal design value rather than a
         measurement; their paper does not say which.

    Lab test T1 settles it: the measured capacity at C/10 and C/2, against the
    known area, gives the areal capacity for THIS cell and removes candidate 3.
    Until then the set uses the mass-closure porosity, because mass is the one
    quantity measured directly on the cell we hold.

    A NOTE ON THE EXISTING AUDIT. byd_stack.capacity_consistency() reports
    +0.51 % and the paper quotes it. That number multiplies Liu's areal
    capacity by the electrode area and compares against Liu's cell capacity --
    it tests whether Liu's own two numbers agree, not whether our parameter
    set reproduces either. It should be described as such in the paper, and
    this function is the audit that actually tests the parameter set.
    """
    import byd_stack as _BS
    F = 96485.33212
    p = params_byd4680()
    c = [L for L in p.layers if L.kind == CATHODE][0]
    dth = c.theta_max - c.theta0
    areal_model = c.eps_s * c.L * c.cs_max * dth * F / 3.6
    areal_meas = _BS.AREAL_CATH
    eps_s_needed = areal_meas * 3.6 / (c.L * c.cs_max * dth * F)
    return dict(
        areal_model=areal_model,
        areal_measured=areal_meas,
        err_pct=(areal_model / areal_meas - 1.0) * 100.0,
        eps_s_from_mass=c.eps_s,
        eps_s_from_capacity=eps_s_needed,
        porosity_from_mass=EPS_CATH,
        porosity_from_capacity=1.0 - eps_s_needed - 0.06,
        capacity_model_Ah=areal_model * separator_area() / 1000.0,
        capacity_measured_Ah=_BS.Q_CELL,
    )


def calibrate_theta_max(target_Ah=None, lo=0.30, hi=0.999):
    """[D] Solve theta_max so the window charge matches the measured capacity.

    RETURNS None IF THE TARGET IS UNREACHABLE, which it currently is: see
    capacity_closure(). The NMC set could calibrate its window because its
    discrepancy was 4.3 %; this cell's is 14.4 % and the window would have to
    exceed 1.0. A bisection that silently returns its upper bound would look
    like a successful calibration, so this one checks and refuses.
    """
    target = BS.Q_CELL if target_Ah is None else target_Ah
    p = params_byd4680()
    c = [L for L in p.layers if L.kind == CATHODE][0]
    c.theta_max = hi
    if capacity_Ah(p) < target:
        return None                      # unreachable; do not pretend
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c.theta_max = mid
        if capacity_Ah(p) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def derived_metrics(p=None):
    """[D] Quantities the parameter set implies, for the paper's tables."""
    p = params_byd4680() if p is None else p
    F = 96485.33212
    a = [L for L in p.layers if L.kind == ANODE][0]
    c = [L for L in p.layers if L.kind == CATHODE][0]
    A = separator_area()
    half = sum(L.L for L in p.layers)
    areal_c = (c.eps_s * c.L * c.cs_max * (c.theta_max - c.theta0)
               * F / 3.6)
    areal_a = (a.eps_s * a.L * a.cs_max * (a.theta0 - a.theta_min)
               * F / 3.6)
    return {
        "half-stack thickness [um]": half / UM,
        "full winding period [um]": 2 * half / UM,
        "active separator area [cm^2]": A,
        "cathode areal capacity [mAh/cm^2]": areal_c,
        "anode areal capacity [mAh/cm^2]": areal_a,
        "N/P ratio [-]": areal_a / areal_c,
        "cell capacity [Ah]": capacity_Ah(p),
        "1C current [A]": capacity_Ah(p),
        "OCV at 100% SOC [V]": float(ocp_lfp_aboutenergy(c.theta0)
                                     - ocp_graphite_aboutenergy(a.theta0)),
        "plateau OCV [V]": float(ocp_lfp_aboutenergy(0.5)
                                 - ocp_graphite_aboutenergy(0.45)),
        "a_s anode [1/cm]": a.a_s,
        "a_s cathode [1/cm]": c.a_s,
        "R_series total [mohm]": p.R_series * 1e3,
    }


# ---------------------------------------------------------------------------
#  THE FULL 78-ROW REGISTRY
#
#  The DFN+T needs the same parameters whatever the chemistry -- it is the same
#  model. So the LFP set must fill every row the NMC set fills, and the paper's
#  table has two value columns with no blanks. Rows not registered individually
#  above are completed here: pure-metal properties identical to the NMC set,
#  derived quantities, and the handful that remain open.
#
#  OPEN is a fifth category, orthogonal to the [M]/[L]/[D]/[E] tag. It marks a
#  row whose VALUE is usable but whose STATUS is unresolved -- an estimate with
#  no measurement behind it, a quantity known to be approximate, or one whose
#  audit was consumed. These are the rows the paper prints in red, for both
#  cells, so a reader sees the open questions rather than having to infer them.
# ---------------------------------------------------------------------------
OPEN: dict = {}


def _open(group, field, why):
    OPEN[(group, field)] = why


# --- open for BOTH cells -----------------------------------------------------
for _g in ("anode", "separator", "cathode"):
    _open(_g, "lam", "through-plane conductivity of the composite is an "
                     "engineering estimate for both cells; no measurement "
                     "exists for either electrode and none is planned")
_open("format", "can-surface heat transfer coefficient",
      "the single largest thermal uncertainty; ~80 % of the total thermal "
      "resistance. LFP: measured by lab test T7. NMC: still estimated")
_open("format", "hardware series resistance",
      "terminal and weld contacts. LFP: measured by lab test T3. NMC: still "
      "estimated")

# --- open for the NMC cell only ---------------------------------------------
_open("cathode", "theta_max", "NMC: windows calibrated to the cut-off voltages "
                              "and the C/20 capacity; C/3 is the check. LFP: "
                              "transferred, "
                              "and its capacity closure was consumed by the "
                              "triangulation -- neither cell offers capacity "
                              "as evidence")
_open("cathode", "cs_max", "NMC mixes two Chen2020 variants -- measured for "
                           "capacity, calibrated for kinetics. Documented, not "
                           "resolved")

# --- open for the LFP cell only ---------------------------------------------
# The m_ref rows were OPEN until 18 September 2026 on an unresolved unit
# conversion. They are not open any more: BPX normalises its concentrations,
# giving m_ref = k F / (cs_max sqrt(ce0)), and crosscheck_pybamm verifies that
# against PyBaMM's own exchange-current function to machine precision. The
# standing caveat that remains is the ordinary [L] one -- these are an 18650's
# kinetics transferred to a 4680 -- and that is carried by the tag, not by a
# red mark. See _k_to_mref.
_open("cathode", "Ds", "LFP: an EFFECTIVE quantity absorbing two-phase "
                       "kinetics into a Fickian form; not a diffusivity. "
                       "NMC: set through the diffusion time r_s^2/D_s = 700 s, "
                       "which the measured heat and DC resistance bound from "
                       "above only (about 1400 s)")

# Rows whose registry entry is in the SOURCE's units, not the solver's. The
# registry gives them their tag and their note; the layer gives them their
# value. See the comment in parameter_rows().
# Rows where the REGISTRY supplies the tag and the note but the LAYER supplies
# the value, with the factor that takes the layer's own unit to the row's unit.
#
#   m_ref   the registry holds About:Energy's BPX rate constant in mol/m^2/s;
#           the layer holds the converted Gu-Wang form, same unit as the row.
#   Cu, Al  the row's symbol is L/2 -- the model resolves one half-stack about
#           each foil centreline -- but these rows alias to the registry's FULL
#           foil thickness, so the override printed 10 um against the NMC
#           column's 4 um. A factor of two, in the one table whose entire
#           purpose is that its two columns be comparable, and far too small
#           for the ratio guard to see. The layer is in cm, the row in um.
_KEEP_LAYER_VALUE = {("anode", "m_ref"): 1.0, ("cathode", "m_ref"): 1.0,
                     ("Cu", "L"): 1e4, ("Al", "L"): 1e4}

# Pure metals: identical to the NMC set, because they are the same materials.
_METAL = {
    ("Cu", "rho"): ("L", "Pure copper."),
    ("Cu", "cp"): ("L", "Pure copper."),
    ("Cu", "lam"): ("L", "Pure copper."),
    ("Cu", "sigma"): ("L", "Pure copper."),
    ("Al", "rho"): ("L", "Pure aluminium."),
    ("Al", "cp"): ("L", "Pure aluminium."),
    ("Al", "lam"): ("L", "Pure aluminium."),
    ("Al", "sigma"): ("L", "Pure aluminium."),
}

# Everything else the layer walk needs a tag and note for.
_EXTRA = {
    ("anode", "rho"): ("D", "Phase-consistent: solid and electrolyte counted "
                            "once each at the derived porosity."),
    ("anode", "cp"): ("D", "Phase-consistent, as above."),
    ("anode", "lam"): ("E", "1.04 W/m/K composite, as the NMC set."),
    ("anode", "eps_s_explicit"): ("D", "1 - porosity - 6 % binder and carbon, "
                                       "the binder fraction confirmed from "
                                       "[S2]'s own a_s and particle radius."),
    ("anode", "alpha_a"): ("L", "Symmetric transfer coefficient."),
    ("anode", "alpha_c"): ("L", "Symmetric transfer coefficient."),
    ("anode", "dUdT"): ("L", "[S2] analytic expression, averaged over the "
                             "working window, as the NMC set's O'Regan et al. "
                             "(2022) curve."),
    ("anode", "theta_max"): ("L", "[S2] maximum stoichiometry."),
    ("separator", "rho"): ("D", "Phase-consistent."),
    ("separator", "cp"): ("D", "Phase-consistent."),
    ("separator", "lam"): ("E", "0.33 W/m/K, as the NMC set."),
    ("separator", "eps_p"): ("D", "Polymer remainder, 1 - porosity."),
    ("cathode", "rho"): ("D", "Phase-consistent at the triangulated porosity."),
    ("cathode", "cp"): ("D", "Phase-consistent."),
    ("cathode", "lam"): ("E", "1.5 W/m/K composite, as the NMC set."),
    ("cathode", "eps_s_explicit"): ("D", "1 - porosity - 6 % binder/carbon."),
    ("cathode", "alpha_a"): ("L", "Symmetric."),
    ("cathode", "alpha_c"): ("L", "Symmetric."),
    ("cathode", "dUdT"): ("L", "[S2] 21-point table, averaged over the window."),
    ("cathode", "theta_min"): ("L", "[S2]."),
    ("cathode", "theta_max"): ("L", "[S2] maximum stoichiometry, 0.950. "
                                    "TRANSFERRED, not calibrated -- unlike the "
                                    "NMC cell, whose window was narrowed to "
                                    "match its measured capacity. Calibration "
                                    "was attempted here and is UNREACHABLE: "
                                    "closing the gap would need the window "
                                    "past 1.0. The triangulation resolved it "
                                    "instead, through the porosity."),
    ("format", "jelly-roll height"): ("M", "[S4] anode roll height, 71 mm."),
    ("format", "jelly-roll core radius"): ("M", "[S4] hexagonal mandrel."),
    ("format", "jelly-roll outer radius"): ("D", "Spiral closure; equals the "
                                                "bore by construction after "
                                                "triangulation."),
    ("format", "full winding period"): ("D", "Sum of the layer stack, doubled."),
    ("correlation", "$U_+(\\theta)$"): ("L", "[S2] LiFePO4 OCP. Plateau slope "
                                              "0.015 V per unit stoichiometry, "
                                              "about 60x flatter than NMC811."),
    ("correlation", "$U_-(\\theta)$"): ("L", "[S2] graphite OCP."),
    ("correlation", "$\\kappa(c_e)$"): ("L", "[S3] Nyman, as the NMC set: the "
                                              "same carbonate electrolyte."),
    ("correlation", "$D_e(c_e)$"): ("L", "[S3] Nyman, as the NMC set."),
}



# Explicit map from the DFN's (group, field) to this module's registry names.
_ALIAS = {
    ("anode", "eps_e"): "anode porosity",
    ("anode", "rs"): "anode particle radius",
    ("anode", "cs_max"): "anode cs_max",
    ("anode", "theta0"): "anode theta0",
    ("anode", "theta_min"): "anode theta_min",
    ("anode", "Ds"): "anode Ds",
    ("anode", "m_ref"): "anode m_ref",
    ("anode", "sigma"): "anode sigma",
    ("anode", "Eact_i0"): "anode Eact_i0",
    ("anode", "Eact_Ds"): "anode Eact_Ds",
    ("anode", "L"): "anode coating thickness",
    ("cathode", "eps_e"): "cathode porosity",
    ("cathode", "rs"): "cathode particle radius",
    ("cathode", "cs_max"): "cathode cs_max",
    ("cathode", "theta0"): "cathode theta0",
    ("cathode", "Ds"): "cathode Ds",
    ("cathode", "m_ref"): "cathode m_ref",
    ("cathode", "sigma"): "cathode sigma",
    ("cathode", "Eact_i0"): "cathode Eact_i0",
    ("cathode", "Eact_Ds"): "cathode Eact_Ds",
    ("cathode", "L"): "cathode coating thickness",
    ("separator", "L"): "separator thickness",
    ("separator", "eps_e"): "separator porosity",
    ("Cu", "L"): "Cu foil thickness (full)",
    ("Al", "L"): "Al foil thickness (full)",
    ("format", "winding length H"): "winding length H",
    ("format", "electrode height"): "electrode height",
    ("format", "model depth"): "model depth",
    ("format", "separator thickness"): "separator thickness",
    ("format", "cathode coating thickness"): "cathode coating thickness",
    ("format", "anode coating thickness"): "anode coating thickness",
    ("format", "Cu foil thickness (full)"): "Cu foil thickness (full)",
    ("format", "Al foil thickness (full)"): "Al foil thickness (full)",
    ("format", "can-surface heat transfer coefficient"):
        "can-surface heat transfer coefficient",
    ("format", "hardware series resistance"): "hardware series resistance",
    ("cell", "ce0"): "ce0", ("cell", "t_plus"): "t_plus",
    ("cell", "dlnf_dlnce"): "dlnf_dlnce",
    ("cell", "Eact_De"): "Eact_De", ("cell", "Eact_kap"): "Eact_kap",
    ("cell", "T0"): "T0", ("cell", "Ta"): "Ta", ("cell", "V_cut"): "V_cut",
}

_ROLL_CORE_R = _p("jelly-roll core radius", "M",
                  BP.MANDREL_HEX_SIDE + BP.MANDREL_WALL, "cm",
                  "[S4] hexagonal mandrel, 2.5 mm side with a 0.66 mm wall.")
_ROLL_H = _p("jelly-roll height", "M", BP.AN_H, "cm",
             "[S4] anode roll height, 71 mm -- the taller of the two.")


def _winding_period():
    """[D] Full period from the triangulated stack."""
    return 2 * (BP.T_CU / 2 + T_ANODE + T_SEP + T_CATHODE + BP.T_AL / 2)


def _roll_outer_radius():
    """[D] Spiral closure. Equals the bore by construction after
    triangulation -- which is the point, and is why it is [D] and not a
    residual."""
    return np.sqrt(_ROLL_CORE_R ** 2
                   + BP.CATH_L * _winding_period() / np.pi)


_p("full winding period", "D", _winding_period(), "cm",
   "Sum of the triangulated layer stack, doubled.")
_p("jelly-roll outer radius", "D", _roll_outer_radius(), "cm",
   "Archimedean spiral closure. After triangulation this EQUALS the can bore "
   "by construction, because the volume closure was one of the three "
   "equations solved. It is therefore not an independent check for this cell, "
   "unlike the NMC cell where it leaves an 8.8 % residual.")



# Unit conversion between this module's internal CGS and the display unit the
# NMC registry uses for the same row. Without this the two columns of the
# paper's table silently disagree by factors of 10^4 -- which is exactly what
# happened: the first render showed a particle radius of "5.86 um" beside
# "4.8e-4", the same quantity in two units, and nothing flagged it.
_LEN = {"um": 1e4, "mm": 10.0, "cm": 1.0, "m": 0.01}


def _convert(v, frm, to):
    """[D] Convert v from unit `frm` to unit `to`, or return it unchanged."""
    if v is None or frm == to:
        return v
    if frm in _LEN and to in _LEN:
        return v * _LEN[to] / _LEN[frm]
    if frm == "ohm" and to == "mohm":
        return v * 1e3
    return v            # not a length; units already agree or are unconvertible


def parameter_rows():
    """[D] All 78 rows, aligned one-to-one with gw4680.parameter_rows().

    Built by walking the same Params and Layer objects the NMC registry walks,
    so the two tables cannot drift out of alignment: if a row exists for one
    cell it exists for the other, and a missing tag raises rather than
    silently printing a blank.
    """
    import gw4680 as _G
    p = params_byd4680()
    by_name = {L.name: L for L in p.layers}
    rows = []
    for r in _G.parameter_rows():
        g, f = r["group"], r["field"]
        # locate the value on our own objects
        if g in by_name and hasattr(by_name[g], f):
            val = getattr(by_name[g], f)
        elif hasattr(p, f):
            val = getattr(p, f)
        else:
            val = None
        # find a tag and note. EXPLICIT alias table, not fuzzy matching:
        # a provenance registry that guesses which entry it means is not a
        # provenance registry. Eight rows were silently untagged while this
        # was a substring search.
        tag = note = None
        key = _ALIAS.get((g, f))
        if key is None and f"{g} {f}" in PROV:
            key = f"{g} {f}"
        if key is None and f in PROV:
            key = f
        if key is not None:
            rec = PROV[key]
            tag, note = rec["tag"], rec["note"]
            # Normally the registry value IS the value, and overwriting the
            # layer's copy with it is the point: it catches a layer built from
            # something other than the declared number. The exception is a row
            # the registry stores in the SOURCE's units rather than the
            # solver's -- m_ref, where About:Energy report a BPX rate constant
            # in mol/m^2/s and _k_to_mref turns it into the solver's form. For
            # those the registry supplies the provenance and the layer supplies
            # the number, because printing the source's constant under the
            # solver's unit is how the table came to show a value nine
            # thousand times out of line with the NMC column.
            if (g, f) in _KEEP_LAYER_VALUE:
                if val is not None:
                    val = val * _KEEP_LAYER_VALUE[(g, f)]
            else:
                val = _convert(rec["value"], rec["unit"], r["unit"])
        if tag is None and (g, f) in _METAL:
            tag, note = _METAL[(g, f)]
        if tag is None and (g, f) in _EXTRA:
            tag, note = _EXTRA[(g, f)]
        if g == "correlation" and val is None:
            # The NMC table prints these evaluated at the reference state,
            # because a table cell cannot hold a function. Same convention.
            val = {"$U_+(\\theta)$": float(ocp_lfp_aboutenergy(THETA0_POS)),
                   "$U_-(\\theta)$": float(
                       ocp_graphite_aboutenergy(THETA0_NEG)),
                   "$\\kappa(c_e)$": float(kappa_nyman2008(CE0)),
                   "$D_e(c_e)$": float(De_nyman2008(CE0))}.get(f)
        rows.append(dict(group=g, symbol=r["symbol"], field=f, value=val,
                         unit=r["unit"], tag=tag or "?", note=note or "",
                         open=OPEN.get((g, f))))

    # GUARD: two columns of the same row describe the same physical quantity
    # for two cells of one format. They may differ by a factor of a few; they
    # may not differ by a factor of a thousand. Anything that extreme is a
    # unit error, not a design difference, and this is how the first version
    # of this table was caught.
    import gw4680 as _G2
    _nmc = {(x["group"], x["field"]): x for x in _G2.parameter_rows()}
    for row in rows:
        a = _nmc[(row["group"], row["field"])]["value"]
        b = row["value"]
        if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and a and b and a > 0 and b > 0):
            ratio = max(a / b, b / a)
            if ratio > 500 and not row["open"]:
                raise ValueError(
                    f"unit mismatch suspected in {row['group']}.{row['field']}: "
                    f"NMC {a:g} vs LFP {b:g} ({row['unit']}), ratio {ratio:.0f}")
    return rows


def main():
    print("=" * 74)
    print("BYD FC4680 -- LFP/graphite parameter set")
    print("=" * 74)
    print(f"\n{len(PROV)} registered parameters")
    import collections
    tags = collections.Counter(v["tag"] for v in PROV.values())
    print("  tags: " + ", ".join(f"[{k}] {tags[k]}"
                                 for k in ("M", "L", "D", "E") if k in tags))

    print("\nOCP plateau check")
    th = np.linspace(0.2, 0.85, 500)
    u = ocp_lfp_aboutenergy(th)
    slope = np.abs(np.gradient(u, th))
    print(f"  U on the plateau      {u.min():.3f} .. {u.max():.3f} V")
    print(f"  |dU/dtheta| median    {np.median(slope):.4f} V")
    print("  (NMC811 is O(1) V over its window -- about 60x steeper)")

    print("\nDerived, BEFORE calibration")
    d = derived_metrics()
    for k, v in d.items():
        print(f"  {k:36s} {v:12.4f}")

    print("\nCAPACITY CLOSURE -- the audit")
    cc = capacity_closure()
    print(f"  areal capacity, from the parameter set   "
          f"{cc['areal_model']:7.3f} mAh/cm2")
    print(f"  areal capacity, measured [Liu]           "
          f"{cc['areal_measured']:7.3f} mAh/cm2")
    print(f"  residual                                 "
          f"{cc['err_pct']:+7.1f} %")
    print()
    print(f"  cathode porosity from the weighed mass   "
          f"{cc['porosity_from_mass']:7.3f}")
    print(f"  cathode porosity from the capacity       "
          f"{cc['porosity_from_capacity']:7.3f}")
    print(f"  the two routes disagree by               "
          f"{abs(cc['porosity_from_mass']-cc['porosity_from_capacity']):7.3f}")
    print()
    tm = calibrate_theta_max()
    if tm is None:
        print("  theta_max calibration: UNREACHABLE. The window would have to")
        print("  exceed 1.0. Unlike the NMC cell's 4.3 % gap, this one cannot")
        print("  be absorbed into the stoichiometry window, so it is reported.")
        print("  See capacity_closure.__doc__ for the three candidate causes;")
        print("  lab test T1 removes one of them.")
    else:
        print(f"  theta_max calibrated: {tm:.4f}")


if __name__ == "__main__":
    main()



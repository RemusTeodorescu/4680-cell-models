"""The 4680's non-jellyroll components, from the Ank et al. (2023) teardown.

The 2-D model treats everything outside the electrode stack as either a lumped
series resistance or an ambient boundary. For an axially resolved model that is
no longer good enough: the two collector disks, the plastic insulator, the can
and the terminals are exactly what make the two ends of the cell different from
each other, and the axial gradient is a difference between the ends.

Every dimension here is measured, from Ank et al. Figs. 6-8. Material
properties are handbook values for the stated materials. Nothing is fitted.

THE ASYMMETRY THAT MATTERS
--------------------------
Ank Fig. 7: the copper ANODE disk is welded to the cell housing at its OUTER
RING, while the aluminium CATHODE disk is welded to the positive terminal at
its CENTRE. A winding at radius r therefore reaches the negative pole by
travelling OUTWARD through copper, and the positive pole by travelling INWARD
through aluminium. The two paths are anti-correlated in r and partially cancel
-- a compensation that falls out of the construction rather than being designed.

Two consequences:

  * electrically, the collection resistance varies with radius but the total is
    ~0.02 mohm against a cell resistance of ~11 mohm, so it cannot redistribute
    current. `collection_resistance()` quantifies this rather than assuming it.

  * thermally, the ends are NOT interchangeable. The anode disk is bolted to
    the can through its whole outer ring, which is an excellent heat path. The
    cathode disk is separated from the can by a 0.65 mm plastic insulator,
    which is a poor one, and reaches the outside only through the terminal.
    This is why heat injected at the positive terminal stays there.

CGS units (cm, s, A, V, W, K, J, g) throughout, matching gw2d.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
#  Measured geometry, Ank et al. (2023)
# ---------------------------------------------------------------------------
CAN_OD = 4.60          # cm, cylindrical outer diameter
CAN_H = 8.00           # cm, cylindrical height (terminal adds 1 mm more)
CAN_WALL = 0.050       # cm, measured 0.5 mm
ROLL_OD = 4.45         # cm
ROLL_H = 7.10          # cm, after removing both disks
CORE_D = 0.50          # cm, empty void, NO mandrel
DISK_T = 0.020         # cm, both Cu and Al collector disks
PLASTIC_T = 0.065      # cm, insulating disk between cathode disk and can
TERMINAL_D = 1.60      # cm, positive terminal

# The two collector disks are NOT the same diameter and neither equals the
# jelly roll's. Ank et al. Fig. 7 dimensions them separately; the teardown
# TEXT does not, which is why the first version of this model used the roll
# diameter for both. The distinction is not cosmetic here, because the two
# ends are limited by different resistances with different radius dependence:
# the base by the copper rim weld, which scales with CIRCUMFERENCE, and the
# cap by conduction through the polymer sheet, which scales with AREA. So the
# smaller cathode disk cuts the cap conductance quadratically and barely
# touches the base.
ANODE_DISK_OD = 4.40      # cm, Ank et al. Fig. 7(b), "44 mm"
CATHODE_DISK_OD = 4.00    # cm, Ank et al. Fig. 7(a), "40 mm"
# The anode disk's rim carries about a dozen 3 mm notches, so the weld to the
# housing does not run the full circumference. Counted from the drawing rather
# than dimensioned on it, and therefore the one estimate in this geometry.
RIM_NOTCH_FRACTION = 0.26

# ---------------------------------------------------------------------------
#  Material properties [L], handbook values for the stated materials
# ---------------------------------------------------------------------------
MAT = {
    #                 rho      cp       lam     sigma
    #               g/cm^3   J/g/K   W/cm/K    S/cm
    "Cu":   dict(rho=8.96, cp=0.385, lam=4.00, sigma=5.96e5),
    "Al":   dict(rho=2.70, cp=0.900, lam=2.38, sigma=3.50e5),
    # Nickel-plated steel, the usual can material. lam is the value that
    # matters and is the least certain of the three.
    "steel": dict(rho=7.90, cp=0.470, lam=0.15, sigma=1.4e4),
    # PP or PET insulating disk: a thermal barrier, and the point of it
    "plastic": dict(rho=1.20, cp=1.70, lam=0.0022, sigma=0.0),
}


# ---------------------------------------------------------------------------
@dataclass
class Parts:
    """Lumped thermal and electrical properties of everything that is not the
    electrode stack."""
    # thermal capacities, J/K
    C_disk_neg: float
    C_disk_pos: float
    C_can: float
    C_plastic: float
    # conductances, W/K
    G_roll_to_disk_neg: float
    G_roll_to_disk_pos: float
    G_disk_neg_to_can: float
    G_disk_pos_to_can: float
    G_can_axial: float
    # electrical, ohm
    R_collect_mean: float
    # areas, cm^2
    A_can_side: float
    A_can_end: float

    def summary(self) -> Dict[str, float]:
        return {
            "negative disk heat capacity [J/K]": self.C_disk_neg,
            "positive disk heat capacity [J/K]": self.C_disk_pos,
            "can heat capacity [J/K]": self.C_can,
            "plastic disk heat capacity [J/K]": self.C_plastic,
            "roll -> negative disk [W/K]": self.G_roll_to_disk_neg,
            "roll -> positive disk [W/K]": self.G_roll_to_disk_pos,
            "negative disk -> can [W/K]": self.G_disk_neg_to_can,
            "positive disk -> can, through plastic [W/K]": self.G_disk_pos_to_can,
            "can wall axial conductance [W/K]": self.G_can_axial,
            "mean collection resistance [mohm]": self.R_collect_mean * 1e3,
            "can side area [cm^2]": self.A_can_side,
            "can end area [cm^2]": self.A_can_end,
        }


# ---------------------------------------------------------------------------
def collection_resistance(r):
    """Resistance from a winding at radius r to each terminal, ohm.

    Radial spreading in a disk of sheet resistance R_s carries dR = R_s dr /
    (2 pi r), so the path from r to a collection radius r0 is
    (R_s / 2 pi) |ln(r/r0)|.

      negative pole: OUTWARD through copper to the rim, which is welded to the
                     can over its whole circumference
      positive pole: INWARD through aluminium to the terminal weld

    Returns (R_neg, R_pos, R_total), each an array matching r.
    """
    r = np.atleast_1d(np.asarray(r, float))
    Rs_cu = 1.0 / (MAT["Cu"]["sigma"] * DISK_T)
    Rs_al = 1.0 / (MAT["Al"]["sigma"] * DISK_T)
    r_rim = ROLL_OD / 2
    r_term = TERMINAL_D / 2

    R_neg = Rs_cu / (2 * np.pi) * np.log(np.maximum(r_rim / r, 1.0))
    # inside the terminal weld there is no spreading path left to travel
    R_pos = Rs_al / (2 * np.pi) * np.log(np.maximum(r / r_term, 1.0))
    return R_neg, R_pos, R_neg + R_pos


def build_parts(n_ring=200, disks="roll", f_weld=None) -> Parts:
    """Assemble the lumped component model.

    The disks are meshed only to average the collection resistance over the
    roll's annular area; their thermal behaviour is lumped, which is justified
    because a 0.2 mm copper disk is nearly isothermal (its own Biot number is
    of order 1e-4).

    disks="roll"      Both collector disks at the jelly-roll diameter. This is
                      the geometry the RELEASED DATASET was generated with, and
                      it remains the default so that the released code
                      reproduces the released data exactly. It is not the
                      better geometry.
    disks="measured"  The separately dimensioned diameters of Ank et al.
                      Fig. 7, with the anode disk's rim notches. This is the
                      better geometry and it is what the validation against
                      their thermal imaging uses. It raises the predicted
                      cap-to-base difference by about 9 %; diskdia.py measures
                      that, and the manuscript reports both.

    Two geometries in one model is a wart, and the honest reason for it is
    chronological: the dataset was generated before the disk diameters were
    read off the dimensioned drawing. Silently changing the default would make
    the released code stop reproducing the released data, which is a worse
    fault than the one it would fix.
    """
    if disks not in ("roll", "measured"):
        raise ValueError("disks must be 'roll' or 'measured'")
    r_in, r_out = CORE_D / 2, ROLL_OD / 2
    r_can = CAN_OD / 2
    if disks == "measured":
        r_neg, r_pos, notch = (ANODE_DISK_OD / 2, CATHODE_DISK_OD / 2,
                               RIM_NOTCH_FRACTION)
    else:
        r_neg, r_pos, notch = r_out, r_out, 0.0

    # ---- thermal capacities ------------------------------------------------
    A_disk = np.pi * (r_out ** 2 - r_in ** 2)
    A_neg = np.pi * (r_neg ** 2 - r_in ** 2)
    A_pos = np.pi * (r_pos ** 2 - r_in ** 2)
    C_disk_neg = MAT["Cu"]["rho"] * MAT["Cu"]["cp"] * A_neg * DISK_T
    C_disk_pos = MAT["Al"]["rho"] * MAT["Al"]["cp"] * A_pos * DISK_T
    C_plastic = (MAT["plastic"]["rho"] * MAT["plastic"]["cp"]
                 * np.pi * r_can ** 2 * PLASTIC_T)
    V_can = (2 * np.pi * r_can * CAN_H * CAN_WALL      # side wall
             + 2 * np.pi * r_can ** 2 * CAN_WALL)      # two ends
    C_can = MAT["steel"]["rho"] * MAT["steel"]["cp"] * V_can

    # ---- conductances ------------------------------------------------------
    # roll end face -> disk: contact over the folded, welded foil edges. Ank
    # report five laser seams per connector, so this is NOT a full-area
    # contact; taken as a fraction of the face area. [E]
    # It turns out not to matter: this conductance is three orders of
    # magnitude larger than the one it sits in series with, so f_weld from
    # 0.08 to 1.0 moves the pair by under 2 % (diskdia.py). One of the model's
    # two estimated parameters is therefore not load-bearing.
    # Ank's own description of the joint -- "five laser-welded zig-zag seams
    # with a width of about 0.5 mm" -- implies f_weld of 0.02 to 0.05 over a
    # 15.2 cm^2 disk, an order below the 0.35 first assumed. fweld_check.py
    # measures what that is worth on the validation.
    f_weld = 0.35 if f_weld is None else f_weld
    G_roll_disk = MAT["Cu"]["lam"] * A_neg * f_weld / (DISK_T / 2)
    G_roll_disk_pos = MAT["Al"]["lam"] * A_pos * f_weld / (DISK_T / 2)

    # negative disk -> can: welded at the outer ring, so the path is the
    # radial spreading resistance of the disk in series with the weld
    L_ring = 2 * np.pi * r_neg * (1.0 - notch)
    G_disk_neg_can = MAT["Cu"]["lam"] * L_ring * DISK_T / (0.1)

    # positive disk -> can: through the plastic insulator, over the disk area.
    # This is the asymmetry: a 0.65 mm plastic sheet at 0.022 W/m/K.
    G_disk_pos_can = MAT["plastic"]["lam"] * np.pi * r_pos ** 2 / PLASTIC_T

    # can wall, axial conduction between the two ends
    A_wall = 2 * np.pi * r_can * CAN_WALL
    G_can_axial = MAT["steel"]["lam"] * A_wall / CAN_H

    # ---- electrical --------------------------------------------------------
    rr = np.linspace(r_in, r_out, n_ring)
    _, _, R_tot = collection_resistance(rr)
    # area-weighted mean: outer rings carry more of the winding
    w = rr / rr.sum()
    R_mean = float((R_tot * w).sum() / w.sum())

    return Parts(
        C_disk_neg=C_disk_neg, C_disk_pos=C_disk_pos, C_can=C_can,
        C_plastic=C_plastic,
        G_roll_to_disk_neg=G_roll_disk, G_roll_to_disk_pos=G_roll_disk_pos,
        G_disk_neg_to_can=G_disk_neg_can, G_disk_pos_to_can=G_disk_pos_can,
        G_can_axial=G_can_axial,
        R_collect_mean=R_mean,
        A_can_side=2 * np.pi * r_can * CAN_H,
        A_can_end=np.pi * r_can ** 2,
    )


# ---------------------------------------------------------------------------
def terminal_power(current_A, R_contact=0.5e-3):
    """Ohmic power dissipated in the two terminal contacts, W.

    Ank attribute the measured axial gradient to this. Their estimate is
    0.5 mohm per terminal; in a welded pack connection it would be lower. Both
    terminals are on the cap side in their fixture, so ALL of this power enters
    at one end -- which is the point.
    """
    return 2.0 * current_A ** 2 * R_contact


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = build_parts()
    print(__doc__)
    print("=" * 70)
    for k, v in p.summary().items():
        print(f"  {k:46s} {v:12.5g}")

    print("\n  collection resistance vs radius")
    r = np.linspace(CORE_D / 2, ROLL_OD / 2, 9)
    Rn, Rp, Rt = collection_resistance(r)
    print(f"    {'r [mm]':>8s} {'R_neg [uohm]':>13s} {'R_pos [uohm]':>13s}"
          f" {'total [uohm]':>13s}")
    for a, b, c, d in zip(r, Rn, Rp, Rt):
        print(f"    {a*10:8.1f} {b*1e6:13.2f} {c*1e6:13.2f} {d*1e6:13.2f}")

    print("\n  the thermal asymmetry between the two ends:")
    ratio = p.G_disk_neg_to_can / p.G_disk_pos_to_can
    print(f"    negative end to can : {p.G_disk_neg_to_can:8.4f} W/K")
    print(f"    positive end to can : {p.G_disk_pos_to_can:8.4f} W/K"
          f"   (through 0.65 mm plastic)")
    print(f"    ratio               : {ratio:8.1f}x")
    print("    The negative end is welded to the can; the positive end is")
    print("    deliberately insulated from it. Heat entering at the positive")
    print("    terminal has nowhere easy to go, which is the mechanism behind")
    print("    an axial gradient.")


# ---------------------------------------------------------------------------
def lumped_axial(Q_total_W, I_A, R_contact=0.5e-3, h_can=5.0e-3, Ta=298.15,
                 lam_z=0.1429, verbose=True):
    """Four-node lumped estimate of the axial temperature split.

    Nodes: roll top half, roll bottom half, can cap, can bottom. Enough to ask
    whether the construction can produce the gradient Ank measured, and cheap
    enough to ask it before writing a solver.

    The question it settles: Ank imaged the CAN. If the roll cannot support a
    10 K axial spread but the cap can, then the measured gradient is a surface
    effect of how the positive end is built, not a gradient in the electrode.
    """
    p = build_parts()
    r_out, r_can = ROLL_OD / 2, CAN_OD / 2
    A_roll = np.pi * (r_out ** 2 - (CORE_D / 2) ** 2)

    G_ax = lam_z * A_roll / (ROLL_H / 2)          # roll half to roll half
    G_side = h_can * (2 * np.pi * r_can * CAN_H) / 2   # per half, to ambient
    G_end = h_can * np.pi * r_can ** 2                 # one can end to ambient

    # roll half -> its can end, in series through the disk
    ser = lambda a, b: 1.0 / (1.0 / a + 1.0 / b)
    G_top_cap = ser(p.G_roll_to_disk_pos, p.G_disk_pos_to_can)   # plastic!
    G_bot_end = ser(p.G_roll_to_disk_neg, p.G_disk_neg_to_can)   # welded

    P_term = terminal_power(I_A, R_contact)       # enters at the cap
    Q = Q_total_W / 2.0                           # per roll half

    # unknowns: T_top, T_bot, T_cap, T_base  (as rises above ambient)
    A = np.array([
        [G_side + G_ax + G_top_cap, -G_ax, -G_top_cap, 0.0],
        [-G_ax, G_side + G_ax + G_bot_end, 0.0, -G_bot_end],
        [-G_top_cap, 0.0, G_top_cap + G_end + p.G_can_axial, -p.G_can_axial],
        [0.0, -G_bot_end, -p.G_can_axial, G_bot_end + G_end + p.G_can_axial],
    ])
    b = np.array([Q, Q, P_term, 0.0])
    th = np.linalg.solve(A, b)

    out = dict(T_roll_top=th[0], T_roll_bot=th[1], T_cap=th[2], T_base=th[3],
               dT_roll=th[0] - th[1], dT_can=th[2] - th[3],
               G_ax=G_ax, G_top_cap=G_top_cap, G_bot_end=G_bot_end,
               P_terminal=P_term)
    if verbose:
        print(f"\n  lumped axial estimate: Q = {Q_total_W:.1f} W internal, "
              f"I = {I_A:.1f} A, terminal loss {P_term:.2f} W at the cap")
        print(f"    roll half -> roll half   G_ax      = {G_ax:8.4f} W/K")
        print(f"    roll top  -> can cap     (plastic) = {G_top_cap:8.4f} W/K")
        print(f"    roll bot  -> can base    (welded)  = {G_bot_end:8.4f} W/K")
        print(f"    can cap   -> can base    (wall)    = {p.G_can_axial:8.4f} W/K")
        print()
        print(f"    roll top   {th[0]:6.2f} K above ambient")
        print(f"    roll bottom{th[1]:6.2f} K")
        print(f"    can cap    {th[2]:6.2f} K")
        print(f"    can base   {th[3]:6.2f} K")
        print(f"    --> axial spread in the ROLL : {out['dT_roll']:5.2f} K")
        print(f"    --> axial spread on the CAN  : {out['dT_can']:5.2f} K")
    return out


def can_wall_resistance():
    """Electrical resistance along the steel can wall, ohm.

    This matters because the can IS the negative terminal (Ank: the anode disk's
    outer ring is welded to the housing). If a test fixture contacts both poles
    at the CAP, as Ank describe, then the negative current must travel the full
    height of the can wall to reach the anode disk at the base -- through
    nickel-plated steel, which is a poor conductor. That resistance is in series
    with the contact resistance and dissipates its heat in the upper can.

    In a vehicle the can would be contacted at the base and this term vanishes,
    so it is a property of the measurement, not of the cell.
    """
    r = CAN_OD / 2
    A = 2 * np.pi * r * CAN_WALL
    return CAN_H / (MAT["steel"]["sigma"] * A)

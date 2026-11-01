"""The BYD FC4680's geometry and thermal network, in paper 1's own form.

    python byd_parts.py

WHAT THIS IS FOR
----------------
Paper 1 identifies a parameter set for a Tesla-like NMC 4680 and tags every
number with where it came from. This module does the same job for the LFP cell
on the bench, so the two can sit in one table under one identification
procedure. It deliberately stops short of the electrochemistry: OCP, solid
diffusivity, kinetics and the entropic coefficient are NOT here, because LFP's
two-phase plateau breaks the single-particle diffusion the DFN assumes and
those parameters cannot be honestly tabulated against that model. Geometry,
material inventory and the thermal/electrical network do not have that problem
-- they are the same physics for both cells -- so they are here in full.

PROVENANCE TAGS, matching gw4680
    [M]  measured and reported in a source
    [D]  derived here from [M] quantities
    [L]  literature / handbook value for the stated material
    [E]  estimated; the honest ones, each with its sensitivity

SOURCES
    [S1] Liu, Yao, Yang, Gonsoulin, Larson, Condon, Attia and Meng,
         "Design and Performance of the BYD LFP/Graphite 4680 Cylindrical
         Cell", J. Electrochem. Soc. 172 110536 (2025).
         DOI 10.1149/1945-7111/ae1e33. CC BY 4.0.
         (The last author is Y. Shirley Meng -- surname Meng. The journal's
         byline renders it "Ying Shirley-Meng", which is where an earlier
         version of this block got "Shirley-Meng" from.)
    [S2] Battery Design, "BYD FC4680P" (dimensions, voltages, temperatures).
    [S3] About:Energy, Voltt cell library, BYD FC4680 (capacity, rate limit).

    The manufacturer's specification sheet is marked Company Confidential and
    is deliberately NOT cited or relied upon. Every value here is independently
    available from [S1]-[S3]. A specification value is a fact about the cell,
    not a creative work; citing the confidential document, or shipping it,
    would be a different matter.

THE ONE THING [S1] DOES NOT GIVE
--------------------------------
The jelly roll's outer diameter, and the can wall thickness. Neither is in the
paper. The roll diameter is recoverable, though, because [S1] DOES give the
unwound electrode length, and a spiral of known length, core and layer pitch
has only one outer radius:

    L = integral of circumference  =>  r_out^2 = r_core^2 + L * p / pi

with p the full winding period. Turned around, a measured r_out fixes p. Both
forms are below, and the disagreement between them is the honest uncertainty
on this cell's roll -- the same consistency check paper 1 runs on the NMC cell,
where it leaves a 10.7 % residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import parts4680 as PT

# ---------------------------------------------------------------------------
#  Cell envelope
# ---------------------------------------------------------------------------
CAN_OD = 4.630          # cm, [M][S2] 46.3 +/- 0.25 mm
CAN_H = 8.020           # cm, [M][S2] 80.2 +/- 0.3 mm
CAN_WALL = 0.04182      # cm, [D] NOT reported by [S1]; determined by the
                        #     mass, capacity and bore closures of
                        #     byd_triangulate (0.418 mm). Superseded: 0.50 mm
                        #     carried over from the NMC cell.
CELL_MASS = 332.0       # g, [M][S1] 335 g, [S2] 0.335 kg -- midpoint
                        #    of the reported spread

# ---------------------------------------------------------------------------
#  Jelly roll, from the unwound electrodes
# ---------------------------------------------------------------------------
CATH_H = 6.90           # cm, [M][S1] LFP cathode height 69 mm
CATH_L = 537.0          # cm, [M][S1] LFP cathode length 5370 mm
AN_H = 7.10             # cm, [M][S1] graphite anode height 71 mm
AN_L = 551.3            # cm, [M][S1] graphite anode length 5513 mm

# Layer stack. [S1] reports the anode's SINGLE-SIDED coating directly, which
# is the reliable one; the "active material thickness" figures are quoted
# without saying whether they are per side or through the sheet, so they are
# treated as through-sheet here and the consistency check below is what tests
# that reading.
T_AN_COAT_1S = 54.5e-4  # cm, [M][S1] 54.5 um single-sided
T_CU = 10.0e-4          # cm, [M][S1] ~10 um
T_AL = 10.0e-4          # cm, [M][S1] ~10 um
T_CATH_SHEET = 150.0e-4  # cm, [M][S1] "~150 um active", read as through-sheet
T_AN_SHEET = 125.0e-4   # cm, [M][S1] "~125 um active", read as through-sheet
T_SEP = 13.73e-4        # cm, [D] determined by byd_triangulate; [S1] gives
                        #     no thickness. Superseded: 12 um [E].

# Core: a plastic mandrel, not the empty void the Tesla cell has
MANDREL_HEX_SIDE = 0.250   # cm, [M][S1] hexagonal cavity, 2.5 mm side
MANDREL_WALL = 0.066       # cm, [M][S1] ~0.66 mm
MANDREL_H = 0.71           # cm, [M][S1] "7.1 mm tall" -- a stub at the core,
                           #     not a full-height former. Thermally minor.

# ---------------------------------------------------------------------------
#  Collectors and insulators
# ---------------------------------------------------------------------------
DISK_T = 0.031          # cm, [M][S1] both disks 0.31 mm (NMC cell: 0.20)
PLASTIC_T = 0.029       # cm, [M][S1] insulator disks 0.29 mm
                        #     (NMC cell: a single 0.65 mm disk at the cap)

# [S1] does not dimension the disks. They cannot exceed the can bore and
# cannot be smaller than the tab circle they serve, whose outermost tab sits
# at 20.0 mm radius on the anode and 18.3 mm on the cathode.
ANODE_DISK_OD = 4.40    # cm, [E] bounded below by the 20.0 mm outer tab radius
CATHODE_DISK_OD = 4.00  # cm, [E] bounded below by the 18.3 mm outer tab radius

# THE STRUCTURAL DIFFERENCE FROM THE TESLA CELL. [S1]: four aluminium cathode
# tabs at ~90 deg, radii 0.4 / 13.1 / 16.0 / 18.3 mm; three nickel anode tabs
# at radii 9.0 / 15.6 / 20.0 mm, one of which is PET-covered and disconnected.
# This is a multi-tab cell, not a tabless one, so the roll reaches its
# collector disk through discrete welds rather than a continuous annulus.
CATHODE_TAB_R = (0.04, 1.31, 1.60, 1.83)   # cm, [M][S1]
ANODE_TAB_R = (0.90, 1.56, 2.00)           # cm, [M][S1]
ANODE_TAB_LIVE = 2                          # [M][S1] one of three is dead
TAB_W = 0.40            # cm, [E] tab width. Not dimensioned by [S1]; 4 mm is
                        #     typical. Enters only through the weld area, and
                        #     weld_sensitivity() shows what it is worth.

RIM_NOTCH_FRACTION = PT.RIM_NOTCH_FRACTION   # [E] carried from paper 1

MAT = PT.MAT            # same handbook values; both cans are plated steel


# ---------------------------------------------------------------------------
#  Derived geometry
# ---------------------------------------------------------------------------
def winding_period() -> float:
    """Full period of the wound stack, cm. [D] from the layer thicknesses."""
    return T_CATH_SHEET + T_AN_SHEET + 2.0 * T_SEP


def core_radius() -> float:
    """Outer radius of the plastic mandrel, cm. [D] from the hexagon."""
    return MANDREL_HEX_SIDE + MANDREL_WALL      # across-corners / 2 + wall


def roll_radius_from_length(L=CATH_L, p=None) -> float:
    """Roll outer radius implied by the unwound length. [D]

    A spiral of constant pitch p from r_core to r_out has total length
    L = pi (r_out^2 - r_core^2) / p. That is exact for an Archimedean spiral
    and good to better than a layer thickness for a wound cell.
    """
    p = winding_period() if p is None else p
    r_core = core_radius()
    return float(np.sqrt(r_core ** 2 + L * p / np.pi))


def period_from_roll_radius(r_out, L=CATH_L) -> float:
    """The inverse: what pitch does a given roll radius imply? [D]"""
    r_core = core_radius()
    return float(np.pi * (r_out ** 2 - r_core ** 2) / L)


def turns(r_out=None) -> float:
    r_out = roll_radius_from_length() if r_out is None else r_out
    return float((r_out - core_radius()) / winding_period())


def roll_radius_max() -> float:
    """The largest roll that fits inside the can bore. [D]"""
    return CAN_OD / 2.0 - CAN_WALL


def consistency() -> dict:
    """Does the electrode length fit inside the can? [D]

    The check paper 1 runs on the NMC cell. Here it is sharper, because the
    roll diameter is NOT reported and this is how we recover it.
    """
    p = winding_period()
    r_len = roll_radius_from_length()
    r_max = roll_radius_max()
    p_fit = period_from_roll_radius(r_max)
    return {
        "winding period from layers [um]": p * 1e4,
        "roll radius implied by length [cm]": r_len,
        "roll radius that fits the bore [cm]": r_max,
        "overshoot [%]": (r_len / r_max - 1.0) * 100.0,
        "period implied by a bore-filling roll [um]": p_fit * 1e4,
        "turns, from layers": turns(r_len),
        "turns, bore-filling": (r_max - core_radius()) / p_fit,
    }


def areal_capacity_check() -> dict:
    """[S1]'s areal capacity against its reported cell capacity. [D]"""
    A_cath = 2.0 * CATH_H * CATH_L        # cm^2, both sides
    A_an = 2.0 * AN_H * AN_L
    return {
        "cathode area, both sides [cm^2]": A_cath,
        "anode area, both sides [cm^2]": A_an,
        "capacity from 2.09 mAh/cm^2 [Ah]": 2.09 * A_cath / 1000.0,
        "capacity reported by [S1] [Ah]": 15.41,
        "anode capacity from 2.34 mAh/cm^2 [Ah]": 2.34 * A_an / 1000.0,
        "N/P from these areas": (2.34 * A_an) / (2.09 * A_cath),
    }


# ---------------------------------------------------------------------------
#  The thermal network
# ---------------------------------------------------------------------------
def tab_weld_area(which="anode") -> float:
    """Total weld footprint between roll and collector disk, cm^2. [D][E]

    A tabless cell welds the whole foil edge to the disk. A multi-tab cell
    welds a handful of strips. This is the single geometric quantity that
    changes most between the two constructions, and it is why the BYD cell's
    roll-to-disk conductance is not paper 1's.
    """
    if which == "anode":
        n = ANODE_TAB_LIVE
    else:
        n = len(CATHODE_TAB_R)
    return n * TAB_W ** 2       # a square weld per tab, side TAB_W


def build(roll_r=None, f_weld=0.5) -> PT.Parts:
    """The BYD cell's lumped network, in paper 1's Parts form.

    roll_r defaults to the bore-filling radius, which is the physically
    admissible end of the consistency check.
    """
    r_out = roll_radius_max() if roll_r is None else roll_r
    r_can = CAN_OD / 2.0
    r_core = core_radius()
    r_neg, r_pos = ANODE_DISK_OD / 2.0, CATHODE_DISK_OD / 2.0
    A_neg, A_pos = np.pi * r_neg ** 2, np.pi * r_pos ** 2

    # heat capacities
    C_disk_neg = MAT["Cu"]["rho"] * MAT["Cu"]["cp"] * A_neg * DISK_T
    C_disk_pos = MAT["Al"]["rho"] * MAT["Al"]["cp"] * A_pos * DISK_T
    # TWO insulator disks on this cell, not one
    C_plastic = (2.0 * MAT["plastic"]["rho"] * MAT["plastic"]["cp"]
                 * np.pi * r_can ** 2 * PLASTIC_T)
    V_can = (2 * np.pi * r_can * CAN_H * CAN_WALL
             + 2 * np.pi * r_can ** 2 * CAN_WALL)
    C_can = MAT["steel"]["rho"] * MAT["steel"]["cp"] * V_can

    # roll -> disk, THROUGH THE TABS rather than a full annulus
    G_roll_disk_neg = (MAT["Cu"]["lam"] * tab_weld_area("anode") * f_weld
                       / (DISK_T / 2))
    G_roll_disk_pos = (MAT["Al"]["lam"] * tab_weld_area("cathode") * f_weld
                       / (DISK_T / 2))

    # disk -> can: base is a rim weld, cap is conduction through plastic
    L_ring = 2 * np.pi * r_neg * (1.0 - RIM_NOTCH_FRACTION)
    G_disk_neg_can = MAT["Cu"]["lam"] * L_ring * DISK_T / 0.1
    G_disk_pos_can = MAT["plastic"]["lam"] * np.pi * r_pos ** 2 / PLASTIC_T

    A_wall = 2 * np.pi * r_can * CAN_WALL
    G_can_axial = MAT["steel"]["lam"] * A_wall / CAN_H

    # electrical collection: tabs in parallel, each a radial foil path
    R_collect = _collection_resistance(r_out)

    return PT.Parts(
        C_disk_neg=C_disk_neg, C_disk_pos=C_disk_pos, C_can=C_can,
        C_plastic=C_plastic,
        G_roll_to_disk_neg=G_roll_disk_neg,
        G_roll_to_disk_pos=G_roll_disk_pos,
        G_disk_neg_to_can=G_disk_neg_can,
        G_disk_pos_to_can=G_disk_pos_can,
        G_can_axial=G_can_axial,
        R_collect_mean=R_collect,
        A_can_side=2 * np.pi * r_can * CAN_H,
        A_can_end=np.pi * r_can ** 2,
    )


def _collection_resistance(r_out) -> float:
    """Mean foil resistance from the winding to the nearest tab. [D]

    Tabless: every point of the foil edge meets the disk, so the path is one
    foil width. Multi-tab: current runs ALONG the foil to the nearest tab, a
    path that scales with the arc between tabs. This is the term that makes a
    multi-tab cell's collection resistance an order above a tabless one.
    """
    Rs_cu = 1.0 / (MAT["Cu"]["sigma"] * T_CU)      # ohm per square
    Rs_al = 1.0 / (MAT["Al"]["sigma"] * T_AL)
    # mean distance to the nearest of n tabs placed around the roll
    def leg(n_tab, height):
        # current collects over the electrode height into a tab; mean path is
        # a quarter of the spacing between neighbouring tabs
        return Rs_cu * 0.0 + 0.25 * (2 * np.pi * r_out / n_tab) / height
    R_neg = Rs_cu * leg(ANODE_TAB_LIVE, AN_H)
    R_pos = Rs_al * leg(len(CATHODE_TAB_R), CATH_H)
    return float(R_neg + R_pos)


def can_wall_resistance() -> float:
    """Axial electrical resistance of the can wall, ohm. [D]"""
    r = CAN_OD / 2
    A = 2 * np.pi * r * CAN_WALL
    return CAN_H / (MAT["steel"]["sigma"] * A)


# ---------------------------------------------------------------------------
#  Sensitivities on the two estimated numbers
# ---------------------------------------------------------------------------
def wall_sensitivity():
    global CAN_WALL
    keep = CAN_WALL
    out = []
    for t in (0.030, 0.040, 0.050, 0.060, 0.070):
        CAN_WALL = t
        p = build()
        out.append((t * 10, can_wall_resistance() * 1e3, p.G_can_axial))
    CAN_WALL = keep
    return out


def weld_sensitivity():
    return [(f, build(f_weld=f)) for f in (0.2, 0.35, 0.5, 0.75, 1.0)]


def main():
    print("=" * 72)
    print("BYD FC4680 geometry, from Liu et al. 2025")
    print("=" * 72)
    for k, v in consistency().items():
        print(f"  {k:42s} {v:10.3f}")
    print()
    for k, v in areal_capacity_check().items():
        print(f"  {k:42s} {v:10.3f}")

    print()
    print("=" * 72)
    print("Thermal network: BYD vs paper 1's cell")
    print("=" * 72)
    b = build()
    n = PT.build_parts(disks="measured")
    print(f"{'quantity':38} {'paper 1':>11} {'BYD':>11} {'ratio':>7}")
    print("-" * 72)
    for key in ("C_disk_neg", "C_disk_pos", "C_can", "C_plastic",
                "G_roll_to_disk_neg", "G_roll_to_disk_pos",
                "G_disk_neg_to_can", "G_disk_pos_to_can", "G_can_axial"):
        a, c = getattr(n, key), getattr(b, key)
        print(f"{key:38} {a:11.4g} {c:11.4g} {c/a:7.2f}")
    print(f"{'R_collect_mean [mohm]':38} {n.R_collect_mean*1e3:11.4g} "
          f"{b.R_collect_mean*1e3:11.4g} {b.R_collect_mean/n.R_collect_mean:7.2f}")
    print(f"{'can wall R [mohm]':38} {PT.can_wall_resistance()*1e3:11.4g} "
          f"{can_wall_resistance()*1e3:11.4g}")

    def ser(a, b_):
        return 1.0 / (1.0 / a + 1.0 / b_)
    print()
    print("End-to-can conductances (series), and the asymmetry that drives")
    print("the axial gradient:")
    for name, p in (("paper 1 (tabless)", n), ("BYD (multi-tab)", b)):
        sb = ser(p.G_roll_to_disk_neg, p.G_disk_neg_to_can)
        sc = ser(p.G_roll_to_disk_pos, p.G_disk_pos_to_can)
        print(f"  {name:20s} base {sb:8.3f}  cap {sc:8.4f}  "
              f"ratio {sb/sc:6.1f}")

    print()
    print("Sensitivity to the can wall thickness, the main [E]:")
    print(f"  {'wall [mm]':>10} {'R_can [mohm]':>14} {'G_can_axial [W/K]':>19}")
    for t, R, G in wall_sensitivity():
        print(f"  {t:10.1f} {R:14.4f} {G:19.5f}")

    print()
    print("Sensitivity to the weld quality factor, the other [E]:")
    print(f"  {'f_weld':>7} {'G_roll_disk_neg':>17} {'series base':>13} "
          f"{'asymmetry':>10}")
    for f, p in weld_sensitivity():
        sb = ser(p.G_roll_to_disk_neg, p.G_disk_neg_to_can)
        sc = ser(p.G_roll_to_disk_pos, p.G_disk_pos_to_can)
        print(f"  {f:7.2f} {p.G_roll_to_disk_neg:17.3f} {sb:13.3f} "
              f"{sb/sc:10.1f}")


if __name__ == "__main__":
    main()

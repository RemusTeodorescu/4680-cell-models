"""Display conventions for every generated table, triangulated from the sources.

    python housestyle.py          # show the rules and a worked sample

WHY THIS EXISTS
---------------
The solver works in CGS, because Gu and Wang's formulation does and the
discretisation inherits it. That is an implementation choice and it has no
business reaching a reader: both papers this work builds on report in SI, and a
battery audience reads thicknesses in micrometres and concentrations in
mol per cubic metre. Printing "0.007 cm" where a reader expects "70 um" is not
a different convention, it is an obstacle.

It also has to be ONE module. Before this file there were three formatting
paths -- `texval` with its own magnitude thresholds, `UNIT_TEX` with its own
unit spellings, and a hand-typed comparison table with per-row `:.4f` choices
-- so the same quantity could appear three ways in one document. A house style
that lives in three places is not a house style.

WHAT THE SOURCES DO
-------------------
Ank et al. (2023), the Tesla 4680 teardown:
  * SI throughout. Cell dimensions in mm ("46 mm", "80 mm", "0.5 mm"),
    electrode lengths in mm ("3403 mm"), particle sizes in um ("3 um to
    16 um"), volumes in cm^3 ("125.6 cm3").
  * Two to four significant figures. Cell dimensions integer or one decimal;
    derived volumes one decimal.
  * Uncertainty as +/- where it is known ("25 +/- 0.5 C").

Liu et al. (2025), the BYD FC4680 teardown:
  * SI throughout. Thicknesses in um ("approximately 150 um"), dimensions in
    mm, areal quantities per cm^2, energy density to one decimal
    ("151 Wh/kg", "374.6 Wh/l").
  * Two to four significant figures; scatter as relative standard deviation.

Neither carries a provenance column. That is ours, and it is the thing this
paper adds, so it gets a column of its own rather than a footnote.

THE RESULTING RULES
-------------------
1. SI for presentation, CGS for computation. The conversion lives here and
   nowhere else.
2. The unit is chosen so the number reads between about 0.1 and 10000 --
   um for coatings and particles, mm for cell hardware, m for winding lengths.
3. Three significant figures, and THREE DIGITS PRINTED. Trailing zeros are not
   stripped: 2.5 V and 2.0 V print as "2.50" and "2.00", because a column in
   which one cell shows "2.5" and the other "2" is a column a reader has to
   decode before comparing. A handful of quantities that are exact by
   definition or known to more figures are listed in SIGFIG.
4. Below 10^-3 and at or above 10^5, proper scientific notation -- never
   e-notation, and never a fixed-point number with six leading zeros.
5. Between 10^3 and 10^5 the value prints as an integer, with a thin space
   grouping five digits or more, following siunitx.
6. One unit per row, in its own column, applying to both cells. If the two
   cells would naturally want different units, the row uses the same one for
   both -- a table whose columns are not directly comparable defeats its own
   purpose. dual_cells() enforces this rather than trusting it.
7. Units are set with siunitx, which the manuscript already loads. Hand-rolled
   "$\\mu$m" and "W\\,cm$^{-1}$" spellings drifted apart across three tables;
   a macro cannot.
"""

from __future__ import annotations

from math import floor, log10

# CGS unit as the registries store it -> (SI unit for display, factor)
# factor converts the stored CGS number to the display unit.
SI = {
    "cm":                      ("um",        1e4),      # default; see OVERRIDE
    "um":                      ("um",        1.0),
    "mm":                      ("mm",        1.0),
    "m":                       ("m",         1.0),
    "mol/cm^3":                ("mol/m^3",   1e6),
    "cm^2/s":                  ("m^2/s",     1e-4),
    "S/cm":                    ("S/m",       1e2),
    "W/cm/K":                  ("W/m/K",     1e2),
    "W/cm^2/K":                ("W/m^2/K",   1e4),
    "g/cm^3":                  ("kg/m^3",    1e3),
    "J/g/K":                   ("J/kg/K",    1e3),
    "J/mol":                   ("kJ/mol",    1e-3),
    "ohm":                     ("mohm",      1e3),
    # A/cm^2 (cm^3/mol)^1.5 -> A/m^2 (m^3/mol)^1.5 -> A m^2.5 mol^-1.5.
    # The reduced form is the same unit with the powers collected, and it is
    # how Chen et al. write it; the unreduced string is 30 % of a column.
    "(A/cm^2)(cm^3/mol)^1.5":  ("A m^2.5/mol^1.5", 1e-5),
    "V":                       ("V",         1.0),
    "V/K":                     ("mV/K",      1e3),
    "K":                       ("K",         1.0),
    "Ah":                      ("Ah",        1.0),
    "g":                       ("g",         1.0),
    "W":                       ("W",         1.0),
    "%":                       ("%",         1.0),
    "K/W":                     ("K/W",       1.0),
    "-":                       ("-",         1.0),
}

# Rows whose natural display unit is not the default for their stored unit.
# Keyed by the registry field name. Lengths especially: a coating wants um, a
# can wants mm, a winding wants m, and forcing one on all three is what makes
# a table unreadable.
OVERRIDE = {
    "winding length H":        ("m",  1e-2),
    "electrode height":        ("mm", 10.0),
    "jelly-roll height":       ("mm", 10.0),
    "model depth":             ("mm", 10.0),
    "jelly-roll core radius":  ("mm", 10.0),
    "jelly-roll outer radius": ("mm", 10.0),
    "can wall thickness":      ("mm", 10.0),
    "full winding period":     ("um", 1e4),
}

# Quantities that are not three-significant-figure measurements.
#   T0, Ta     a reference temperature quoted to the hundredth of a kelvin;
#              "298 K" would lose the convention, not just precision
#   alpha      exactly one half by assumption of symmetric transfer
#   dlnf       exactly zero (the ideal-solution assumption)
SIGFIG = {"T0": 5, "Ta": 5, "alpha_a": 1, "alpha_c": 1, "dlnf_dlnce": 1}

# LaTeX for each display unit. siunitx, so spacing and upright/italic are
# decided once by the package rather than per table by hand.
TEX = {
    "um": r"\si{\micro\metre}", "mm": r"\si{\milli\metre}",
    "m": r"\si{\metre}", "K": r"\si{\kelvin}", "V": r"\si{\volt}",
    "mV/K": r"\si{\milli\volt\per\kelvin}",
    "mol/m^3": r"\si{\mole\per\cubic\metre}",
    "m^2/s": r"\si{\square\metre\per\second}",
    "S/m": r"\si{\siemens\per\metre}",
    "W/m/K": r"\si{\watt\per\metre\per\kelvin}",
    "W/m^2/K": r"\si{\watt\per\square\metre\per\kelvin}",
    "kg/m^3": r"\si{\kilo\gram\per\cubic\metre}",
    "J/kg/K": r"\si{\joule\per\kilo\gram\per\kelvin}",
    "kJ/mol": r"\si{\kilo\joule\per\mole}",
    "mohm": r"\si{\milli\ohm}",
    "Ah": r"\si{\ampere\hour}", "g": r"\si{\gram}", "W": r"\si{\watt}",
    "K/W": r"\si{\kelvin\per\watt}", "%": r"\si{\percent}",
    "A m^2.5/mol^1.5":
        r"\si{\ampere\metre\tothe{2.5}\per\mole\tothe{1.5}}",
    "-": "---",
}


def display_unit(field, stored_unit):
    """[D] The SI unit this row is printed in, and the conversion factor."""
    if field in OVERRIDE:
        return OVERRIDE[field]
    if stored_unit not in SI:
        raise KeyError(f"housestyle has no display rule for unit "
                       f"{stored_unit!r} (field {field!r}); add one to SI "
                       f"rather than letting a CGS number reach the page")
    return SI[stored_unit]


def unit_tex(field, stored_unit):
    """[D] The siunitx spelling of the unit this row prints in."""
    u, _ = display_unit(field, stored_unit)
    return TEX.get(u, u)


def fmt(value, sig=3):
    """[D] A number as a reader wants it. Rule 3, 4 and 5 above.

    Trailing zeros are kept deliberately -- see rule 3. The negative sign is a
    maths minus, not a hyphen, because a hyphen in a numeric column is a
    typographic error that also reads as a range separator.
    """
    if value is None:
        return "---"
    v = float(value)
    if v == 0:
        return "0"
    a, neg = abs(v), v < 0

    if a >= 1e5 or a < 1e-3:                       # rule 4
        e = int(floor(log10(a)))
        m = a / 10.0 ** e
        ms = f"{m:.{max(0, sig - 1)}f}"
        if float(ms) >= 10.0:                      # 9.996 -> 10.0, renormalise
            e += 1
            ms = f"{a / 10.0 ** e:.{max(0, sig - 1)}f}"
        return rf"${'-' if neg else ''}{ms}\times10^{{{e}}}$"

    if a >= 1e3:                                   # rule 5
        s = f"{a:,.0f}".replace(",", "\\,")
        return ("$-$" if neg else "") + s

    d = max(0, sig - 1 - int(floor(log10(a))))     # rule 3
    return ("$-$" if neg else "") + f"{a:.{d}f}"


def convert(value, field, stored_unit):
    """[D] Value in its display unit, and the siunitx unit, for a table cell."""
    unit, factor = display_unit(field, stored_unit)
    v = None if value is None else float(value) * factor
    return v, TEX.get(unit, unit)


def cell(value, field, stored_unit, sig=None):
    """[D] One formatted table cell: converted, rounded and typeset."""
    v, _ = convert(value, field, stored_unit)
    return fmt(v, SIGFIG.get(field, 3) if sig is None else sig)


def dual_cells(row_a, row_b):
    """[D] Both cells of one two-column row, plus the unit they share.

    Rule 6 is enforced here. If the two registries ever store the same
    parameter in different units the columns would silently stop being
    comparable, which is the one failure mode a side-by-side table cannot
    survive, so it raises instead.
    """
    ua, fa = display_unit(row_a["field"], row_a["unit"])
    ub, fb = display_unit(row_b["field"], row_b["unit"])
    if ua != ub:
        raise ValueError(
            f"{row_a['group']}.{row_a['field']}: the two cells would print in "
            f"different units ({ua} vs {ub}); the columns must be comparable")
    sig = SIGFIG.get(row_a["field"], 3)
    va = None if row_a["value"] is None else row_a["value"] * fa
    vb = None if row_b["value"] is None else row_b["value"] * fb
    return fmt(va, sig), fmt(vb, sig), TEX.get(ua, ua)


def main():
    import byd4680_params as BYD
    import gw4680 as G
    nmc = {(r["group"], r["field"]): r for r in G.parameter_rows()}
    lfp = {(r["group"], r["field"]): r for r in BYD.parameter_rows()}

    print("HOUSE STYLE -- CGS stored, SI displayed, one system for both cells")
    print(f"\n  {'parameter':26s} {'stored (CGS)':>14s}   "
          f"{'NMC':>12s} {'LFP':>12s}   unit")
    show = [("format", "winding length H"), ("format", "model depth"),
            ("format", "full winding period"), ("cathode", "L"),
            ("cathode", "rs"), ("cathode", "cs_max"), ("cathode", "Ds"),
            ("cathode", "sigma"), ("cathode", "lam"), ("cathode", "rho"),
            ("cathode", "cp"), ("cathode", "Eact_Ds"), ("cathode", "theta0"),
            ("anode", "dUdT"), ("anode", "alpha_a"), ("cell", "T0"),
            ("cell", "V_cut"), ("Cu", "sigma"),
            ("format", "hardware series resistance")]
    for k in show:
        if k not in lfp or k not in nmc:
            continue
        a, b = nmc[k], lfp[k]
        va, vb, u = dual_cells(a, b)
        raw = a["value"]
        print(f"  {a['field']:26s} {raw if raw is None else f'{raw:14.4g}'}   "
              f"{va:>12s} {vb:>12s}   {u}")

    print("\n  every row above prints three digits in both columns and shares "
          "one unit;\n  that is the whole of the convention.")

    # the cases the rules exist for, shown explicitly
    print("\n  edge cases")
    for v, why in [(0.0, "exactly zero"), (-2.754e-5 * 1e3, "negative"),
                   (5.96e5 * 1e2, "large, scientific"),
                   (6.873e-13 * 1e-4, "small, scientific"),
                   (0.030555 * 1e6, "five digits, thin-space grouped"),
                   (2.0, "trailing zeros kept"),
                   (0.0016261, "small but fixed point")]:
        print(f"    {why:34s} {fmt(v)}")


if __name__ == "__main__":
    main()

"""Metadata organised into the Battery Data Genome's three tiers.

    python bdg.py                  # print the tiers
    python bdg.py --csv out/       # write metadata_{primary,secondary,tertiary}.csv

WHY TIERS AND NOT A FORMAT
--------------------------
Ward et al., "Principles of the Battery Data Genome", Joule 6, 2253-2271 (2022),
DOI 10.1016/j.joule.2022.08.008, is a Perspective. It proposes principles and an
organising structure; it does not specify a file format, and says so:
"Establishing what information is needed and its format will require not only
extensive community discussion, but also innovations to enable broadest possible
participation."

So there is nothing to convert to, and claiming conformance with a format that
does not exist would be worse than staying quiet. What the paper DOES give is a
three-tier metadata model, and that is adoptable today:

  PRIMARY    "the minimal mandatory information ... data owners and generators,
             a unique identifier for each cell, experimental objective, the
             nature of the device/s that were tested including chemistry, and
             specific tests"
  SECONDARY  "more comprehensive, with significant experimental details
             including cell and electrode design, commercial-cell
             identification codes, the manner of attachments to test equipment"
  TERTIARY   "highly specialized, nonroutine information ... for example,
             material synthesis protocols and SIMULATION MODEL PARAMETERS"

That last clause is why this module exists. A DFN parameter set is not adjacent
to their taxonomy; it is a named case in it. Sorting the data into the tiers is
a real alignment rather than a gesture, and it costs an hour.

THE QUALITY PRINCIPLE, AND WHAT WE CONTRIBUTE TO IT
---------------------------------------------------
Their third operating principle is that "all performance and metadata must be
cleaned, curated, and quality controlled before submission to a data hub ...
the foundation of ensuring scientific robustness". For cycler time series they
point at existing tools. For SIMULATION PARAMETERS they state the requirement
and leave the method open.

The closure audits are an answer to that. Testing a parameter set against
measurements never used to build it -- cell mass, cell capacity, unwound
electrode length -- and reporting the residual is quality control for tertiary
metadata. It is also discriminating rather than ceremonial: the same audit
leaves +9.8 % on one cell and -2.3 % on another, which localises the first
cell's shortfall rather than merely passing.

ON FORMATS THAT DO EXIST
------------------------
The LF Energy Battery Data Alliance's Battery Data Format (BDF, Apache-2.0)
specifies CSV with RDF/OWL-backed column headers for CYCLER TIME SERIES: its
required quantities are current, test time and voltage. It has no provision for
a parameter set, and its own roadmap lists metadata as "an immediate next step".
It is therefore the right target for the lab campaign's raw traces and the wrong
one for this dataset. BattINFO (EMMO-based) is the ontology to name parameters
against when that work is done.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import byd_parts as BP
import byd_stack as BS
import gw4680 as G
import parts4680 as PT
import stack4680 as S4

CITATION = ("Ward et al., Principles of the Battery Data Genome, "
            "Joule 6, 2253-2271 (2022), doi:10.1016/j.joule.2022.08.008")

# ---------------------------------------------------------------------------
#  PRIMARY -- the minimal mandatory record, one row per cell
# ---------------------------------------------------------------------------
def primary():
    m = G.derived_metrics(G.params_4680(C_rate=1.0))
    return [
        dict(cell_id="NMC-4680-ANK2023",
             manufacturer="Tesla (as characterised by Ank et al. 2023)",
             model="4680", form_factor="cylindrical 46 x 80 mm",
             positive_electrode="NMC811", negative_electrode="graphite",
             nominal_capacity_Ah=f"{m['cell capacity [Ah]']:.3f}",
             nominal_voltage_V="3.70",
             mass_g="355.9",
             data_generator="Teodorescu, Aalborg University",
             data_source="Ank et al. 2023, JES 170 120536 (CC BY 4.0)",
             objective=("identify a DFN + thermal parameter set and audit it "
                        "against independently measured quantities"),
             tests="none performed here; parameters identified from literature"),
        dict(cell_id="LFP-FC4680-LIU2025",
             manufacturer="BYD", model="FC4680",
             form_factor="cylindrical 46.3 x 80.2 mm",
             positive_electrode="LFP", negative_electrode="graphite",
             nominal_capacity_Ah=f"{BS.Q_CELL:.2f}",
             nominal_voltage_V="3.20",
             mass_g=f"{BS.M_CELL:.1f}",
             data_generator="Teodorescu, Aalborg University",
             data_source="Liu et al. 2025, JES 172 110536 (CC BY 4.0)",
             objective=("identify geometry and thermal network; validate the "
                        "axial gradient by a two-configuration experiment"),
             tests="axial-gradient protocol specified; not yet performed"),
    ]


# ---------------------------------------------------------------------------
#  SECONDARY -- cell and electrode design
# ---------------------------------------------------------------------------
def secondary():
    m = G.derived_metrics(G.params_4680(C_rate=1.0))
    rows = []

    def add(cell, q, v, u, prov):
        rows.append(dict(cell_id=cell, quantity=q,
                         value=f"{v:.6g}" if isinstance(v, float) else v,
                         unit=u, provenance=prov))

    N, L = "NMC-4680-ANK2023", "LFP-FC4680-LIU2025"
    for c, od, h, wall in ((N, PT.CAN_OD * 10, PT.CAN_H * 10, PT.CAN_WALL * 10),
                           (L, BP.CAN_OD * 10, BP.CAN_H * 10,
                            BP.CAN_WALL * 10)):
        add(c, "can_outer_diameter", od, "mm", "M" if c == N else "M")
        add(c, "can_height", h, "mm", "M")
        add(c, "can_wall_thickness", wall, "mm", "M" if c == N else "E")

    add(N, "cathode_length", G.WIND_LENGTH * 10, "mm", "M")
    add(N, "cathode_coating_per_side", G.T_CATHODE * 1e4, "um", "M")
    add(N, "anode_coating_per_side", G.T_ANODE * 1e4, "um", "E")
    add(N, "cathode_porosity", 0.335, "-", "L")
    add(N, "anode_porosity", 0.25, "-", "L")
    add(N, "collector_disk_thickness", PT.DISK_T * 10, "mm", "M")
    add(N, "insulator_thickness", PT.PLASTIC_T * 10, "mm", "M")
    add(N, "insulator_count", 1, "-", "M")
    add(N, "current_collection", "tabless", "-", "M")
    add(N, "core", "5 mm void, no mandrel", "-", "M")
    add(N, "N_P_ratio", m["N/P ratio [-]"], "-", "D")

    add(L, "cathode_height", BS.CATH_H * 10, "mm", "M")
    add(L, "cathode_length", BS.CATH_L * 10, "mm", "M")
    add(L, "anode_height", BS.AN_H * 10, "mm", "M")
    add(L, "anode_length", BS.AN_L * 10, "mm", "M")
    add(L, "cathode_coating_per_side", BS.T_CATH_1S * 1e4, "um", "D")
    add(L, "anode_coating_per_side", BS.T_AN_1S * 1e4, "um", "M")
    add(L, "foil_cu_thickness", BS.T_CU * 1e4, "um", "M")
    add(L, "foil_al_thickness", BS.T_AL * 1e4, "um", "M")
    add(L, "separator_thickness", BS.T_SEP * 1e4, "um", "E")
    add(L, "cathode_porosity", BS.solve_porosity(), "-", "D from cell mass")
    add(L, "anode_porosity", BS.EPS_AN, "-", "E")
    add(L, "collector_disk_thickness", BP.DISK_T * 10, "mm", "M")
    add(L, "insulator_thickness", BP.PLASTIC_T * 10, "mm", "M")
    add(L, "insulator_count", 2, "-", "M")
    add(L, "current_collection", "multi-tab, 4 Al + 3 Ni", "-", "M")
    add(L, "core", "PP mandrel, 7.1 mm tall", "-", "M")
    add(L, "areal_capacity_cathode", BS.AREAL_CATH, "mAh/cm2", "M")
    add(L, "areal_capacity_anode", BS.AREAL_AN, "mAh/cm2", "M")
    add(L, "roll_outer_radius", BP.roll_radius_from_length(), "cm", "D")
    return rows


# ---------------------------------------------------------------------------
#  TERTIARY -- simulation model parameters, named as such by Ward et al.
# ---------------------------------------------------------------------------
def tertiary():
    rows = []
    # The registry calls it "group", not "section". An earlier version read
    # r["section"], which does not exist, so every one of these 78 rows
    # exported with an empty domain -- in the very file whose purpose is to
    # demonstrate metadata quality. Caught by counting blanks in the export
    # rather than by reading the code, which is the argument for checking
    # outputs rather than inputs.
    #
    # "field" is carried alongside the LaTeX symbol because a consumer of this
    # CSV needs a machine-readable parameter name; "$L_{\mathrm{Cu}}/2$" is
    # for a typesetter, not for a data pipeline.
    for r in G.parameter_rows():
        rows.append(dict(cell_id="NMC-4680-ANK2023",
                         domain=r.get("group", ""),
                         parameter=r.get("field", ""),
                         symbol=r.get("symbol", ""),
                         value=r.get("value", ""), unit=r.get("unit", ""),
                         provenance=r.get("tag", ""), note=r.get("note", "")))
    n, b = PT.build_parts(disks="measured"), BP.build()
    for cell, p in (("NMC-4680-ANK2023", n), ("LFP-FC4680-LIU2025", b)):
        for k, u in (("G_roll_to_disk_neg", "W/K"),
                     ("G_roll_to_disk_pos", "W/K"),
                     ("G_disk_neg_to_can", "W/K"),
                     ("G_disk_pos_to_can", "W/K"),
                     ("G_can_axial", "W/K"),
                     ("R_collect_mean", "ohm")):
            rows.append(dict(cell_id=cell, domain="thermal network",
                             parameter=k, symbol=k,
                             value=f"{getattr(p, k):.6g}", unit=u,
                             provenance="D", note="lumped component model"))
    return rows


# ---------------------------------------------------------------------------
#  QUALITY -- the closure audits, as quality control on tertiary metadata
# ---------------------------------------------------------------------------
def quality():
    c4 = S4.closure()
    inv = BS.inventory()
    accb = (sum(x["m_solid"] + x["m_ely"] for x in inv)
            + sum(BS.hardware().values()))
    cc = BS.capacity_consistency()
    jr = G.jelly_roll_consistency()
    con = BP.consistency()
    return [
        dict(audit="cell mass", cell_id="NMC-4680-ANK2023",
             model=f"{c4['accounted']:.1f} g",
             reference=f"{c4['measured']:.1f} g",
             residual_pct=f"{c4['residual_pct']:+.2f}",
             reference_source="Ank et al. 2023, weighed"),
        dict(audit="cell mass", cell_id="LFP-FC4680-LIU2025",
             model=f"{accb:.1f} g", reference=f"{BS.M_CELL:.1f} g",
             residual_pct=f"{(BS.M_CELL-accb)/BS.M_CELL*100:+.2f}",
             reference_source="Liu et al. 2025 / public listings"),
        dict(audit="cell capacity", cell_id="LFP-FC4680-LIU2025",
             model=f"{cc['Q_from_cathode']:.3f} Ah",
             reference=f"{cc['Q_measured']:.3f} Ah",
             residual_pct=f"{cc['err_pct']:+.2f}",
             reference_source="Liu et al. 2025, measured"),
        dict(audit="electrode length", cell_id="NMC-4680-ANK2023",
             model=f"{jr['implied winding length [cm]']:.1f} cm",
             reference=f"{jr['measured winding length [cm]']:.1f} cm",
             residual_pct=f"{jr['discrepancy [%]']:+.2f}",
             reference_source="Ank et al. 2023, unwound and measured"),
        dict(audit="roll radius", cell_id="LFP-FC4680-LIU2025",
             model=f"{con['roll radius implied by length [cm]']:.3f} cm",
             reference=f"{con['roll radius that fits the bore [cm]']:.3f} cm",
             residual_pct=f"{con['overshoot [%]']:+.2f}",
             reference_source="bore of the measured can"),
    ]


TIERS = {"metadata_primary": primary, "metadata_secondary": secondary,
         "metadata_tertiary": tertiary, "quality_audits": quality}


def write_csv(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in TIERS.items():
        rows = fn()
        keys = list(rows[0].keys())

        # Two checks, because this module's claim is that the export is
        # quality controlled, and an unchecked export is only an assertion.
        #
        # 1. Every row must carry the same fields. DictWriter takes its header
        #    from row 0 and would otherwise either drop columns silently or
        #    raise deep inside the write.
        odd = [i for i, r in enumerate(rows) if list(r.keys()) != keys]
        if odd:
            raise ValueError(
                f"{name}: rows {odd[:5]} have fields differing from row 0; "
                f"expected {keys}, got {list(rows[odd[0]].keys())}")

        # 2. A column that is empty for most rows is a bug, not a sparse
        #    field -- it is how the "domain" column shipped blank for 78 of
        #    90 tertiary rows. Warn rather than raise, since some fields are
        #    legitimately optional:
        #      note    free text, present only where there is something to say
        #      symbol  the LaTeX symbol used in the manuscript. 14 format-level
        #              quantities (winding length, electrode height, foil
        #              thicknesses) have no symbol in the paper's notation and
        #              are not going to be given invented ones to fill a
        #              column. They are fully identified by "parameter", which
        #              is the machine-readable name and is never blank.
        OPTIONAL = ("note", "symbol")
        for k in keys:
            blank = sum(1 for r in rows if not str(r.get(k, "")).strip())
            if blank and k not in OPTIONAL:
                print(f"  NOTE {name}.{k}: {blank}/{len(rows)} rows blank")

        p = outdir / f"{name}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        made.append(p.name)
    with open(outdir / "README_metadata.txt", "w") as fh:
        fh.write("Metadata organised into the three tiers proposed by:\n  "
                 + CITATION + "\n\n"
                 "  metadata_primary.csv    minimal mandatory record, per cell\n"
                 "  metadata_secondary.csv  cell and electrode design\n"
                 "  metadata_tertiary.csv   simulation model parameters\n"
                 "  quality_audits.csv      closure audits: quality control on\n"
                 "                          the tertiary tier, against measurements\n"
                 "                          never used to build the parameter set\n\n"
                 "Provenance tags: [M] measured  [L] literature  "
                 "[D] derived  [E] estimated\n")
    return made + ["README_metadata.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", metavar="DIR")
    a = ap.parse_args()
    if a.csv:
        for f in write_csv(Path(a.csv)):
            print("  wrote", f)
        return
    print("=" * 74)
    print("BATTERY DATA GENOME TIERS")
    print("=" * 74)
    print(" ", CITATION, "\n")
    for name, fn in TIERS.items():
        rows = fn()
        print(f"  {name:24s} {len(rows):4d} rows   "
              f"fields: {', '.join(list(rows[0].keys())[:4])} ...")
    print()
    print("QUALITY AUDITS — quality control on the tertiary tier")
    print(f"  {'audit':16s} {'cell':22s} {'model':>14s} {'reference':>14s} "
          f"{'resid':>8s}")
    for q in quality():
        print(f"  {q['audit']:16s} {q['cell_id']:22s} {q['model']:>14s} "
              f"{q['reference']:>14s} {q['residual_pct']:>7s} %")


if __name__ == "__main__":
    main()

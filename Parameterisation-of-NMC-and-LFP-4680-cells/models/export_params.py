"""Export both parameter sets in formats other people's code can read.

    python export_params.py                # write data/, then verify
    python export_params.py --no-verify    # write only

WHY THIS EXISTS
---------------
A parameter set released as Python modules is released to people who will run
our Python. That is not reuse, it is distribution. The modules stay -- they
carry the provenance, and the provenance is the paper's contribution -- but
the numbers also have to leave the repository in a form that a reader can
load into their own model without reading our code at all.

Three formats, because they answer three different questions.

  BPX JSON        "give me this cell in PyBaMM."  Battery Parameter eXchange
                  is the Faraday Institution's interchange standard, and
                  pybamm.ParameterValues.create_from_bpx reads it directly.
                  One line and the cell is running in somebody else's model.

  parameters CSV  "what are the numbers?"  A flat table, SI, one row per
                  parameter, with the provenance tag and its reason. Opens in
                  a spreadsheet. This is the only one of the three that
                  carries WHERE EACH NUMBER CAME FROM, which is the thing BPX
                  has no field for and the thing this work is actually about.

  parameters JSON the same records, for code that would rather not parse CSV.

WHAT BPX CANNOT CARRY, AND WHY THAT IS NOT A CRITICISM
------------------------------------------------------
BPX describes a DFN cell. It has no field for a provenance tag, none for the
closure checks, and none for the wound-cell thermal network -- the collector
disks, the plastic insulator, the can wall and the two end conductances that
produce the axial gradient this work is about. So the BPX files are the
electrochemical core and the lumped thermal properties, and they are complete
for what BPX is for. The rest is in the CSV and in the modules, and the BPX
header says so rather than leaving a reader to discover it.

THE LICENCE POSITION, WHICH IS NOT THE SAME FOR THE TWO FILES
-------------------------------------------------------------
The NMC file is CC BY 4.0. Its geometry is ours and its electrochemistry is
transferred from Chen et al. (2020), a published paper whose parameter values
are facts and which PyBaMM itself ships.

The LFP file is NOT. Its electrochemistry is transferred from About:Energy's
BPX parameterisation of an LFP/graphite 18650, which is CC-BY-SA 4.0. Our
position throughout has been that individual values are facts while their
file is a work, and the modules therefore transcribe values with [L] tags and
do not redistribute the file. Assembling those values back into a BPX file is
a different act, and pretending otherwise would be convenient rather than
honest: what comes out is close enough to their work in kind that share-alike
should be respected. So the LFP BPX file is offered under CC-BY-SA 4.0, with
attribution in its header, while every other artefact in the release stays
CC BY 4.0. A reader who wants the LFP geometry alone, unencumbered, has it in
the CSV, which contains no About:Energy material.

This is a judgement, not a legal opinion, and it is deliberately the
conservative one. It costs us nothing -- nobody is harmed by a share-alike
interchange file -- and the alternative costs an argument with the source of
half the parameter set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import housestyle as HS

F = 96485.33212
NPTS = 200          # samples per OCP table


# ---------------------------------------------------------------------------
#  BPX ASSEMBLY
# ---------------------------------------------------------------------------
def _electrode(layer, ocp_fn, dudt, Eact_i0, Eact_Ds, k_bpx, hi_override=None):
    """[D] One electrode block, CGS layer -> BPX's SI names.

    The one conversion worth stating: BPX's reaction rate constant is not our
    m_ref. BPX normalises its concentrations, so m_ref = k F / (cs_max
    sqrt(ce0)); going out we invert it. crosscheck_pybamm.verify_kinetics
    checks the round trip against PyBaMM's own exchange-current function.
    """
    lo = min(layer.theta0, layer.theta_min if layer.theta_min else layer.theta0,
             layer.theta_max if layer.theta_max < 1.0 else layer.theta0)
    hi = max(layer.theta0, layer.theta_min,
             layer.theta_max if layer.theta_max < 1.0 else layer.theta0)
    if hi_override is not None:
        hi = hi_override
    x = np.linspace(lo, hi, NPTS)
    eps_s = layer.eps_s_explicit
    a_s = 3.0 * eps_s / (layer.rs * 1e-2)            # m^-1
    block = {
        "Minimum stoichiometry": round(float(lo), 7),
        "Maximum stoichiometry": round(float(hi), 7),
        "Maximum concentration [mol.m-3]": float(layer.cs_max * 1e6),
        "Particle radius [m]": float(layer.rs * 1e-2),
        "Surface area per unit volume [m-1]": float(a_s),
        "Diffusivity [m2.s-1]": float(layer.Ds * 1e-4),
        "Diffusivity activation energy [J.mol-1]": float(Eact_Ds),
        "OCP [V]": {"x": [float(v) for v in x],
                    "y": [float(v) for v in np.asarray(ocp_fn(x), float)]},
        "Entropic change coefficient [V.K-1]": float(dudt),
        "Reaction rate constant [mol.m-2.s-1]": float(k_bpx),
        "Reaction rate constant activation energy [J.mol-1]": float(Eact_i0),
        "Thickness [m]": float(layer.L * 1e-2),
        "Porosity": float(layer.eps_e),
        "Transport efficiency": round(float(layer.eps_e ** 1.5), 6),
        # EFFECTIVE solid conductivity. The solver uses sigma_eff =
        # eps_s * sigma (Gu & Wang Eq. 21), and PyBaMM's BPX import applies no
        # electrode Bruggeman factor, so exporting sigma itself gave PyBaMM
        # users a more conductive electrode than ours (8 mV at the end of a
        # 10 s 2C pulse on the NMC cell; paper 1.1). Fixed 27 Sep 2026.
        "Conductivity [S.m-1]": float(layer.sigma_eff * 1e2),
    }
    return block


# Nyman et al. (2008), in BPX's own variable convention (x is c_e in mol/m^3).
# These are not a re-fit: evaluated at 1000 mol/m^3 they give 0.9487 S/m and
# 1.7694e-10 m^2/s, which are the numbers gw4680's own correlations return, so
# the exported file and the solver share one electrolyte.
NYMAN_KAPPA = ("0.1297 * (x / 1000) ** 3 - 2.51 * (x / 1000) ** 1.5 "
               "+ 3.329 * (x / 1000)")
NYMAN_DE = ("8.794e-11 * (x / 1000) ** 2 - 3.972e-10 * (x / 1000) "
            "+ 4.862e-10")


def _stack_thermal(p):
    """[D] Jelly-roll density and specific heat, as a thickness-weighted
    average over the repeating unit.

    Computed from the layer stack rather than from a roll volume, because the
    two Params objects do not both carry roll geometry and because the stack
    average is the quantity BPX's lumped thermal model actually wants. The
    layer rho and cp are already phase-consistent -- solid and imbibed
    electrolyte counted once each -- so this is a second average over an
    already-averaged quantity and nothing is double-counted.
    """
    tot_L = sum(x.L for x in p.layers)
    mass = sum(x.rho * x.L for x in p.layers)                # g/cm^2 of stack
    heat = sum(x.rho * x.cp * x.L for x in p.layers)
    return mass / tot_L * 1e3, heat / mass * 1e3             # kg/m^3, J/kg/K


def _common(p, cell_mass_g, can_od_cm, can_h_cm, capacity_ah, thermal):
    """[D] The Cell and State blocks, which are geometry and not chemistry."""
    r, h = can_od_cm / 2.0, can_h_cm
    volume = np.pi * r ** 2 * h * 1e-6                     # m^3
    ext_area = (2 * np.pi * r * h + 2 * np.pi * r ** 2) * 1e-4
    rho, cp = thermal
    cell = {
        # The wound roll opened out: BPX wants one electrode pair of a given
        # area, and the solver's own model depth already counts both coated
        # faces, so the area is the winding length times that depth.
        "Electrode area [m2]": float(p.H * p.depth * 1e-4),
        "Number of electrode pairs connected in parallel to make a cell": 1,
        "Lower voltage cut-off [V]": float(p.V_cut),
        "Upper voltage cut-off [V]": 4.2 if p.V_cut > 2.2 else 3.6,
        "Nominal cell capacity [A.h]": round(float(capacity_ah), 3),
        "Reference temperature [K]": float(p.T0),
        # Jelly-roll averages, not whole-cell: the can and the hardware are a
        # separate thermal network in this work and BPX has no field for it.
        "Density [kg.m-3]": round(float(rho), 1),
        "Specific heat capacity [J.K-1.kg-1]": round(float(cp), 1),
        "External surface area [m2]": round(float(ext_area), 6),
        "Volume [m3]": float(f"{volume:.6g}"),
    }
    state = {
        "Initial conditions": {
            "Initial state-of-charge": 1.0,
            "Initial temperature [K]": float(p.T0),
            "Initial electrolyte concentration [mol.m-3]": float(p.ce0 * 1e6),
        },
        "Thermal environment": {
            "Ambient temperature [K]": float(p.Ta),
            "Heat transfer coefficient [W.m-2.K-1]": float(p.h_can * 1e4),
        },
    }
    return cell, state


def bpx_nmc():
    """[D] The NMC 4680 as BPX. CC BY 4.0."""
    import gw4680 as G
    import parts4680 as PT
    import stack4680 as S4

    p = G.params_4680(C_rate=1.0)
    L = {x.name: x for x in p.layers}
    d = G.derived_metrics(p)
    cell, state = _common(p, S4.closure()["measured"], PT.CAN_OD, PT.CAN_H,
                          d["cell capacity [Ah]"], _stack_thermal(p))
    return {
        "Header": {
            "BPX": "1.0.0",
            "Title": "Tesla-like NMC811/graphite 4680 cylindrical cell",
            "Description":
                "Parameter set for a 4680 cylindrical cell with an "
                "NMC811/graphite chemistry. Geometry and the phase inventory "
                "are identified in this work from the teardown of Ank et al. "
                "(2023, doi:10.1149/1945-7111/ad14d0) and Battery Design's "
                "public 4680 data; the electrochemistry is transferred from "
                "the LG M50 parameterisation of Chen et al. (2020, "
                "doi:10.1149/1945-7111/ab9050); electrolyte transport is "
                "Nyman et al. (2008, doi:10.1016/j.electacta.2008.04.023). "
                "The positive-electrode diffusivity, its activation energy "
                "and the anode window are CALIBRATED (paper 1 v2, Section "
                "5.1, 1 Oct 2026) on the open 1C and 2C charge test of Ank "
                "et al. (2023, Fig. 16): can temperature and charge at 4.2 V, "
                "Nelder-Mead on a normalised least-squares cost. Result: "
                "diffusion time r_s^2/D_s = 8031 s (within x1.2 of the "
                "transferred Chen2020 value), E_a,Ds = 11.0 kJ/mol, anode "
                "theta0 = 0.700. The calibration also identified a holder "
                "contact heat of 0.94 mOhm outside the voltage path (0.5 "
                "mOhm per terminal, as Ank et al. estimate); it is a property "
                "of their holder and is NOT in this file. The entropic "
                "coefficient is the average over the working window of "
                "O'Regan et al. (2022, doi:10.1016/j.electacta.2022.140700); "
                "the solver uses the function of stoichiometry. The "
                "stoichiometry windows are calibrated to the cut-off "
                "voltages: OCV(100 %) = 4.2 V and a C/20 discharge to 2.5 V "
                "delivering 22.31 Ah (Ank et al.), with the Chen2020 OCPs and "
                "the loadings kept. BPX has no field for a provenance tag, "
                "for the closure checks, or for the wound-cell thermal "
                "network; those are in parameters_nmc.csv and in the released "
                "modules. VERSION 1.2 (1 Oct 2026): the calibrated values "
                "above (version 1.1 carried r_s^2/D_s = 700 s, E_a,Ds = 25 "
                "kJ/mol, theta0 = 0.701). VERSION 1.1 (27 Sep 2026) corrected "
                "the version 1.0 files, which carried the solid conductivity "
                "sigma instead of the effective eps_s*sigma the solver uses. "
                "Licence: CC BY 4.0.",
            "References":
                "Chen et al. 2020; Ank et al. 2023; Nyman et al. 2008; "
                "O'Regan et al. 2022.",
            "Model": "DFN",
        },
        "State": state,
        "Parameterisation": {
            "Cell": cell,
            "Electrolyte": {
                "Cation transference number": float(p.t_plus),
                "Conductivity [S.m-1]": NYMAN_KAPPA,
                "Diffusivity [m2.s-1]": NYMAN_DE,
                "Conductivity activation energy [J.mol-1]": float(p.Eact_kap),
                "Diffusivity activation energy [J.mol-1]": float(p.Eact_De),
            },
            "Negative electrode": _electrode(
                L["anode"], G.ocp_graphite_chen2020, L["anode"].dUdT,
                L["anode"].Eact_i0, L["anode"].Eact_Ds,
                _mref_to_k(L["anode"], p)),
            "Positive electrode": _electrode(
                L["cathode"], G.ocp_nmc811_chen2020, L["cathode"].dUdT,
                L["cathode"].Eact_i0, L["cathode"].Eact_Ds,
                _mref_to_k(L["cathode"], p)),
            "Separator": {
                "Thickness [m]": float(L["separator"].L * 1e-2),
                "Porosity": float(L["separator"].eps_e),
                "Transport efficiency": round(
                    float(L["separator"].eps_e ** 1.5), 6),
            },
        },
    }


def bpx_lfp():
    """[D] The BYD FC4680 as BPX. CC-BY-SA 4.0 -- see the module docstring."""
    import byd4680_params as B
    import byd_parts as BP
    import byd_stack as BS
    import gw4680 as G

    p = B.params_byd4680(C_rate=1.0)
    L = {x.name: x for x in p.layers}
    cell, state = _common(p, BS.M_CELL, BP.CAN_OD, BP.CAN_H, BS.Q_CELL,
                          _stack_thermal(p))
    return {
        "Header": {
            "BPX": "1.0.0",
            "Title": "BYD FC4680 LFP/graphite 4680 cylindrical cell",
            "Description":
                "Parameter set for the BYD FC4680, an LFP/graphite 4680 "
                "cylindrical cell. Geometry and the phase inventory are "
                "identified in this work from the teardown of Liu et al. "
                "(2025, doi:10.1149/1945-7111/ae1e33) and from an "
                "over-determined system of mass, capacity and volume "
                "closures, which determines the cathode porosity, the can "
                "wall and the separator thickness that no source reports. "
                "The ELECTROCHEMISTRY is transferred from About:Energy's BPX "
                "parameterisation of an LFP/graphite 2 Ah 18650 "
                "(github.com/About-Energy-OpenSource/"
                "About-Energy-BPX-Parameterisation), used under CC-BY-SA 4.0; "
                "electrolyte transport is Nyman et al. (2008). "
                "KNOWN LIMITATION: LiFePO4 discharges through a two-phase "
                "transformation, so the solid diffusivity here is an "
                "EFFECTIVE quantity absorbing phase-boundary kinetics into a "
                "Fickian form, and the model cannot represent voltage "
                "hysteresis. Do not read the diffusivity as a material "
                "property. "
                "CALIBRATED (paper 1 v2, Section 5.2, 1 Oct 2026) on the "
                "author's bench test of cell c1 (1C and 2C discharges to "
                "2.0 V at 25 C, three surface thermocouples; data released "
                "with this set): cathode D_s x0.767 (r_s^2/D_s = 4740 s), "
                "E_a,Ds = 18.8 kJ/mol, anode theta0 = 0.796, and an "
                "end-of-lithiation drop added to the About:Energy cathode "
                "potential, -1.4 V ((theta - 1 + w)/w)^n for theta > 1 - w, "
                "w = 0.131, n = 2.58, which the window-guarded fit lacked; "
                "the OCP table of the positive electrode therefore runs to "
                "theta = 1. The heat-transfer coefficient is the measured "
                "22.4 W/m^2/K of the bench fixture (hA = 0.337 W/K). The "
                "calibration also identified a clamp contact heat of 1.34 "
                "mOhm outside the voltage path; it is a property of the "
                "clamp and is NOT in this file. "
                "VERSION 1.2 (1 Oct 2026): the calibrated values above. "
                "VERSION 1.1 (27 Sep 2026): corrected the version 1.0 files, "
                "which carried the solid conductivity sigma instead of the "
                "effective eps_s*sigma the solver uses and a zero "
                "heat-transfer coefficient. "
                "LICENCE: CC-BY-SA 4.0, because it incorporates About:Energy "
                "material. The rest of this dataset is CC BY 4.0.",
            "References":
                "Liu et al. 2025; About:Energy LFP BPX parameterisation "
                "(CC-BY-SA 4.0); Nyman et al. 2008.",
            "Model": "DFN",
        },
        "State": state,
        "Parameterisation": {
            "Cell": cell,
            "Electrolyte": {
                "Cation transference number": float(p.t_plus),
                "Conductivity [S.m-1]": NYMAN_KAPPA,
                "Diffusivity [m2.s-1]": NYMAN_DE,
                "Conductivity activation energy [J.mol-1]": float(p.Eact_kap),
                "Diffusivity activation energy [J.mol-1]": float(p.Eact_De),
            },
            "Negative electrode": _electrode(
                L["anode"], B.ocp_graphite_aboutenergy, L["anode"].dUdT,
                L["anode"].Eact_i0, L["anode"].Eact_Ds, B.M_REF_NEG),
            "Positive electrode": _electrode(
                L["cathode"], B.ocp_lfp_aboutenergy, L["cathode"].dUdT,
                L["cathode"].Eact_i0, L["cathode"].Eact_Ds, B.M_REF_POS,
                hi_override=1.0),
            "Separator": {
                "Thickness [m]": float(L["separator"].L * 1e-2),
                "Porosity": float(L["separator"].eps_e),
                "Transport efficiency": round(
                    float(L["separator"].eps_e ** 1.5), 6),
            },
        },
    }


def _mref_to_k(layer, p):
    """[D] The inverse of byd4680_params._k_to_mref, for a set that stores
    m_ref rather than a BPX constant. m_ref = k F / (cs_max sqrt(ce0))."""
    m_si = layer.m_ref * 1e-5
    return m_si * (layer.cs_max * 1e6) * (p.ce0 * 1e6) ** 0.5 / F


# ---------------------------------------------------------------------------
#  THE PROVENANCE TABLE, WHICH BPX HAS NO ROOM FOR
# ---------------------------------------------------------------------------
def provenance_records():
    """[D] Every parameter of both cells, in SI, with its tag and its reason."""
    import byd4680_params as B
    import gw4680 as G

    nmc = {(r["group"], r["field"]): r for r in G.parameter_rows()}
    lfp = {(r["group"], r["field"]): r for r in B.parameter_rows()}
    out = []
    for k, a in nmc.items():
        b = lfp[k]
        unit, factor = HS.display_unit(a["field"], a["unit"])
        out.append(dict(
            group=a["group"], parameter=a["field"], symbol=a["symbol"] or "",
            unit=unit,
            nmc_value=None if a["value"] is None else a["value"] * factor,
            nmc_tag=a["tag"], nmc_source=a["note"],
            lfp_value=None if b["value"] is None else b["value"] * factor,
            lfp_tag=b["tag"], lfp_source=b["note"],
            open_issue=b.get("open") or ""))
    return out


def write(dest="data", verify=True):
    os.makedirs(dest, exist_ok=True)
    files = []

    for name, obj in (("bpx_nmc4680.json", bpx_nmc()),
                      ("bpx_lfp_fc4680.json", bpx_lfp())):
        path = os.path.join(dest, name)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1)
        files.append(path)

    recs = provenance_records()
    path = os.path.join(dest, "parameters_both_cells.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    files.append(path)

    path = os.path.join(dest, "parameters_both_cells.json")
    with open(path, "w") as fh:
        json.dump(recs, fh, indent=1)
    files.append(path)

    for f in files:
        print(f"  wrote {f}  ({os.path.getsize(f) / 1024:.0f} kB)")
    if verify:
        print()
        verify_exports(dest)
    return files


# ---------------------------------------------------------------------------
#  VERIFICATION -- a file that has not been loaded back is not an export
# ---------------------------------------------------------------------------
def verify_exports(dest="data"):
    """[D] Load both files with PyBaMM and check they describe our cells.

    Writing JSON always succeeds. The question is whether the file, read by
    the tool it was written for, reproduces the numbers the solver holds --
    so the check is done through PyBaMM's parser rather than by re-reading our
    own dictionary, which would only prove that json round-trips.
    """
    try:
        import pybamm
    except ImportError:
        print("  PyBaMM not installed -- export written but NOT verified.")
        return False

    import warnings
    ok = True
    for name, mod, cells in (
            ("bpx_nmc4680.json", "gw4680", ("Negative", "Positive")),
            ("bpx_lfp_fc4680.json", "byd4680_params", ("Negative", "Positive"))):
        path = os.path.join(dest, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pv = pybamm.ParameterValues.create_from_bpx(path)
        src = __import__(mod)
        p = (src.params_4680(C_rate=1.0) if mod == "gw4680"
             else src.params_byd4680(C_rate=1.0))
        L = {x.name: x for x in p.layers}
        print(f"  {name}")
        for dom, our in (("Negative", "anode"), ("Positive", "cathode")):
            lay = L[our]
            checks = [
                ("thickness [m]",
                 pv[f"{dom} electrode thickness [m]"], lay.L * 1e-2),
                ("particle radius [m]",
                 pv[f"{dom} particle radius [m]"], lay.rs * 1e-2),
                ("max concentration [mol.m-3]",
                 pv[f"Maximum concentration in {dom.lower()} electrode "
                    f"[mol.m-3]"], lay.cs_max * 1e6),
                ("porosity", pv[f"{dom} electrode porosity"], lay.eps_e),
                ("conductivity [S.m-1]",
                 pv[f"{dom} electrode conductivity [S.m-1]"],
                 lay.sigma_eff * 1e2),        # eps_s * sigma, as the solver
            ]
            for what, got, want in checks:
                rel = abs(got - want) / abs(want) if want else abs(got)
                flag = "ok " if rel < 1e-9 else "FAIL"
                if rel >= 1e-9:
                    ok = False
                    print(f"    {flag} {dom:8s} {what:30s} "
                          f"{got:.6g} vs {want:.6g}")
            # the kinetics round trip, which is the one that was wrong
            fn = pv[f"{dom} electrode exchange-current density [A.m-2]"]
            cs = 0.5 * lay.cs_max * 1e6
            v = fn(p.ce0 * 1e6, cs, lay.cs_max * 1e6, p.T0)
            i0 = float(v.evaluate()) if hasattr(v, "evaluate") else float(v)
            ours = (lay.m_ref * 1e-5) * np.sqrt(
                (p.ce0 * 1e6) * cs * (lay.cs_max * 1e6 - cs))
            rel = abs(i0 - ours) / ours
            ok &= rel < 1e-9
            print(f"    {'ok ' if rel < 1e-9 else 'FAIL'} {dom:8s} "
                  f"{'exchange current i0 [A/m2]':30s} "
                  f"{i0:.6g} vs {ours:.6g}   rel {rel:.1e}")
        cap = pv["Nominal cell capacity [A.h]"]
        print(f"    .. nominal capacity {cap:.3f} A.h")
    print(f"\n  {'ALL EXPORTS VERIFIED' if ok else 'EXPORT VERIFICATION FAILED'}"
          f" against PyBaMM's own parser.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    write(a.dest, not a.no_verify)

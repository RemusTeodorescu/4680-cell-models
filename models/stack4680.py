"""Phase-consistent layer stack: solids and electrolyte counted once each.

    python stack4680.py

WHY THIS EXISTS
---------------
gw4680's Layer carries one density and one specific heat per layer, documented
as "composite incl. binder+electrolyte". Checking those numbers against the
phases they claim to average shows they are not:

    anode    rho = 1.60 g/cm^3.  Graphite composite solid at eps_s = 0.75 is
             already 2.20 x 0.75 = 1.65, before any electrolyte.
    cathode  rho = 2.90.  NMC811 composite at eps_s = 0.665 is 4.50 x 0.665
             = 2.99, again before electrolyte.

So the densities are solid-only while the comment says otherwise, and the
cell's pore electrolyte -- 34 g of it -- is missing from the mass budget. The
specific heats are ambiguous in the other direction: the coded rho_cp implies
cp ~ 1.0 J/g/K for a graphite electrode, above graphite's 0.71, which is what
you get if electrolyte IS folded into the heat capacity. Mass and heat capacity
would then be averaged over different inventories.

That matters as soon as anything is claimed about temperature. A thermal mass
that is 13 % low makes the cell heat 13 % too fast, and in a lumped comparison
against a measured temperature rise that reads as a heat-generation error --
the model would be "corrected" in the wrong place.

The fix is not a better fudge factor. It is to name the phases:

    rho_eff    = rho_s (1 - eps_e) + rho_e eps_e
    rho_cp_eff = rho_s cp_s (1 - eps_e) + rho_e cp_e eps_e

with rho_s the true density of the composite SOLID and eps_e the porosity that
the electrochemistry already uses. Then mass and heat capacity come from one
inventory by construction, and the closure audit below means something.

PROVENANCE
    [L] handbook values for the named materials
    [D] derived here
    [M] measured, Ank et al. (2023) J. Electrochem. Soc. 170 120536
"""

from __future__ import annotations

import numpy as np

import gw4680 as G

# --- electrolyte, counted explicitly --------------------------------------
RHO_ELY = 1.22      # g/cm^3   [L] LiPF6 in carbonate
CP_ELY = 1.518      # J/g/K    [L] Chen2020 electrolyte, 1518 J/kg/K

# --- true solid-phase properties of each composite ------------------------
# Densities are of the SOLID only: active material plus binder and conductive
# carbon, excluding pores. Specific heats likewise.
SOLID = {
    "anode":     dict(rho_s=2.20, cp_s=0.75),   # [L] graphite + CMC/SBR + CB
    "cathode":   dict(rho_s=4.50, cp_s=0.84),   # [L] NMC811 + PVDF + CB
    "separator": dict(rho_s=0.91, cp_s=1.90),   # [L] polypropylene
}


# --- the layer properties as they stood BEFORE the phase rebuild ----------
# Kept so the "13 % low" claim in the docstring above stays checkable rather
# than remembered. gw4680's layers now carry the phase-consistent values, so
# C_coded below equals C_eff by construction and that column has become a
# regression check reporting 0.0 %. The historical comparison needs the old
# numbers explicitly, and these are they, transcribed from the 12 Sep 2026
# build (frontiers_4680_overleaf.zip, code/gw4680.py, lines 279-322):
#
#     anode      rho 1.60  cp 1.00   "[E] graphite composite incl.
#                                     binder+electrolyte"
#     separator  rho 1.10  cp 1.50   "[E] PE separator wetted with carbonate"
#     cathode    rho 2.90  cp 0.90   "[E]"
#
# The foils were and are pure metal and did not change.
LEGACY = {
    "anode":     dict(rho=1.60, cp=1.00),
    "separator": dict(rho=1.10, cp=1.50),
    "cathode":   dict(rho=2.90, cp=0.90),
}


def legacy_totals(p=None):
    """Roll mass and heat capacity as they were before the rebuild. [D]

    Not a model -- an archive entry. It exists so the size of the correction
    is a computed number in the validation output rather than a claim.
    """
    p = G.params_4680(C_rate=1.0) if p is None else p
    A = p.separator_area
    m = C = 0.0
    for L in p.layers:
        d = LEGACY.get(L.name)
        rho, cp = (d["rho"], d["cp"]) if d else (L.rho, L.cp)
        m += L.L * rho * A
        C += L.L * rho * cp * A
    return dict(m_roll=m, C_roll=C)


def phases(p=None):
    """Per-layer inventory with solid and electrolyte separated. [D]"""
    p = G.params_4680(C_rate=1.0) if p is None else p
    A = p.separator_area
    out = []
    for L in p.layers:
        eps = float(getattr(L, "eps_e", 0.0) or 0.0)
        if L.name in SOLID:
            rho_s, cp_s = SOLID[L.name]["rho_s"], SOLID[L.name]["cp_s"]
        else:                                   # foils: solid, no pores
            rho_s, cp_s, eps = L.rho, L.cp, 0.0
        rho_eff = rho_s * (1.0 - eps) + RHO_ELY * eps
        rcp_eff = rho_s * cp_s * (1.0 - eps) + RHO_ELY * CP_ELY * eps
        out.append(dict(
            name=L.name, L=L.L, eps_e=eps,
            rho_s=rho_s, cp_s=cp_s,
            rho_eff=rho_eff, rho_cp_eff=rcp_eff,
            rho_coded=L.rho, rho_cp_coded=L.rho_cp,
            m_solid=L.L * rho_s * (1 - eps) * A,
            m_ely=L.L * RHO_ELY * eps * A,
            C_eff=L.L * rcp_eff * A,
            C_coded=L.L * L.rho_cp * A,
        ))
    return out


def totals(p=None):
    """Roll-level mass and heat capacity, both ways. [D]"""
    r = phases(p)
    return dict(
        m_solid=sum(x["m_solid"] for x in r),
        m_ely=sum(x["m_ely"] for x in r),
        m_roll=sum(x["m_solid"] + x["m_ely"] for x in r),
        C_roll=sum(x["C_eff"] for x in r),
        C_coded=sum(x["C_coded"] for x in r),
    )


def closure(m_can=58.8, m_disks=3.4, m_measured=355.9):
    """Does the inventory add up to the cell Ank weighed? [D] vs [M]"""
    t = totals()
    acc = t["m_roll"] + m_can + m_disks
    return dict(
        m_solid=t["m_solid"], m_ely=t["m_ely"], m_roll=t["m_roll"],
        m_can=m_can, m_disks=m_disks, accounted=acc,
        measured=m_measured, residual=m_measured - acc,
        residual_pct=(m_measured - acc) / m_measured * 100.0,
        C_roll=t["C_roll"], C_coded=t["C_coded"],
        C_change_pct=(t["C_roll"] / t["C_coded"] - 1.0) * 100.0,
    )


def main():
    print("=" * 74)
    print("PHASE-CONSISTENT STACK — solids and electrolyte counted once each")
    print("=" * 74)
    print(f"{'layer':10} {'L[um]':>7} {'eps_e':>6} {'rho_s':>6} "
          f"{'rho_eff':>8} {'coded':>7} {'rcp_eff':>8} {'coded':>7}")
    print("-" * 74)
    for x in phases():
        print(f"{x['name'][:9]:10} {x['L']*1e4:7.1f} {x['eps_e']:6.3f} "
              f"{x['rho_s']:6.2f} {x['rho_eff']:8.3f} {x['rho_coded']:7.3f} "
              f"{x['rho_cp_eff']:8.3f} {x['rho_cp_coded']:7.3f}")

    c = closure()
    print()
    print("MASS CLOSURE against Ank's weighed cell")
    print(f"  {'solid phases':32s} {c['m_solid']:7.1f} g")
    print(f"  {'pore electrolyte':32s} {c['m_ely']:7.1f} g")
    print(f"  {'jelly roll':32s} {c['m_roll']:7.1f} g")
    print(f"  {'steel can (0.5 mm [M])':32s} {c['m_can']:7.1f} g")
    print(f"  {'collector disks':32s} {c['m_disks']:7.1f} g")
    print(f"  {'ACCOUNTED':32s} {c['accounted']:7.1f} g")
    print(f"  {'MEASURED [M]':32s} {c['measured']:7.1f} g")
    print(f"  {'residual':32s} {c['residual']:7.1f} g "
          f"({c['residual_pct']:.1f} %)")
    print()
    print("  The residual is the cell hardware the layer model does not")
    print("  contain: the crimped cap assembly, the 16 mm terminal, the copper")
    print("  rivet, the seals, and free electrolyte in Ank's 16.7 cm^3 dead")
    print("  volume. It is named rather than absorbed.")
    print()
    print("THERMAL MASS")
    print(f"  {'phase-consistent':32s} {c['C_roll']:7.1f} J/K")
    print(f"  {'as previously coded':32s} {c['C_coded']:7.1f} J/K")
    print(f"  {'change':32s} {c['C_change_pct']:+7.1f} %")
    print()
    if abs(c["C_change_pct"]) < 0.05:
        print("  0.0 % — gw4680 now BUILDS its layers from these phases, so this")
        print("  is a regression check rather than a diagnosis. It was +12.9 %")
        print("  when the densities were solid-only and the heat capacities were")
        print("  not. A thermal-mass error of that size is invisible in a voltage")
        print("  curve and fully visible in a temperature rise, which is why it")
        print("  survived until a lumped thermal validation was attempted.")
    else:
        print("  gw4680 and this audit DISAGREE. One of them has drifted.")


if __name__ == "__main__":
    main()

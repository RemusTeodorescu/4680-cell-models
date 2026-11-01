"""Constant-current discharge of either 4680 cell with the thermal DFN (TDFN) of paper 1.

    python examples/run_discharge.py lfp 1      # BYD FC4680, 1C, bench conditions
    python examples/run_discharge.py nmc 2      # Tesla 4680 (NMC811), 2C

Writes discharge_<cell>_<C>C.csv (time, current, voltage, roll and can temperature).
Model: quartic particle profile, reaction heat at the open-circuit potential of the
particle-mean concentration, two-node (roll + can) thermal balance.
LFP: fixture of the bench test in data/bench (hA = 0.337 W/K, whole-cell heat
capacity 277 J/K, effective ambient 23.2 C), cut-off 2.0 V.
NMC: whole-can convection h = 10 W/m2/K, ambient 25 C, cut-off 2.5 V.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))

import byd4680_params as B          # noqa: E402
import byd_parts as BP              # noqa: E402
import gw4680 as G                  # noqa: E402
from ddfnt import run_profile       # noqa: E402
from gw2d import GuWang2D           # noqa: E402

NX = (1, 10, 4, 10, 1)              # control volumes: Cu, anode, separator, cathode, Al


def lfp_model():
    T = 23.2 + 273.15
    r = BP.CAN_OD / 2.0
    A_ext = 2.0 * np.pi * r * BP.CAN_H + 2.0 * np.pi * r ** 2          # cm^2
    hA, C_cell = 0.337, 277.0                                          # W/K, J/K
    p = B.params_byd4680(C_rate=1.0, two_node=True, two_node_A_ext=A_ext,
                         particle_closure="quartic", heat_ocp="mean",
                         T0=T, Ta=T, h_can=hA / A_ext)
    m = GuWang2D(p, nx_per_layer=NX, ny=1)
    p.two_node_C_can = C_cell - float((m.rho_cp * m.m.V).sum())
    return GuWang2D(p, nx_per_layer=NX, ny=1), 2.0


def nmc_model():
    T = 25.0 + 273.15
    p = G.params_4680_two_node(C_rate=1.0, particle_closure="quartic", heat_ocp="mean",
                               T0=T, Ta=T)
    return GuWang2D(p, nx_per_layer=NX, ny=1), 2.5


def main(cell="lfp", C=1.0):
    m, V_cut = (lfp_model if cell == "lfp" else nmc_model)()
    h = run_profile(m, [(1.3 * 3600.0 / C, C)], dt_first=0.05, dt_max=20.0,
                    dV_max=0.005, V_min=V_cut, V_max=5.0)
    out = f"discharge_{cell}_{C:g}C.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "current_A", "voltage_V", "T_roll_C", "T_can_C", "charge_Ah"])
        for i in range(len(h["t"])):
            w.writerow([f"{h['t'][i]:.2f}", f"{h['I'][i]:.4f}", f"{h['V'][i]:.5f}",
                        f"{h['Tavg'][i] - 273.15:.3f}", f"{h['Tcan'][i] - 273.15:.3f}",
                        f"{h['Q_Ah'][i]:.4f}"])
    print(f"{cell.upper()} {C:g}C: {h['Q_Ah'][-1]:.2f} Ah to {V_cut} V, "
          f"roll {h['Tavg'][-1] - 273.15:.1f} C at cut-off -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lfp",
         float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)

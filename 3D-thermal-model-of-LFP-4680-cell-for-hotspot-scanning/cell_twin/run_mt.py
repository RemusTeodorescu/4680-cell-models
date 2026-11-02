"""Coupled sheet + 3-D runs with MT-TDFN columns (32 patches), 2C from 80 % SOC, 600 s."""
import json, time, sys
import numpy as np
import sheet, byd_parts
from geom3d import Cell
c = Cell()
g_cal, Rc = sheet.calibrate_foil_gain(byd_parts.build().R_collect_mean)
cases = {
    "mt_B_no_short": dict(foil_gain=g_cal),
    "mt_B_short_3ohm_Al-anode_mid": dict(foil_gain=g_cal, short=dict(s=float(c.s_of_r(13.0)), z=c.H_cath / 2, R=3.0, type="Al-anode")),
    "mt_A_literal_no_short": dict(foil_gain=1.0),
}
which = sys.argv[1:] or list(cases)
out = {}
for k in which:
    t0 = time.time()
    r = sheet.coupled_run(t_end=600, dt=10, local="mt", n_s=16, n_z=2, label=k, **cases[k])
    r["seconds"] = time.time() - t0; r["foil_gain"] = cases[k]["foil_gain"]
    out[k] = r
    h = r["hist"][-1]
    print(k, f"{r['seconds']:.0f}s V={h['V']:.4f} P={h['P_total']:.2f} Tin={h['max_internal']:.2f} Tside={h['max_side']:.2f} "
          f"j={h['j_min']:.0f}..{h['j_max']:.0f} tabs={np.round(h['tab_currents'],1)}", flush=True)
    json.dump(dict(foil_gain_cal=g_cal, runs=out), open("out/coupled_mt.json", "w"), indent=1, default=float)

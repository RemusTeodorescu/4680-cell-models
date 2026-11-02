"""Point 4: unit-short response library of the cell twin, for the module twin.
For each location (radius x height x angle class) the response to a 1 W step
short: hottest point inside, hottest point on the can side, mean can side.
Linear model -> any short history P(t) follows by superposition (Duhamel).
Peaks are from the coarse library grid; the footprint/patch correction of
patch_study.json is applied in the twin (factor per footprint)."""
import json, time, numpy as np
from geom3d import Cell, build_grid
from thermal3d import BC, assemble, Solver, box_source, surface_masks, field_stats

c = Cell()
L = {n: (a, b) for n, a, b in c.z_layout()}
z0, z1 = L["roll"]
R = [4.0, 8.0, 12.0, 16.0, 20.0, 22.2]
Z = {"base_end": z0 + 2.0, "lower": z0 + 0.25 * (z1 - z0), "mid": 0.5 * (z0 + z1), "upper": z0 + 0.75 * (z1 - z0), "cap_end": z1 - 2.0}
tab = next(t for t in c.tabs() if t["kind"] == "cathode" and abs(t["r"] - 16) < 0.5)
T_OUT = np.unique(np.round(np.concatenate([[0.5, 1, 2, 3, 5], np.geomspace(10, 3600, 30)]), 1))

g = build_grid(c, refine=0.75)
K, C, Gb = assemble(g, BC(22.4, 22.4, 22.4)); sol = Solver(K, C, tol=1e-8)
m = surface_masks(g)
sched = [(0.5, 4), (1.0, 8), (5.0, 10), (15.0, 16), (30.0, 10), (60.0, 20), (120.0, 20)]   # to 3600 s
lib = dict(t=T_OUT.tolist(), radii_mm=R, heights=list(Z), grid=g.shape, entries=[])
t0 = time.time()
for zname, zz in Z.items():
    for r in R:
        th = np.pi if True else 0.0
        for ang, th0 in (("away_from_tab", 0.25 * np.pi), ("at_tab_angle", tab["theta"])):
            if ang == "at_tab_angle" and zname != "cap_end":
                continue
            Q = box_source(g, 1.0, r, th0, zz, d_r=0.6, w_arc=2.0, h_z=2.0)
            rec = {"t": [], "pk": [], "side_max": [], "side_mean": []}
            def cb(t, T):
                st = field_stats(g, T, m)
                rec["t"].append(t); rec["pk"].append(float(T.max())); rec["side_max"].append(st["max_side"]); rec["side_mean"].append(st["mean_side"])
            sol.run(Q, sched, callback=cb)
            tt = np.array(rec["t"])
            e = dict(r=r, height=zname, z=zz, angle=ang,
                     peak=np.interp(T_OUT, tt, rec["pk"]).round(5).tolist(),
                     side_max=np.interp(T_OUT, tt, rec["side_max"]).round(5).tolist(),
                     side_mean=np.interp(T_OUT, tt, rec["side_mean"]).round(5).tolist())
            lib["entries"].append(e)
            print(zname, r, ang, f"{time.time()-t0:.0f}s pk3600={e['peak'][-1]:.2f}", flush=True)
            json.dump(lib, open("out/response_library.json", "w"))

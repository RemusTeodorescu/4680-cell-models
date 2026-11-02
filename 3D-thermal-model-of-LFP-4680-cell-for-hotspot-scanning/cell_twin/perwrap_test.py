"""Answer to 'one node per layer?': short peak vs radial resolution, steady and 60 s."""
import json, time, sys, numpy as np
from geom3d import Cell, build_grid
from thermal3d import BC, assemble, Solver, box_source, surface_masks, field_stats, schedule_to, short_locations
c = Cell(); loc = short_locations(c)["mid_radius"]
res = []
for label, kw in [("44x48x48 (base)", dict()),
                  ("1 cell per wrap, 2 mm", dict(n_roll_r=65, n_th=48, n_roll_z=36)),
                  ("1 cell per wrap, 1 mm", dict(n_roll_r=65, n_th=96, n_roll_z=71)),
                  ("2 cells per wrap, 1 mm", dict(n_roll_r=130, n_th=96, n_roll_z=71))][:int(sys.argv[1]) if len(sys.argv) > 1 else 4]:
    t0 = time.time()
    g = build_grid(c, **kw); K, C, Gb = assemble(g, BC(22.4, 22.4, 22.4)); sol = Solver(K, C, tol=1e-8)
    Q = box_source(g, 1.0, *loc); m = surface_masks(g)
    Ts = sol.steady(Q); st = field_stats(g, Ts, m)
    T60 = sol.run(Q, schedule_to(60)); s60 = field_stats(g, T60, m)
    r = dict(grid=label, shape=g.shape, n=g.n, steady_peak=st["max_internal"], steady_side=st["max_side"],
             peak_60s=s60["max_internal"], side_60s=s60["max_side"], seconds=time.time() - t0)
    print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open("out/perwrap.json", "w"), indent=1)

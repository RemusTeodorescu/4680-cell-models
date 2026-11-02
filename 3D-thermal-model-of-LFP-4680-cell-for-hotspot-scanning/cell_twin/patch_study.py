"""Point 3: locally refined patch at the short. Peak temperature rise per watt
against the short's footprint, with grid convergence; transient (pulse) shorts.
Short at r = 13 mm, mid-height, one wrap thick (0.3 mm), square footprint w x w."""
import json, time, numpy as np
from geom3d import Cell, build_grid
from thermal3d import BC, assemble, Solver, box_source, surface_masks, field_stats, schedule_to

c = Cell()
L = {n: (a, b) for n, a, b in c.z_layout()}
r0, th0, z0 = 13.0, np.pi, 0.5 * sum(L["roll"])
res = []

def case(w, h, pulses=()):
    t0 = time.time()
    g = build_grid(c, n_roll_r=65, patch=dict(r0=r0, th0=th0, z0=z0, h_inplane=h, half=max(2.0, 2 * w)))
    K, C, Gb = assemble(g, BC(22.4, 22.4, 22.4)); sol = Solver(K, C, tol=1e-8)
    Q = box_source(g, 1.0, r0, th0, z0, d_r=c.pitch, w_arc=w, h_z=w)
    m = surface_masks(g)
    roll = None
    pk = {}
    def cb(t, T):
        for tt in (1, 10, 60):
            if abs(t - tt) < 1e-6:
                pk[tt] = float(T.max())
    sched = [(0.25, 4), (0.5, 2), (1.0, 8), (5.0, 10)]           # to 60 s
    sol.run(Q, sched, callback=cb)
    Ts = sol.steady(Q)
    r = dict(w_mm=w, h_mm=h, shape=g.shape, n=g.n, peak_1s=pk.get(1), peak_10s=pk.get(10), peak_60s=pk.get(60),
             peak_steady=float(Ts.max()), side_steady=field_stats(g, Ts, m)["max_side"])
    for tau in pulses:                                            # 1 J/s for tau seconds, then off
        mx, t_mx = [0.0], [0.0]
        def cb2(t, T):
            v = float(T.max())
            if v > mx[0]:
                mx[0], t_mx[0] = v, t
        Qf = lambda t, tau=tau: Q if t <= tau + 1e-9 else 0.0 * Q
        sched = [(min(0.25, tau / 4), int(round(tau / min(0.25, tau / 4))))] + [(1.0, 20), (5.0, 20)]
        sol.run(Qf, sched, callback=cb2)
        r[f"pulse_{tau:g}s_peak"] = mx[0]; r[f"pulse_{tau:g}s_t_peak"] = t_mx[0]
    r["seconds"] = time.time() - t0
    print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open("out/patch_study.json", "w"), indent=1)

# convergence for the 1 mm footprint
for h in (0.5, 0.25, 0.125):
    case(1.0, h, pulses=(1.0, 10.0) if h == 0.25 else ())
# footprint sweep at h = w/4 (min 0.125)
for w in (0.25, 0.5, 2.0, 4.0):
    case(w, max(0.125, w / 4))

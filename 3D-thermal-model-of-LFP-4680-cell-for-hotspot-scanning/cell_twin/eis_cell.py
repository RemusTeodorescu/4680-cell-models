"""Stage 2: after the voltage GRU flags a group, the BMA7418 scans the 9 cells of that group and
names the faulted cell.

Assumes cell-level EIS access inside a 9p group (NXP states cell or parallel-group level; to be
confirmed). Per cell: Z at 10 and 1000 Hz with the BMA7418 errors of eis.py, 6 scans 10 s apart.
Index x = log|Z(10 Hz)| - log|Z(1000 Hz)|: the per-cell resistance spread multiplies both
frequencies and cancels, while temperature lowers the charge-transfer part (10 Hz, 50 kJ/mol)
much more than the ohmic part (1 kHz, 15 kJ/mol). The shorted cell runs warmer, so its x is the
lowest of the nine. Robust z-score against the other eight: z = (x - median) / (1.4826 MAD).

    python eis_cell.py      # on the GRU test scenarios, at the GRU alarm time
"""
import json
import numpy as np

import module_sim as S
from eis import z_cell, z_with_short, Channel, BMA, OUT, NS, NP

F2 = np.array([10.0, 1000.0])
N_SCAN, DT_SCAN = 6, 10


def cell_index(Zc):
    """Zc (..., 9, 2) measured cell impedances -> x (..., 9)"""
    return np.log(np.abs(Zc[..., 0])) - np.log(np.abs(Zc[..., 1]))


def zscore(x):
    med = np.median(x, -1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), -1, keepdims=True) + 1e-4
    return (x - med) / mad


def run(n_shards=8, thr=None):
    rows, healthy = [], []
    comb = json.loads((OUT / "det_eval_combined.json").read_text())["rows"]
    it = iter(comb)
    Rdc = S.P["R0"] + S.P["R1"]
    for i in range(n_shards):
        seed = 9000 + i
        rng = np.random.default_rng(seed)
        sc = S.sample_scenarios(rng, 64, 7200.0, 1.0)
        out = S.simulate(sc, rng, state_every=DT_SCAN)
        T = out["T_state"] + 273.15; soc = out["soc_state"]
        rng2 = np.random.default_rng(seed + 4242)
        for b in range(64):
            ch = Channel(NP, rng2)                     # 9 cell channels
            scale_all = sc["res"][b] / S.P["R0"]
            if not sc["has"][b]:
                # healthy reference: a random group at a random time
                g = int(rng2.integers(NS)); k0 = int(rng2.integers(30, T.shape[1] - N_SCAN))
                xs = []
                for k in range(k0, k0 + N_SCAN):
                    Zc = z_cell(F2, T[b, k, g * NP:(g + 1) * NP], soc[b, k, g * NP:(g + 1) * NP], scale_all[g * NP:(g + 1) * NP])
                    xs.append(cell_index(ch.measure_fast(Zc, F2)))
                xm = np.mean(xs, 0); healthy.append(float(np.median(xm) - xm.min()))
                continue
            r = next(it)
            if not r["detected"]:
                continue
            g, c = divmod(int(sc["k"][b]), NP)
            t_al = sc["ton"][b] + r["delay"]
            k0 = int(t_al // DT_SCAN)
            if k0 + N_SCAN >= T.shape[1]:
                continue
            xs = []
            for k in range(k0, k0 + N_SCAN):
                Zc = z_cell(F2, T[b, k, g * NP:(g + 1) * NP], soc[b, k, g * NP:(g + 1) * NP], scale_all[g * NP:(g + 1) * NP])
                on = S.short_on({kk: (v[b:b + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (64,) else v) for kk, v in sc.items()},
                                (k + 1) * DT_SCAN)[0]
                if on:
                    R = np.full(NP, np.inf); R[c] = sc["R"][b]
                    ftr = np.zeros(NP); ftr[c] = sc["f_tr"][b]
                    zth = np.zeros(NP); zth[c] = sc["f_th"][b] * Rdc
                    Zc = z_with_short(Zc, F2, R, ftr, zth)
                xs.append(cell_index(ch.measure_fast(Zc, F2)))
            xm = np.mean(xs, 0)
            rows.append(dict(R=float(sc["R"][b]), dur=float(sc["dur"][b]), right=bool(np.argmin(xm) == c),
                             zmin=-float(np.median(xm) - xm.min()), z_true=float(xm[c] - np.median(xm))))
        print("shard", i, len(rows), flush=True)
    healthy = np.array(healthy)
    # statistic: drop of the lowest cell's index below the group median (ln units); named when the drop
    # exceeds what 99 % of healthy groups show
    thr = thr if thr is not None else -float(np.quantile(healthy, 0.99))
    res = dict(thr=thr, healthy_zmin=healthy.tolist(), rows=rows, n_scan=N_SCAN, dt_scan=DT_SCAN)
    (OUT / "eis_cell_eval.json").write_text(json.dumps(res))
    print(f"threshold: drop > {-thr:.5f} (99 % of healthy groups below)")
    for lo, hi in zip([0.3, 1, 3, 10], [1, 3, 10, 3000]):
        rr = [r for r in rows if lo <= r["R"] < hi]
        if rr:
            named = [r for r in rr if r["zmin"] < thr]
            print(f"R {lo}-{hi}: n={len(rr)}  right cell (argmin) {100*np.mean([r['right'] for r in rr]):.0f} %  "
                  f"named with confidence {100*len(named)/len(rr):.0f} %, of which right {100*np.mean([r['right'] for r in named]) if named else float('nan'):.0f} %")
    return res


if __name__ == "__main__":
    run()

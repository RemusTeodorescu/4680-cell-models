"""Training and test data for the short detector, from module_sim.py.

    python gen_data.py [n_shards] [split]

Each shard: 64 scenarios of 2 h at 1 Hz. Stored per scenario:
  dv   (T, 23)  group voltage minus the median of the groups [mV], as measured
  I    (T,)     module current [A]
  dT   (T/10, 207) impedance temperature minus its group mean [K], 0.1 Hz
  ntc  (T, 2)
and the truth: shorted cell, R, onset, duration, intermittent flag, short current (T,).
"""
import sys
import time
from pathlib import Path

import numpy as np

import module_sim as S

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)


def shard(seed, B=64, T=7200.0):
    rng = np.random.default_rng(seed)
    sc = S.sample_scenarios(rng, B, T, 1.0)
    out = S.simulate(sc, rng)
    Vg = out["Vg"]
    dv = ((Vg - np.median(Vg, axis=2, keepdims=True)) * 1e3).astype(np.float32)
    ti = out["timp"].reshape(B, -1, S.NS, S.NP)
    dT = (ti - ti.mean(3, keepdims=True)).reshape(B, -1, S.N).astype(np.float16)
    return dict(dv=dv, I=out["I"].astype(np.float32), dT=dT, ntc=out["ntc"].astype(np.float32),
                isc=out["isc"].astype(np.float32), has=sc["has"], k=sc["k"], R=sc["R"], ton=sc["ton"],
                dur=sc["dur"], inter=sc["inter"], period=sc["period"], f_tr=sc["f_tr"], f_th=sc["f_th"],
                ipos=sc["ipos"], Tin=sc["Tin"], soc0=sc["soc0"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    base = {"train": 1000, "val": 5000, "test": 9000}[split]
    for i in range(n):
        f = DATA / f"{split}_{i:03d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        np.savez_compressed(f, **shard(base + i))
        print(split, i, round(time.time() - t0, 1), "s", flush=True)

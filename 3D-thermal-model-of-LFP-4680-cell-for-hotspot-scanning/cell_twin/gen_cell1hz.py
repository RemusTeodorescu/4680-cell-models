"""Data for the 1 Hz cell-level network: impedance temperature of 9 cells every second,
for the shorted group (or a random group without a short) and one other group per scenario."""
import sys, time
import numpy as np
import module_sim as S
from gen_data import DATA

def shard(seed, B=64, T=7200.0):
    rng = np.random.default_rng(seed)
    sc = S.sample_scenarios(rng, B, T, 1.0, p_short=0.9)
    g = sc["k"] // S.NP
    other = (g + rng.integers(1, S.NS, B)) % S.NS
    keep = np.stack([g, other], 1)
    out = S.simulate(sc, rng, keep_groups=keep)
    return dict(timp1=out["timp1"], keep=keep, has=sc["has"], k=sc["k"], R=sc["R"], ton=sc["ton"], dur=sc["dur"], inter=sc["inter"])

if __name__ == "__main__":
    n = int(sys.argv[1]); split = sys.argv[2]
    base = {"train": 20000, "test": 29000}[split]
    for i in range(n):
        f = DATA / f"cell1hz_{split}_{i:03d}.npz"
        if f.exists(): continue
        t0 = time.time(); np.savez_compressed(f, **shard(base + i)); print(split, i, round(time.time() - t0, 1), flush=True)

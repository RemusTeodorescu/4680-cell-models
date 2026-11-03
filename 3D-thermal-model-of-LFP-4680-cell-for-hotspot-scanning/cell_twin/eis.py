"""Paper 1.2 - impedance-based short detector (BMA7418_EIS_TWIN) on the same module scenarios
as the voltage GRU, so the two detectors can be compared one to one.

The cell impedance model, the BMA7418 measurement twin (fast statistical path) and the detector
are taken over unchanged from the BMA7418_EIS_TWIN package of 3 Nov 2026 (twin.py, detector.py;
README in the project: claude/paper12/bma7418-eis-twin-2026-11-03.md). What changes here is the
module: instead of that package's own first-order thermal loop, the per-cell temperature (mean of
core and can), state of charge and the short come from module_sim.py, i.e. from exactly the
scenarios the GRU is tested on (drive cycles included).

    python eis.py states      # re-simulate val/test shards, keep T, SOC every 60 s
    python eis.py eval        # calibrate on val (no-short runs), test, compare with the GRU
    python eis.py export      # detector constants for the browser twin
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

import module_sim as S

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "out"
NS, NP = 23, 9
R_GAS = 8.314

# ---------------------------------------------------------------- twin.py (BMA7418_EIS_TWIN)
CELL = dict(L=20e-9, Rs=2.5e-3, Ea_s=15e3,
            Rct_a=0.75e-3, C_a=1.1, a_a=0.90,
            Rct_c=0.75e-3, C_c=268.0, a_c=0.85,
            Ea_ct=50e3, sig_w=0.6e-3, Ea_w=30e3, T_ref=298.15)
BMA = dict(I_exc=5.0, waveform="sine", fs=8000.0, d=2.0, sig_v=50e-6, lsb_v=30e-6, gain_v=1.6e-4,
           sig_i=5e-3, gain_i=1e-3, skew_std=150e-9, C_diff=6e5)
FREQS = np.array([10.0, 100.0, 1000.0])


def _arr(Ea, T, Tref):
    return np.exp(Ea / R_GAS * (1.0 / T - 1.0 / Tref))


def z_cell(f, T, soc, scale=1.0, p=CELL):
    f = np.asarray(f, float)
    T = np.asarray(T, float)[..., None]
    soc = np.asarray(soc, float)[..., None]
    scale = np.asarray(scale, float)[..., None]
    w = 2 * np.pi * f
    socf = 1 + 0.6 * np.exp(-soc / 0.08) + 0.3 * np.exp(-(1 - soc) / 0.05)
    a_ct = _arr(p["Ea_ct"], T, p["T_ref"]) * socf
    Zs = 1j * w * p["L"] + p["Rs"] * _arr(p["Ea_s"], T, p["T_ref"])
    zarc = lambda R, C, a: R / (1 + (1j * w * R * C) ** a)
    Za = zarc(p["Rct_a"] * a_ct, p["C_a"], p["a_a"])
    Zc = zarc(p["Rct_c"] * a_ct, p["C_c"], p["a_c"])
    Zw = p["sig_w"] * _arr(p["Ea_w"], T, p["T_ref"]) * socf * (1 - 1j) / np.sqrt(w)
    return scale * (Zs + Za + Zc + Zw)


def short_share(f, f_tr0):
    f = np.asarray(f, float)
    return np.asarray(f_tr0, float)[..., None] * (0.55 + 0.45 / (1 + f / 200.0))


def z_with_short(Zc, f, R_sc, f_tr0, Z_th):
    R_sc = np.asarray(R_sc, float)[..., None]
    Z_th = np.asarray(Z_th, float)[..., None]
    Ztr = short_share(f, f_tr0) * Zc
    corr = np.where(np.isfinite(R_sc), Ztr ** 2 / (np.where(np.isfinite(R_sc), R_sc, 1.0) + Z_th), 0)
    return Zc - corr


class Channel:
    def __init__(self, n, rng, cfg=BMA):
        self.cfg, self.rng = cfg, rng
        self.gv = 1 + cfg["gain_v"] * rng.standard_normal(n)
        self.gi = 1 + cfg["gain_i"] * rng.standard_normal()

    def measure_fast(self, Z_true, f):
        c, rng = self.cfg, self.rng
        n, F = Z_true.shape
        N = int(round(c["d"] * c["fs"]))
        I1 = c["I_exc"] * (4 / np.pi if c["waveform"] == "square" else 1.0)
        skew = c["skew_std"] * rng.standard_normal((n, 1))
        Zm = Z_true * (self.gv[:, None] / self.gi) * np.exp(-1j * 2 * np.pi * f[None, :] * skew)
        sv = np.sqrt(c["sig_v"] ** 2 + c["lsb_v"] ** 2 / 12) * np.sqrt(2.0 / N)
        si = c["sig_i"] * np.sqrt(2.0 / N)
        nv = sv * (rng.standard_normal((n, F)) + 1j * rng.standard_normal((n, F)))
        ni = si * (rng.standard_normal((1, F)) + 1j * rng.standard_normal((1, F)))
        return (Zm * I1 - nv) / (I1 + ni)


# ---------------------------------------------------------------- detector.py (BMA7418_EIS_TWIN)
def detrend_groups(Z):
    K, G, F = Z.shape
    y = np.log(np.abs(Z))
    x = np.linspace(-1, 1, G)
    A = np.vstack([np.ones(G), x, x ** 2]).T
    e = np.empty_like(y)
    for k in range(K):
        for f in range(F):
            m = np.ones(G, bool)
            for it in range(2):
                c = np.linalg.lstsq(A[m], y[k, m, f], rcond=None)[0]
                r = y[k, :, f] - A @ c
                if it == 0:
                    m[np.argsort(-np.abs(r))[:2]] = False
            e[k, :, f] = r
    return e


class EISDetector:
    def __init__(self, sigma=None, thr=5.0, n_init=10, tau_scans=60, n_smooth=3, w=(1.0, 1.0, 0.5)):
        self.sigma, self.thr, self.n_init, self.tau, self.n_smooth = sigma, thr, n_init, tau_scans, n_smooth
        self.w = np.array(w)

    def residuals(self, Z):
        e = detrend_groups(Z)
        K, G, F = e.shape
        d = np.zeros_like(e)
        b = e[:self.n_init].mean(0)
        a = 1.0 / self.tau
        sig = self.sigma if self.sigma is not None else np.full(F, 1.0)
        for k in range(K):
            d[k] = e[k] - b
            if k >= self.n_init:
                s = self._score_one(d[k], sig)
                upd = s < 0.5 * self.thr
                b[upd] += a * (e[k, upd] - b[upd])
        return d

    def _score_one(self, dk, sig):
        return -(dk / sig * self.w).sum(-1) / np.sqrt((self.w ** 2).sum())

    def score(self, Z):
        d = self.residuals(Z)
        s = self._score_one(d, self.sigma)
        n = self.n_smooth
        sm = np.copy(s)
        for k in range(len(s)):
            sm[k] = s[max(0, k - n + 1):k + 1].mean(0)
        sm[:self.n_init] = 0
        return sm


# ---------------------------------------------------------------- module_sim scenarios -> BMA scans
SCAN = 60          # s between scans (all groups at once)


def scans_for_shard(f, seed):
    """Re-simulate a gen_data shard (same seed, same scenarios) keeping T and SOC every 60 s,
    then build the measured group impedances (B, K, 23, 3)."""
    rng = np.random.default_rng(seed)
    sc = S.sample_scenarios(rng, 64, 7200.0, 1.0)
    out = S.simulate(sc, rng, state_every=SCAN)
    T = out["T_state"] + 273.15; soc = out["soc_state"]
    B, K, N = T.shape
    rng2 = np.random.default_rng(seed + 777)
    Rdc = S.P["R0"] + S.P["R1"]
    Zall = np.zeros((B, K, NS, len(FREQS)), complex)
    t_scan = (np.arange(K) + 1) * SCAN
    for b in range(B):
        ch = Channel(NS, rng2)
        scale = (sc["res"][b] / S.P["R0"]).reshape(NS, NP)
        for k in range(K):
            Zc = z_cell(FREQS, T[b, k].reshape(NS, NP), soc[b, k].reshape(NS, NP), scale)
            if sc["has"][b]:
                on = S.short_on({kk: (v[b:b + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (B,) else v)
                                 for kk, v in sc.items()}, t_scan[k])[0]
                if on:
                    g, c = divmod(int(sc["k"][b]), NP)
                    R = np.full((NS, NP), np.inf); R[g, c] = sc["R"][b]
                    ftr = np.zeros((NS, NP)); ftr[g, c] = sc["f_tr"][b]
                    zth = np.zeros((NS, NP)); zth[g, c] = sc["f_th"][b] * Rdc
                    Zc = z_with_short(Zc, FREQS, R, ftr, zth)
            Zg = 1.0 / (1.0 / Zc).sum(1)
            Zall[b, k] = ch.measure_fast(Zg, FREQS)
    return Zall, t_scan


def states(split, n):
    base = {"val": 5000, "test": 9000}[split]
    for i in range(n):
        f = DATA / f"eis_{split}_{i:03d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        Z, t = scans_for_shard(f, base + i)
        np.savez_compressed(f, Z=Z.astype(np.complex64), t=t)
        print(split, i, round(time.time() - t0, 1), "s", flush=True)


def load_eis(split):
    files = sorted(glob.glob(str(DATA / f"eis_{split}_*.npz")))
    Z = np.concatenate([np.load(f)["Z"] for f in files])
    t = np.load(files[0])["t"]
    import train_det as T
    D = T.load(split, len(files))
    return Z.astype(complex), t, D


def first_alarms(det, Z, t):
    """(S, 23) first alarm time per group (NaN if none): score > thr."""
    S_, K, G, F = Z.shape
    first = np.full((S_, G), np.nan)
    scores = []
    for s in range(S_):
        sc = det.score(Z[s])
        scores.append(sc.max())
        for g in range(G):
            hit = np.nonzero(sc[:, g] > det.thr)[0]
            if len(hit):
                first[s, g] = t[hit[0]]
    return first, np.array(scores)


def evaluate(fa_target=0.02):
    Zv, t, V = load_eis("val")
    det = EISDetector(sigma=np.ones(len(FREQS)), thr=1e9)
    nos = np.nonzero(~V["has"])[0]
    d = np.concatenate([det.residuals(Zv[s])[det.n_init:].reshape(-1, len(FREQS)) for s in nos])
    sigma = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0)
    det = EISDetector(sigma=sigma, thr=1e9)
    peaks = np.array([det.score(Zv[s]).max() for s in nos])
    thr_pkg = float(peaks.max() * 1.15)                     # the package's rule: 15 % above the worst no-short run
    det.thr = thr_pkg
    gv = V["k"] // NP
    fv, _ = first_alarms(det, Zv, t)
    hours_v = len(gv) * 2.0
    fa_v = sum(1 for s in range(len(gv)) for q in range(NS)
               if np.isfinite(fv[s, q]) and not (V["has"][s] and q == gv[s] and fv[s, q] >= V["ton"][s]))
    print(f"EIS: sigma {np.round(sigma, 5)}, threshold {thr_pkg:.2f}, val false alarms {fa_v} in {hours_v:.0f} module-h", flush=True)
    Zt, t, D = load_eis("test")
    ft, _ = first_alarms(det, Zt, t)
    g = D["k"] // NP
    fa = 0; rows = []
    G = json.loads((OUT / "det_eval_combined.json").read_text())["rows"]
    gi = iter(G)
    for s in range(len(g)):
        fa += sum(1 for q in range(NS) if np.isfinite(ft[s, q]) and not (D["has"][s] and q == g[s] and ft[s, q] >= D["ton"][s]))
        if not D["has"][s]:
            continue
        gr = next(gi)
        ok = bool(np.isfinite(ft[s, g[s]]) and ft[s, g[s]] >= D["ton"][s])
        rows.append(dict(R=float(D["R"][s]), dur=float(D["dur"][s]), f_tr=float(D["f_tr"][s]),
                         eis=ok, eis_delay=float(ft[s, g[s]] - D["ton"][s]) if ok else float("nan"),
                         gru=gr["detected"], gru_delay=gr["delay"]))
    hours = len(g) * 2.0
    res = dict(sigma=sigma.tolist(), thr=thr_pkg, w=[1.0, 1.0, 0.5], n_init=10, tau_scans=60, n_smooth=3, scan_s=SCAN,
               fa_val=fa_v / hours_v, fa_test=fa / hours, rows=rows)
    (OUT / "eis_eval.json").write_text(json.dumps(res))
    print(f"EIS test: false alarms {fa / hours:.4f} per module-hour", flush=True)
    print(f"{'R':>10} {'kind':9} {'n':>3} | {'GRU found':>9} {'delay':>6} | {'EIS found':>9} {'delay':>6} | {'either':>6}")
    for lo, hi in zip([0.3, 1, 3, 10, 30, 100], [1, 3, 10, 30, 100, 3000]):
        for kind, sel in (("permanent", lambda r: r["dur"] > 1e8), ("transient", lambda r: r["dur"] < 1e8)):
            rr = [r for r in rows if lo <= r["R"] < hi and sel(r)]
            if not rr:
                continue
            md = lambda k, d: (np.median([r[d] for r in rr if r[k]]) / 60 if any(r[k] for r in rr) else float("nan"))
            print(f"{lo:>4}-{hi:<5} {kind:9} {len(rr):3d} | {100*np.mean([r['gru'] for r in rr]):8.0f}% {md('gru','gru_delay'):5.1f}m"
                  f" | {100*np.mean([r['eis'] for r in rr]):8.0f}% {md('eis','eis_delay'):5.1f}m"
                  f" | {100*np.mean([r['gru'] or r['eis'] for r in rr]):5.0f}%", flush=True)
    return res


def export():
    E = json.loads((OUT / "eis_eval.json").read_text())
    rows = E["rows"]
    def rate(lo, hi):
        rr = [r for r in rows if lo <= r["R"] < hi and r["dur"] > 1e8]
        return 100 * np.mean([r["eis"] for r in rr]) if rr else float("nan")
    W = dict(cell=CELL, bma=BMA, freqs=FREQS.tolist(), sigma=E["sigma"], thr=E["thr"], w=E["w"], n_init=E["n_init"],
             tau_scans=E["tau_scans"], n_smooth=E["n_smooth"], scan_s=E["scan_s"],
             limits=(f"Test set, permanent shorts within 2 h: {rate(0.3, 3):.0f} % found at 0.3-3 ohm, "
                     f"{rate(3, 30):.0f} % at 3-30 ohm, {rate(30, 300):.0f} % at 30-300 ohm; "
                     f"{E['fa_test']:.3f} false alarms per module-hour."))
    (HERE.parent / "module_twin" / "eis_params.json").write_text(json.dumps(W))
    print("exported EIS;", W["limits"])


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "states":
        states("val", 4); states("test", 8)
    elif what == "eval":
        evaluate()
    elif what == "export":
        export()

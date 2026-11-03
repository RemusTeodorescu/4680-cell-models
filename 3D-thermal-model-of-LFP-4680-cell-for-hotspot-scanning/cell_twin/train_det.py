"""Train the short detector (Physics AI NN), stage 1 at 1 Hz, in JAX.

    python train_det.py group      # group-level GRU on group voltages + current
    python train_det.py cell       # cell-level GRU on impedance temperatures
    python train_det.py eval       # detection statistics on the test set

Group level. One small GRU with shared weights runs on every series group. Input
per second: the group's voltage minus the median of the groups [mV], the cell
C-rate, its change. Outputs per second: logit of 'this group has (had) a short',
log10 of the short resistance, the leakage current [A].

Loss = data + physics:
  data:    weighted cross-entropy on the latched label, Huber on log10 R for the
           shorted group, and log-scaled error of the leakage current;
  physics: the predicted leakage current must explain the group's voltage
           deviation through the group equations. With i_leak(t) and the charge
           lost Q(t) = integral of i_leak, the voltage deviation of a group is
           dv = -a i_leak - b Q + c I + d (a: share of the short's resistance seen at
           the terminals, b: OCV slope over the group charge, c: resistance spread,
           d: offset). a, b, c, d are not known per group, so they are fitted by
           least squares inside the loss, and the residual is penalised: a leakage
           estimate that cannot explain the measured voltage is unphysical.

Cell level. A second small GRU on each cell's impedance temperature minus its
group mean (0.1 Hz); softmax over the 9 cells of a group gives which cell.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
NS, NP, N = 23, 9, 207
CAP = 9 * 15.4


# ------------------------------------------------------------------ data
def load(split, max_shards=None):
    files = sorted(glob.glob(str(DATA / f"{split}_*.npz")))[:max_shards]
    D = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([d[k] for d in D]) for k in D[0]}


TAU_RLS = 900.0      # s, forgetting time of the per-group observer
W10 = 10             # GRU runs every 10 s


def rls_features(dv, I):
    """Physics-based observer + features for the group GRU.

    Per group, recursive least squares (forgetting time TAU_RLS) on the group equation
        dv_g(t) = r_g Ic(t) + c_g(t) + e,
    i.e. a resistance deviation r_g times the load and an offset c_g. An internal short
    drains the group: c_g steps down at onset (the short's current through the cell
    impedance) and then drifts down (charge lost on the OCV slope).
    Features every 10 s (all minus the median of the groups, i.e. common mode removed):
      f1 change of c_g over 5 min, f2 change of c_g over 30 min, f3 mean residual,
      f4 mean |C-rate| (context), f5 median over groups of |f1| (module context).
    dv (S, T, 23) mV, I (S, T) A  ->  (S, T/10, 23, 5)."""
    dv = np.clip(np.asarray(dv, np.float64), -20, 20)
    S, Tn, G = dv.shape
    Ic = np.asarray(I, np.float64) / CAP
    lam = 1 - 1 / TAU_RLS
    th = np.zeros((S, G, 2)); Pm = np.zeros((S, G, 2, 2)); Pm[..., 0, 0] = Pm[..., 1, 1] = 100.0
    n10 = Tn // W10
    cbar = np.zeros((S, n10, G)); ebar = np.zeros((S, n10, G)); ibar = np.zeros((S, n10))
    for t in range(n10 * W10):
        x0 = Ic[:, t][:, None]                                         # regressor 1 (same for all groups)
        e = dv[:, t] - (th[..., 0] * x0 + th[..., 1])
        p00, p01, p11 = Pm[..., 0, 0], Pm[..., 0, 1], Pm[..., 1, 1]
        px0 = p00 * x0 + p01; px1 = p01 * x0 + p11
        den = lam + x0 * px0 + px1
        k0, k1 = px0 / den, px1 / den
        th[..., 0] += k0 * e; th[..., 1] += k1 * e
        Pm[..., 0, 0] = (p00 - k0 * px0) / lam
        Pm[..., 0, 1] = Pm[..., 1, 0] = (p01 - k0 * px1) / lam
        Pm[..., 1, 1] = (p11 - k1 * px1) / lam
        m = t // W10
        cbar[:, m] += th[..., 1] / W10; ebar[:, m] += e / W10; ibar[:, m] += np.abs(Ic[:, t]) / W10
    def lagdiff(c, L):
        prev = np.concatenate([np.repeat(c[:, :1], L, 1), c[:, :-L]], 1)
        return c - prev
    d1 = lagdiff(cbar, 30); d2 = lagdiff(cbar, 180)
    cm = lambda a: a - np.median(a, axis=2, keepdims=True)
    f1, f2, f3 = cm(d1), cm(d2), cm(ebar)
    f4 = np.repeat(ibar[..., None], G, 2)
    f5 = np.repeat(np.median(np.abs(d1), axis=2, keepdims=True), G, 2)
    F = np.stack([f1, f2, f3, f4, f5], -1)
    return np.clip(F, -30, 30).astype(np.float32)


def features_cached(split, D):
    f = DATA / f"feat_{split}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    F = np.concatenate([rls_features(D["dv"][s0:s0 + 64], D["I"][s0:s0 + 64]) for s0 in range(0, len(D["k"]), 64)])
    np.save(f, F)
    return F


def group_features(dv, I):
    """dv (..., T, 23) mV, I (..., T) A -> (..., T, 23, 3)"""
    Ic = I / CAP
    dIc = np.diff(Ic, axis=-1, prepend=Ic[..., :1])
    x = np.stack([np.clip(dv, -20, 20),
                  np.broadcast_to(Ic[..., None], dv.shape),
                  np.broadcast_to(dIc[..., None], dv.shape)], -1)
    return x.astype(np.float32)


def group_labels(D, T):
    """y (S, T, 23) latched label, logR (S,), g (S,) shorted group, isc (S, T)."""
    S = len(D["has"])
    t = np.arange(1, T + 1)
    g = D["k"] // NP
    y = np.zeros((S, T, NS), np.float32)
    for s in range(S):
        if D["has"][s]:
            y[s, :, g[s]] = (t >= D["ton"][s]).astype(np.float32)
    return y, np.log10(D["R"]).astype(np.float32), g


# ------------------------------------------------------------------ model
def gru_init(key, F, H, n_out):
    k1, k2, k3 = jax.random.split(key, 3)
    s = 1 / np.sqrt(H)
    return dict(Wx=jax.random.normal(k1, (F, 3 * H)) * (1 / np.sqrt(F)), Wh=jax.random.normal(k2, (H, 3 * H)) * s,
                b=jnp.zeros(3 * H), Wo=jax.random.normal(k3, (H, n_out)) * s, bo=jnp.zeros(n_out))


def gru_run(p, x, h0=None):
    """x (B, T, F) -> outputs (B, T, n_out), final h."""
    H = p["Wh"].shape[0]
    B = x.shape[0]
    h = jnp.zeros((B, H)) if h0 is None else h0
    xs = jnp.einsum("btf,fg->tbg", x, p["Wx"]) + p["b"]

    def cell(h, xg):
        hg = h @ p["Wh"]
        z = jax.nn.sigmoid(xg[:, :H] + hg[:, :H])
        r = jax.nn.sigmoid(xg[:, H:2 * H] + hg[:, H:2 * H])
        n = jnp.tanh(xg[:, 2 * H:] + r * hg[:, 2 * H:])
        h = (1 - z) * n + z * h
        return h, h
    h, hs = jax.lax.scan(cell, h, xs)
    out = jnp.einsum("tbh,ho->bto", hs, p["Wo"]) + p["bo"]
    return out, h


# ------------------------------------------------------------------ group-level training
def physics_residual(dv, ileak, Ic):
    """Residual of the best fit dv ~ -a i - b Q + c I + d (per sequence), relative to the
    best fit without the leakage terms: < 1 when the predicted leak explains part of dv."""
    Q = jnp.cumsum(ileak, axis=1) / 3600.0                       # Ah
    X = jnp.stack([ileak, Q, Ic, jnp.ones_like(Ic)], -1)         # (B, T, 4)
    sc = jnp.sqrt(jnp.mean(X ** 2, 1, keepdims=True) + 1e-6)      # column scaling for conditioning
    Xn = X / sc

    def fit(Xn):
        k = Xn.shape[-1]
        A = jnp.einsum("bti,btj->bij", Xn, Xn) / Xn.shape[1] + 1e-2 * jnp.eye(k)
        y = jnp.einsum("bti,bt->bi", Xn, dv) / Xn.shape[1]
        beta = jnp.linalg.solve(A, y[..., None])[..., 0]
        return dv - jnp.einsum("bti,bi->bt", Xn, beta)
    r = fit(Xn)
    r0 = jax.lax.stop_gradient(fit(Xn[..., 2:]))
    return jnp.mean(r ** 2, 1) / (jnp.mean(r0 ** 2, 1) + 1e-2)


def win10(a):
    """mean over 10 s windows along axis 1."""
    n = a.shape[1] // W10
    return a[:, :n * W10].reshape(a.shape[0], n, W10, *a.shape[2:]).mean(2)


def train_group(steps=2000, H=24, lam_phys=0.05, seed=0):
    D = load("train", int(__import__("os").environ.get("NSHARD", "999")))
    F = np.asarray(features_cached("train", D))                     # (S, T10, 23, 5)
    S, T10 = F.shape[:2]
    g = D["k"] // NP; logR = np.log10(D["R"]).astype(np.float32)
    isc10 = win10(D["isc"]).astype(np.float32)                      # (S, T10)
    dv10 = win10(np.clip(D["dv"], -20, 20)).astype(np.float32)       # (S, T10, 23)
    ic10 = win10(D["I"] / CAP).astype(np.float32)
    t10 = (np.arange(T10) + 1) * W10
    print("train scenarios", S, "T10", T10, flush=True)
    p = gru_init(jax.random.PRNGKey(seed), F.shape[-1], H, 3)
    sched = optax.cosine_decay_schedule(3e-3, steps, 0.05)
    opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)), 20)
    st = opt.init(p)

    def loss_fn(p, x, yb, lr, ib, mpos, dvb, icb):
        out, _ = gru_run(p, x)
        logit, rhat, lhat = out[..., 0], out[..., 1], jax.nn.softplus(out[..., 2]) * 0.1
        w = 1.0 + 4.0 * yb
        bce = jnp.mean(w * optax.sigmoid_binary_cross_entropy(logit, yb))
        mask = yb * mpos[:, None]
        reg = jnp.sum(mask * optax.huber_loss(rhat - lr[:, None], delta=0.5)) / (jnp.sum(mask) + 1)
        leak = jnp.mean((jnp.log1p(lhat / 0.01) - jnp.log1p(ib / 0.01)) ** 2)
        phys = jnp.mean(physics_residual(dvb, lhat, icb) * mpos)
        return bce + 0.3 * reg + 0.1 * leak + lam_phys * phys, (bce, reg, leak, phys)

    @jax.jit
    def update(p, st, *b):
        (l, aux), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, *b)
        upd, st = opt.update(gr, st, p)
        return optax.apply_updates(p, upd), st, l, aux

    rng = np.random.default_rng(seed)
    nsc, nother = 48, 3
    t0 = time.time(); hist = []
    for it in range(steps):
        sidx = rng.integers(0, S, nsc)
        cols = [[] for _ in range(7)]
        for s in sidx:
            gs = [g[s]] if D["has"][s] else []
            others = [q for q in rng.permutation(NS) if q not in gs][:nother + (1 - len(gs))]
            for q in gs + others:
                pos = float(D["has"][s] and q == g[s])
                for c, v in zip(cols, (F[s, :, q], pos * (t10 >= D["ton"][s]), logR[s], isc10[s] * pos, pos, dv10[s, :, q], ic10[s])):
                    c.append(v)
        batch = [jnp.asarray(np.array(c, np.float32)) for c in cols]
        p, st, l, aux = update(p, st, *batch)
        if it % 50 == 0 or it == steps - 1:
            hist.append([it, float(l)] + [float(a) for a in aux])
            print(f"it {it} loss {float(l):.4f} bce {float(aux[0]):.4f} reg {float(aux[1]):.3f} leak {float(aux[2]):.3f} "
                  f"phys {float(aux[3]):.3f}  {time.time()-t0:.0f}s", flush=True)
    np.savez(OUT / "det_group_params.npz", **{k: np.asarray(v) for k, v in p.items()})
    (OUT / "det_group_hist.json").write_text(json.dumps(hist))
    return p


# ------------------------------------------------------------------ cell-level training
def cell_features(dT):
    """dT (..., T10, 207) K -> (..., T10, 207, 2): value and change over 60 s."""
    dT = dT.astype(np.float32)
    ch = dT - np.concatenate([dT[..., :1, :].repeat(6, -2), dT[..., :-6, :]], -2)
    return np.stack([np.clip(dT, -10, 10), np.clip(ch, -5, 5)], -1)


def train_cell(steps=800, H=16, seed=1):
    D = load("train", int(__import__("os").environ.get("NSHARD", "999")))
    S = len(D["has"])
    T10 = D["dT"].shape[1]
    key = jax.random.PRNGKey(seed)
    p = gru_init(key, 2, H, 1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, steps, 0.05)))
    st = opt.init(p)
    pos = np.nonzero(D["has"])[0]

    def loss_fn(p, x, lab, m):
        # x (B, T10, 9, 2) for one group per sample -> run the 9 cells as a batch of sequences
        Bn = x.shape[0]
        out, _ = gru_run(p, x.transpose(0, 2, 1, 3).reshape(Bn * NP, T10, 2))
        logit = out[..., 0].reshape(Bn, NP, T10).transpose(0, 2, 1)       # (B, T10, 9)
        ce = -jnp.sum(jax.nn.log_softmax(logit, -1) * lab, -1)            # (B, T10)
        return jnp.sum(ce * m) / (jnp.sum(m) + 1)

    @jax.jit
    def update(p, st, x, lab, m):
        l, gr = jax.value_and_grad(loss_fn)(p, x, lab, m)
        upd, st = opt.update(gr, st, p)
        return optax.apply_updates(p, upd), st, l

    rng = np.random.default_rng(seed)
    t0 = time.time()
    t10 = (np.arange(T10) + 1) * 10
    for it in range(steps):
        sidx = rng.choice(pos, 32)
        xs, labs, ms = [], [], []
        for s in sidx:
            gq = D["k"][s] // NP; j = D["k"][s] % NP
            xs.append(cell_features(D["dT"][s, :, gq * NP:(gq + 1) * NP]))
            lab = np.zeros((T10, NP), np.float32); lab[:, j] = 1
            labs.append(lab)
            ms.append((t10 >= D["ton"][s] + 60).astype(np.float32))
        p, st, l = update(p, st, jnp.asarray(np.array(xs)), jnp.asarray(np.array(labs)), jnp.asarray(np.array(ms)))
        if it % 50 == 0 or it == steps - 1:
            print(f"cell it {it} ce {float(l):.4f}  {time.time()-t0:.0f}s", flush=True)
    np.savez(OUT / "det_cell_params.npz", **{k: np.asarray(v) for k, v in p.items()})
    return p


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "group"
    if what == "group":
        train_group(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    elif what == "cell":
        train_cell(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 800)


# ------------------------------------------------------------------ evaluation and export
def run_group(p, D, split):
    """p_short (S, T10, 23), log10 R (S, T10, 23) over full scenarios, all groups, every 10 s."""
    F = np.asarray(features_cached(split, D))
    S, T10 = F.shape[:2]
    f = jax.jit(lambda x: gru_run(p, x)[0])
    P_, R_ = np.zeros((S, T10, NS), np.float32), np.zeros((S, T10, NS), np.float32)
    for s0 in range(0, S, 32):
        xb = F[s0:s0 + 32].transpose(0, 2, 1, 3).reshape(-1, T10, F.shape[-1])
        o = np.asarray(f(jnp.asarray(xb))).reshape(-1, NS, T10, 3).transpose(0, 2, 1, 3)
        P_[s0:s0 + 32] = 1 / (1 + np.exp(-o[..., 0])); R_[s0:s0 + 32] = o[..., 1]
    return P_, R_


def alarms(P_, thr, K):
    """First alarm time per (scenario, group), NaN if none: p > thr for K consecutive seconds."""
    S, T, G = P_.shape
    above = P_ > thr
    run = np.zeros((S, G), int); first = np.full((S, G), np.nan)
    for t in range(T):
        run = np.where(above[:, t], run + 1, 0)
        hit = (run >= K) & np.isnan(first)
        first[hit] = (t + 1) * W10
    return first


def choose_threshold(p, K=3, fa_target=0.02):
    V = load("val")
    P_, _ = run_group(p, V, "val")
    V = {k: V[k] for k in ("k", "has", "ton")}
    g = V["k"] // NP
    T = P_.shape[1] * W10 / 3600.0
    for thr in np.arange(0.5, 0.99, 0.02):
        first = alarms(P_, thr, K)
        fa = 0
        for s in range(len(g)):
            for q in range(NS):
                if np.isfinite(first[s, q]) and not (V["has"][s] and q == g[s] and first[s, q] >= V["ton"][s]):
                    fa += 1
        rate = fa / (len(g) * T)
        if rate <= fa_target:
            return float(thr), rate
    return 0.98, rate


def evaluate():
    pg = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_group_params.npz").items()}
    pc = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_cell_params.npz").items()}
    K = 3
    thr, fa_val = choose_threshold(pg, K)
    print("threshold", thr, "false alarms per module-hour on val", fa_val, flush=True)
    D = load("test")
    P_, R_ = run_group(pg, D, "test")
    first = alarms(P_, thr, K)
    g = D["k"] // NP
    S, T = len(g), P_.shape[1] * W10
    rows = []
    fa = 0
    # cell level

    fc = jax.jit(lambda x: gru_run(pc, x)[0])
    for s in range(S):
        wrong = [q for q in range(NS) if np.isfinite(first[s, q]) and not (D["has"][s] and q == g[s] and first[s, q] >= D["ton"][s])]
        fa += len(wrong)
        if not D["has"][s]:
            continue
        det = np.isfinite(first[s, g[s]]) and first[s, g[s]] >= D["ton"][s]
        delay = first[s, g[s]] - D["ton"][s] if det else np.nan
        Rhat = 10 ** R_[s, -1, g[s]]
        # cell: probability of the true cell at the end
        xc = cell_features(D["dT"][s, :, g[s] * NP:(g[s] + 1) * NP]).transpose(1, 0, 2)
        lc = np.asarray(fc(jnp.asarray(xc)))[:, -1, 0]
        pcell = np.exp(lc - lc.max()); pcell /= pcell.sum()
        j = D["k"][s] % NP
        rows.append(dict(R=float(D["R"][s]), dur=float(D["dur"][s]), inter=bool(D["inter"][s]), ton=float(D["ton"][s]),
                         f_tr=float(D["f_tr"][s]), detected=bool(det), delay=float(delay), Rhat=float(Rhat),
                         cell_ok=bool(np.argmax(pcell) == j), p_cell=float(pcell[j]), p_best=float(pcell.max())))
    fa_rate = fa / (S * T / 3600)
    res = dict(threshold=thr, k_consec=K, fa_per_module_hour_val=fa_val, fa_per_module_hour_test=fa_rate, rows=rows)
    (OUT / "det_eval.json").write_text(json.dumps(res))
    # summary table
    bins = [0.3, 1, 3, 10, 30, 100, 300, 3000]
    print("false alarms per module-hour (test):", round(fa_rate, 4))
    for lo, hi in zip(bins[:-1], bins[1:]):
        for kind, sel in (("permanent", lambda r: r["dur"] > 1e8), ("transient", lambda r: r["dur"] < 1e8)):
            rr = [r for r in rows if lo <= r["R"] < hi and sel(r)]
            if not rr:
                continue
            d = [r for r in rr if r["detected"]]
            print(f"R {lo:>5}-{hi:<5} {kind:9s} n={len(rr):3d} detected {100*len(d)/len(rr):5.1f} %  "
                  f"median delay {np.median([r['delay'] for r in d]) if d else float('nan'):7.0f} s  "
                  f"cell right {100*np.mean([r['cell_ok'] for r in d]) if d else float('nan'):5.1f} %", flush=True)
    return res


def export():
    """Weights and alarm rule as JSON for the browser twin (module_twin/detector_weights.json)."""
    pg = dict(np.load(OUT / "det_group_params.npz")); pc = dict(np.load(OUT / "det_cell_params.npz"))
    E = json.loads((OUT / "det_eval.json").read_text())
    rows = E["rows"]
    def rate(lo, hi):
        rr = [r for r in rows if lo <= r["R"] < hi and r["dur"] > 1e8]
        return 100 * np.mean([r["detected"] for r in rr]) if rr else float("nan")
    limits = (f"Test set, permanent shorts within 2 h: {rate(0.3, 3):.0f} % found at 0.3-3 ohm, {rate(3, 30):.0f} % at 3-30 ohm, "
              f"{rate(30, 300):.0f} % at 30-300 ohm; {E['fa_per_module_hour_test']:.3f} false alarms per module-hour.")
    W = dict(group=dict(params={k: v.tolist() for k, v in pg.items()}, thr=E["threshold"], k_consec=E["k_consec"], limits=limits,
                        tau_rls=TAU_RLS, w10=W10),
             cell=dict(params={k: v.tolist() for k, v in pc.items()}, p_min=0.5))
    (HERE.parent / "module_twin" / "detector_weights.json").write_text(json.dumps(W))
    print("exported;", limits)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("eval", "export"):
    evaluate() if sys.argv[1] == "eval" else export()


# ------------------------------------------------------------------ 1 Hz version (every second, one NPU call)
def rls_features_1hz(dv, I):
    """Observer (same RLS as rls_features) with features every second:
      f1 offset change over 5 min, f2 over 30 min, f3 residual this second, f4 |C-rate|,
      f5 change of C-rate this second, f6 median over groups of |f1|, f7 median over groups of |f3|.
    f1, f2, f3 minus the median of the groups.  (S, T, 23) -> (S, T, 23, 7)."""
    dv = np.clip(np.asarray(dv, np.float64), -20, 20)
    S, Tn, G = dv.shape
    Ic = np.asarray(I, np.float64) / CAP
    lam = 1 - 1 / TAU_RLS
    th = np.zeros((S, G, 2)); Pm = np.zeros((S, G, 2, 2)); Pm[..., 0, 0] = Pm[..., 1, 1] = 100.0
    C = np.zeros((S, Tn, G), np.float32); E = np.zeros((S, Tn, G), np.float32)
    for t in range(Tn):
        x0 = Ic[:, t][:, None]
        e = dv[:, t] - (th[..., 0] * x0 + th[..., 1])
        p00, p01, p11 = Pm[..., 0, 0], Pm[..., 0, 1], Pm[..., 1, 1]
        px0 = p00 * x0 + p01; px1 = p01 * x0 + p11
        den = lam + x0 * px0 + px1
        k0, k1 = px0 / den, px1 / den
        th[..., 0] += k0 * e; th[..., 1] += k1 * e
        Pm[..., 0, 0] = (p00 - k0 * px0) / lam
        Pm[..., 0, 1] = Pm[..., 1, 0] = (p01 - k0 * px1) / lam
        Pm[..., 1, 1] = (p11 - k1 * px1) / lam
        C[:, t] = th[..., 1]; E[:, t] = e
    def lagdiff(c, L):
        return c - np.concatenate([np.repeat(c[:, :1], L, 1), c[:, :-L]], 1)
    cm = lambda a: a - np.median(a, axis=2, keepdims=True)
    d1, d2 = lagdiff(C, 300), lagdiff(C, 1800)
    f1, f2, f3 = cm(d1), cm(d2), cm(E)
    dIc = np.diff(Ic, axis=1, prepend=Ic[:, :1])
    f4 = np.repeat(np.abs(Ic)[..., None], G, 2); f5 = np.repeat(dIc[..., None], G, 2)
    f6 = np.repeat(np.median(np.abs(d1), axis=2, keepdims=True), G, 2)
    f7 = np.repeat(np.median(np.abs(E), axis=2, keepdims=True), G, 2)
    F = np.stack([f1, f2, f3, f4, f5, f6, f7], -1)
    return np.clip(F, -30, 30).astype(np.float32)


def train_group_1hz(steps=2000, H=32, lam_phys=0.05, seed=0, Lwin=3600):
    D = load("train", int(__import__("os").environ.get("NSHARD", "999")))
    S, T = D["dv"].shape[:2]
    g = D["k"] // NP; logR = np.log10(D["R"]).astype(np.float32)
    tt = np.arange(1, T + 1)
    print("1 Hz: train scenarios", S, "T", T, flush=True)
    p = gru_init(jax.random.PRNGKey(seed), 7, H, 3)
    opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0),
                                            optax.adam(optax.cosine_decay_schedule(3e-3, steps, 0.05))), 20)
    st = opt.init(p)

    def loss_fn(p, x, yb, lr, ib, mpos, dvb, icb):
        out, _ = gru_run(p, x)
        logit, rhat, lhat = out[..., 0], out[..., 1], jax.nn.softplus(out[..., 2]) * 0.1
        w = 1.0 + 4.0 * yb
        bce = jnp.mean(w * optax.sigmoid_binary_cross_entropy(logit, yb))
        mask = yb * mpos[:, None]
        reg = jnp.sum(mask * optax.huber_loss(rhat - lr[:, None], delta=0.5)) / (jnp.sum(mask) + 1)
        leak = jnp.mean((jnp.log1p(lhat / 0.01) - jnp.log1p(ib / 0.01)) ** 2)
        phys = jnp.mean(physics_residual(dvb, lhat, icb) * mpos)
        return bce + 0.3 * reg + 0.1 * leak + lam_phys * phys, (bce, reg, leak, phys)

    @jax.jit
    def update(p, st, *b):
        (l, aux), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, *b)
        upd, st = opt.update(gr, st, p)
        return optax.apply_updates(p, upd), st, l, aux

    rng = np.random.default_rng(seed)
    nsc, nother = 32, 3
    t0 = time.time(); hist = []
    for it in range(steps):
        sidx = rng.integers(0, S, nsc)
        F = rls_features_1hz(D["dv"][sidx], D["I"][sidx])          # whole 2 h, observer warmed up from t = 0
        t_s = int(rng.integers(0, T - Lwin + 1))
        cols = [[] for _ in range(7)]
        for b, s in enumerate(sidx):
            gs = [g[s]] if D["has"][s] else []
            others = [q for q in rng.permutation(NS) if q not in gs][:nother + (1 - len(gs))]
            for q in gs + others:
                pos = float(D["has"][s] and q == g[s])
                sl = slice(t_s, t_s + Lwin)
                for c, v in zip(cols, (F[b, sl, q], pos * (tt[sl] >= D["ton"][s]), logR[s], D["isc"][s, sl] * pos, pos,
                                       np.clip(D["dv"][s, sl, q], -20, 20), D["I"][s, sl] / CAP)):
                    c.append(v)
        batch = [jnp.asarray(np.array(c, np.float32)) for c in cols]
        p, st, l, aux = update(p, st, *batch)
        if it % 50 == 0 or it == steps - 1:
            hist.append([it, float(l)] + [float(a) for a in aux])
            print(f"1hz it {it} loss {float(l):.4f} bce {float(aux[0]):.4f} reg {float(aux[1]):.3f} leak {float(aux[2]):.3f} "
                  f"phys {float(aux[3]):.3f}  {time.time()-t0:.0f}s", flush=True)
    np.savez(OUT / "det_group1hz_params.npz", **{k: np.asarray(v) for k, v in p.items()})
    (OUT / "det_group1hz_hist.json").write_text(json.dumps(hist))
    return p


def evaluate_1hz(K=5, fa_target=0.02):
    """Same metrics as evaluate(), for the 1 Hz group network (alarm: K consecutive seconds above threshold)."""
    pg = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_group1hz_params.npz").items()}
    f = jax.jit(lambda x: gru_run(pg, x)[0])
    def run(D):
        S, T = D["dv"].shape[:2]
        P_ = np.zeros((S, T, NS), np.float32); R_ = np.zeros((S, T, NS), np.float32)
        for s0 in range(0, S, 8):
            F = rls_features_1hz(D["dv"][s0:s0 + 8], D["I"][s0:s0 + 8])
            o = np.asarray(f(jnp.asarray(F.transpose(0, 2, 1, 3).reshape(-1, T, 7)))).reshape(-1, NS, T, 3).transpose(0, 2, 1, 3)
            P_[s0:s0 + 8] = 1 / (1 + np.exp(-o[..., 0])); R_[s0:s0 + 8] = o[..., 1]
        return P_, R_
    def first_alarm_k(P_, thr, K_):
        S, T, G = P_.shape
        run_ = np.zeros((S, G), int); first = np.full((S, G), np.nan)
        for t in range(T):
            run_ = np.where(P_[:, t] > thr, run_ + 1, 0)
            hit = (run_ >= K_) & np.isnan(first); first[hit] = t + 1
        return first
    first_alarm = lambda P_, thr: first_alarm_k(P_, thr, K)
    V = load("val"); Pv, _ = run(V); gv = V["k"] // NP
    hours = len(gv) * Pv.shape[1] / 3600
    # alarm rule: K consecutive seconds above thr; pick (K, thr) on the validation set with
    # false alarms <= fa_target that finds the most shorts (ties: shortest K)
    best = None
    for K_ in (5, 15, 30, 60, 120):
        for th in (0.9, 0.95, 0.98, 0.99, 0.995, 0.998):
            first = first_alarm_k(Pv, th, K_)
            fa = sum(1 for s in range(len(gv)) for q in range(NS)
                     if np.isfinite(first[s, q]) and not (V["has"][s] and q == gv[s] and first[s, q] >= V["ton"][s]))
            det = sum(1 for s in range(len(gv)) if V["has"][s] and np.isfinite(first[s, gv[s]]) and first[s, gv[s]] >= V["ton"][s])
            print(f"  val K={K_:4d} thr={th:.3f} false alarms/h {fa/hours:.4f} found {det}", flush=True)
            if fa / hours <= fa_target and (best is None or det > best[2]):
                best = (K_, th, det)
    K, thr = (best[0], float(best[1])) if best else (120, 0.998)
    del V, Pv
    D = load("test"); P_, R_ = run(D); g = D["k"] // NP
    first = first_alarm_k(P_, thr, K)
    rows, fa = [], 0
    for s in range(len(g)):
        fa += sum(1 for q in range(NS) if np.isfinite(first[s, q]) and not (D["has"][s] and q == g[s] and first[s, q] >= D["ton"][s]))
        if not D["has"][s]:
            continue
        det = bool(np.isfinite(first[s, g[s]]) and first[s, g[s]] >= D["ton"][s])
        rows.append(dict(R=float(D["R"][s]), dur=float(D["dur"][s]), inter=bool(D["inter"][s]), detected=det,
                         delay=float(first[s, g[s]] - D["ton"][s]) if det else float("nan"),
                         Rhat=float(10 ** R_[s, -1, g[s]])))
    res = dict(threshold=thr, k_consec=K, fa_per_module_hour_test=fa / (len(g) * P_.shape[1] / 3600), rows=rows)
    (OUT / "det_eval_1hz.json").write_text(json.dumps(res))
    print("1 Hz rule: K", K, "threshold", thr, "false alarms per module-hour (test):", round(res["fa_per_module_hour_test"], 4))
    bins = [0.3, 1, 3, 10, 30, 100, 300, 3000]
    for lo, hi in zip(bins[:-1], bins[1:]):
        for kind, sel in (("permanent", lambda r: r["dur"] > 1e8), ("transient", lambda r: r["dur"] < 1e8)):
            rr = [r for r in rows if lo <= r["R"] < hi and sel(r)]
            if rr:
                d = [r for r in rr if r["detected"]]
                print(f"R {lo:>5}-{hi:<5} {kind:9s} n={len(rr):3d} detected {100*len(d)/len(rr):5.1f} %  "
                      f"median delay {np.median([r['delay'] for r in d]) if d else float('nan'):7.0f} s", flush=True)
    return res


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("group1hz", "eval1hz"):
    train_group_1hz(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 2000) if sys.argv[1] == "group1hz" else evaluate_1hz()


# ------------------------------------------------------------------ 1 Hz cell-level network
def cell_features_1hz(t1):
    """t1 (..., T, 9) impedance temperature every second -> (..., T, 9, 2):
    10 s moving average minus its group mean, and its change over 60 s."""
    t1 = np.asarray(t1, np.float32)
    cs = np.cumsum(np.concatenate([np.zeros_like(t1[..., :1, :]), t1], -2), -2)
    T = t1.shape[-2]
    idx = np.arange(T)
    lo = np.maximum(idx - 9, 0)
    m10 = (cs[..., idx + 1, :] - cs[..., lo, :]) / (idx - lo + 1)[:, None]
    x0 = m10 - m10.mean(-1, keepdims=True)
    prev = np.concatenate([np.repeat(x0[..., :1, :], 60, -2), x0[..., :-60, :]], -2)
    return np.stack([np.clip(x0, -10, 10), np.clip(x0 - prev, -5, 5)], -1)


def load_cell1hz(split):
    files = sorted(glob.glob(str(DATA / f"cell1hz_{split}_*.npz")))
    Dl = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([d[k] for d in Dl]) for k in Dl[0]}


def train_cell_1hz(steps=800, H=16, seed=1, Lwin=3600):
    D = load_cell1hz("train")
    pos = np.nonzero(D["has"])[0]
    T = D["timp1"].shape[1]
    tt = np.arange(1, T + 1)
    p = gru_init(jax.random.PRNGKey(seed), 2, H, 1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, steps, 0.05)))
    st = opt.init(p)

    def loss_fn(p, x, lab, m):
        Bn, L = x.shape[0], x.shape[1]
        out, _ = gru_run(p, x.transpose(0, 2, 1, 3).reshape(Bn * NP, L, 2))
        logit = out[..., 0].reshape(Bn, NP, L).transpose(0, 2, 1)
        ce = -jnp.sum(jax.nn.log_softmax(logit, -1) * lab, -1)
        return jnp.sum(ce * m) / (jnp.sum(m) + 1)

    @jax.jit
    def update(p, st, x, lab, m):
        l, gr = jax.value_and_grad(loss_fn)(p, x, lab, m)
        upd, st = opt.update(gr, st, p)
        return optax.apply_updates(p, upd), st, l

    rng = np.random.default_rng(seed); t0 = time.time()
    for it in range(steps):
        sidx = rng.choice(pos, 32)
        t_s = int(rng.integers(0, T - Lwin + 1)); sl = slice(t_s, t_s + Lwin)
        X = cell_features_1hz(D["timp1"][sidx, :, 0, :])[:, sl]          # group 0 of 'keep' = the shorted group
        lab = np.zeros((32, Lwin, NP), np.float32); lab[np.arange(32), :, D["k"][sidx] % NP] = 1
        m = np.array([(tt[sl] >= D["ton"][s] + 60) for s in sidx], np.float32)
        p, st, l = update(p, st, jnp.asarray(X), jnp.asarray(lab), jnp.asarray(m))
        if it % 50 == 0 or it == steps - 1:
            print(f"cell1hz it {it} ce {float(l):.4f}  {time.time()-t0:.0f}s", flush=True)
    np.savez(OUT / "det_cell1hz_params.npz", **{k: np.asarray(v) for k, v in p.items()})
    # test: right cell at the end of the scenario, by short resistance
    Dt = load_cell1hz("test")
    f = jax.jit(lambda x: gru_run(p, x)[0])
    rows = []
    for s in np.nonzero(Dt["has"])[0]:
        X = cell_features_1hz(Dt["timp1"][s, :, 0, :])                   # (T, 9, 2)
        lg = np.asarray(f(jnp.asarray(X.transpose(1, 0, 2))))[:, -1, 0]
        pc = np.exp(lg - lg.max()); pc /= pc.sum()
        rows.append(dict(R=float(Dt["R"][s]), dur=float(Dt["dur"][s]), ok=bool(np.argmax(pc) == Dt["k"][s] % NP), p=float(pc.max())))
    (OUT / "det_cell1hz_eval.json").write_text(json.dumps(rows))
    for lo, hi in zip([0.3, 1, 3, 10, 30], [1, 3, 10, 30, 3000]):
        rr = [r for r in rows if lo <= r["R"] < hi and r["dur"] > 1e8]
        if rr:
            print(f"cell 1 Hz, permanent R {lo}-{hi}: right cell {100*np.mean([r['ok'] for r in rr]):.0f} % (n={len(rr)})", flush=True)
    return p


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cell1hz":
    train_cell_1hz(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 800)


def export_1hz():
    pg = dict(np.load(OUT / "det_group1hz_params.npz")); pc = dict(np.load(OUT / "det_cell1hz_params.npz"))
    E = json.loads((OUT / "det_eval_1hz.json").read_text()); rows = E["rows"]
    def rate(lo, hi):
        rr = [r for r in rows if lo <= r["R"] < hi and r["dur"] > 1e8]
        return 100 * np.mean([r["detected"] for r in rr]) if rr else float("nan")
    limits = (f"Test set, permanent shorts within 2 h: {rate(0.3, 3):.0f} % found at 0.3-3 ohm, {rate(3, 30):.0f} % at 3-30 ohm, "
              f"{rate(30, 300):.0f} % at 30-300 ohm; {E['fa_per_module_hour_test']:.3f} false alarms per module-hour.")
    W = dict(version="1hz",
             group=dict(params={k: v.tolist() for k, v in pg.items()}, thr=E["threshold"], k_consec=E["k_consec"], limits=limits, tau_rls=TAU_RLS),
             cell=dict(params={k: v.tolist() for k, v in pc.items()}, p_min=0.5))
    (HERE.parent / "module_twin" / "detector_weights.json").write_text(json.dumps(W))
    print("exported 1 Hz;", limits)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "export1hz":
    export_1hz()


# ------------------------------------------------------------------ combined detector: fast path (1 Hz) + slow path (10 s)
def _first(P_, thr, K_, step_s):
    S, T, G = P_.shape
    run_ = np.zeros((S, G), int); first = np.full((S, G), np.nan)
    for t in range(T):
        run_ = np.where(P_[:, t] > thr, run_ + 1, 0)
        hit = (run_ >= K_) & np.isnan(first); first[hit] = (t + 1) * step_s
    return first


def _run_1hz(pg, D):
    f = jax.jit(lambda x: gru_run(pg, x)[0])
    S, T = D["dv"].shape[:2]
    P_ = np.zeros((S, T, NS), np.float32)
    for s0 in range(0, S, 8):
        F = rls_features_1hz(D["dv"][s0:s0 + 8], D["I"][s0:s0 + 8])
        o = np.asarray(f(jnp.asarray(F.transpose(0, 2, 1, 3).reshape(-1, T, 7)))).reshape(-1, NS, T, 3).transpose(0, 2, 1, 3)
        P_[s0:s0 + 8] = 1 / (1 + np.exp(-o[..., 0]))
    return P_


def _fa_and_found(first, D):
    g = D["k"] // NP
    fa = sum(1 for s in range(len(g)) for q in range(NS)
             if np.isfinite(first[s, q]) and not (D["has"][s] and q == g[s] and first[s, q] >= D["ton"][s]))
    found = sum(1 for s in range(len(g)) if D["has"][s] and np.isfinite(first[s, g[s]]) and first[s, g[s]] >= D["ton"][s])
    return fa, found


def evaluate_combined(fa_total=0.02):
    """Alarm = slow path OR fast path. The rules are chosen together on the validation set:
    the most shorts found with at most fa_total false alarms per module-hour in total."""
    p10 = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_group_params.npz").items()}
    p1 = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_group1hz_params.npz").items()}
    V = load("val")
    P10v, _ = run_group(p10, V, "val"); P1v = _run_1hz(p1, V)
    hours = len(V["k"]) * V["dv"].shape[1] / 3600
    slow_rules = [(K, th) for K in (3, 6) for th in (0.95, 0.98, 0.99)]
    fast_rules = [(K, th) for K in (5, 30, 60, 120) for th in (0.98, 0.99, 0.995)]
    firsts10 = {r: _first(P10v, r[1], r[0], W10) for r in slow_rules}
    firsts1 = {r: _first(P1v, r[1], r[0], 1) for r in fast_rules}
    best = None
    for rs, f10 in firsts10.items():
        for rf, f1 in firsts1.items():
            first = np.fmin(f10, f1)
            fa, found = _fa_and_found(first, V)
            if fa / hours <= fa_total and (best is None or found > best[2]):
                best = (rs, rf, found, fa / hours)
    print("combined rule (slow K, thr), (fast K, thr):", best, flush=True)
    rs, rf = best[0], best[1]
    del V, P10v, P1v
    D = load("test")
    P10, R10 = run_group(p10, D, "test"); P1 = _run_1hz(p1, D)
    f10 = _first(P10, rs[1], rs[0], W10); f1 = _first(P1, rf[1], rf[0], 1)
    first = np.fmin(f10, f1)
    g = D["k"] // NP
    fa, _ = _fa_and_found(first, D)
    rows = []
    for s in range(len(g)):
        if not D["has"][s]:
            continue
        ok = np.isfinite(first[s, g[s]]) and first[s, g[s]] >= D["ton"][s]
        by = "fast" if ok and np.isfinite(f1[s, g[s]]) and f1[s, g[s]] == first[s, g[s]] else ("slow" if ok else "")
        rows.append(dict(R=float(D["R"][s]), dur=float(D["dur"][s]), detected=bool(ok),
                         delay=float(first[s, g[s]] - D["ton"][s]) if ok else float("nan"), by=by,
                         Rhat=float(10 ** R10[s, -1, g[s]])))
    res = dict(slow=dict(k_consec=rs[0], thr=rs[1]), fast=dict(k_consec=rf[0], thr=rf[1]),
               fa_per_module_hour_test=fa / (len(g) * D["dv"].shape[1] / 3600), rows=rows)
    (OUT / "det_eval_combined.json").write_text(json.dumps(res))
    print("false alarms per module-hour (test):", round(res["fa_per_module_hour_test"], 4))
    for lo, hi in zip([0.3, 1, 3, 10, 30], [1, 3, 10, 30, 3000]):
        for kind, sel in (("permanent", lambda r: r["dur"] > 1e8), ("transient", lambda r: r["dur"] < 1e8)):
            rr = [r for r in rows if lo <= r["R"] < hi and sel(r)]
            if rr:
                d = [r for r in rr if r["detected"]]
                print(f"R {lo:>4}-{hi:<5} {kind:9s} n={len(rr):3d} found {100*len(d)/len(rr):5.1f} %  median delay "
                      f"{np.median([r['delay'] for r in d]) if d else float('nan'):6.0f} s  by fast path {sum(r['by']=='fast' for r in d)}", flush=True)
    return res


def export_combined():
    p10 = dict(np.load(OUT / "det_group_params.npz")); p1 = dict(np.load(OUT / "det_group1hz_params.npz"))
    pc = dict(np.load(OUT / "det_cell1hz_params.npz"))
    E = json.loads((OUT / "det_eval_combined.json").read_text()); rows = E["rows"]
    def rate(lo, hi):
        rr = [r for r in rows if lo <= r["R"] < hi and r["dur"] > 1e8]
        return 100 * np.mean([r["detected"] for r in rr]) if rr else float("nan")
    limits = (f"Test set, permanent shorts within 2 h: {rate(0.3, 3):.0f} % found at 0.3-3 ohm, {rate(3, 30):.0f} % at 3-30 ohm, "
              f"{rate(30, 300):.0f} % at 30-300 ohm; {E['fa_per_module_hour_test']:.3f} false alarms per module-hour.")
    W = dict(version="combined", tau_rls=TAU_RLS, limits=limits,
             slow=dict(params={k: v.tolist() for k, v in p10.items()}, **E["slow"], w10=W10),
             fast=dict(params={k: v.tolist() for k, v in p1.items()}, **E["fast"]),
             cell=dict(params={k: v.tolist() for k, v in pc.items()}, p_min=0.2))
    (HERE.parent / "module_twin" / "detector_weights.json").write_text(json.dumps(W))
    print("exported combined;", limits)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("evalcomb", "exportcomb"):
    evaluate_combined() if sys.argv[1] == "evalcomb" else export_combined()

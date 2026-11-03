"""One two-rate GRU for the group detector (replaces the slow 10 s + fast 1 Hz pair).

One network, two clocks (clockwork-style):
  slow cell  GRU, H=24, steps every 10 s on the 10 s observer features (rls_features):
             averaging over 10 s cuts the noise ~3x and 720 steps per 2 h make long trends easy to learn;
  fast cell  GRU, H=32, steps every second on the 1 Hz observer features (rls_features_1hz)
             plus the last completed slow state (held for 10 s, causal);
  one head   on the fast state, every second: logit(short in this group), log10 R, leakage.
One output, one alarm rule (K consecutive seconds above a threshold).

Warm start: slow cell from det_group_params, fast cell from det_group1hz_params (the rows of Wx for
the slow state start at zero, so at step 0 the network is the 1 Hz detector).
Loss: the 1 Hz loss of train_det (data + physics residual) on the head, plus 0.3 x BCE of the slow
cell's own head on 10 s labels so that the slow state stays a short detector.

    python train_mr.py train [steps]
    python train_mr.py eval
    python train_mr.py export
"""
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from train_det import (load, features_cached, rls_features_1hz, gru_run, physics_residual, win10, _first,
                       _fa_and_found, OUT, HERE, NS, NP, CAP, W10, TAU_RLS)

HS, HF = 24, 32


def init():
    p10 = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_group_params.npz").items()}
    p1 = {k: jnp.asarray(v) for k, v in np.load(OUT / "det_group1hz_params.npz").items()}
    Wx = jnp.concatenate([p1["Wx"], jnp.zeros((HS, 3 * HF))], 0)
    return dict(s=p10, f=dict(Wx=Wx, Wh=p1["Wh"], b=p1["b"], Wo=p1["Wo"], bo=p1["bo"]))


def slow_states(ps, x10):
    """x10 (B, T10, 5) -> slow outputs (B, T10, 3), states (B, T10, HS)."""
    H = ps["Wh"].shape[0]
    B = x10.shape[0]
    xs = jnp.einsum("btf,fg->tbg", x10, ps["Wx"]) + ps["b"]

    def cell(h, xg):
        hg = h @ ps["Wh"]
        z = jax.nn.sigmoid(xg[:, :H] + hg[:, :H])
        r = jax.nn.sigmoid(xg[:, H:2 * H] + hg[:, H:2 * H])
        n = jnp.tanh(xg[:, 2 * H:] + r * hg[:, 2 * H:])
        h = (1 - z) * n + z * h
        return h, h
    _, hs = jax.lax.scan(cell, jnp.zeros((B, H)), xs)
    hs = hs.transpose(1, 0, 2)
    return jnp.einsum("bth,ho->bto", hs, ps["Wo"]) + ps["bo"], hs


def held(hs, t0, L):
    """slow state available at seconds t0+1 .. t0+L (window m done at second 10(m+1))."""
    m = (t0 + jnp.arange(L) + 1) // W10 - 1
    h = jnp.take(hs, jnp.clip(m, 0, hs.shape[1] - 1), axis=1)
    return jnp.where((m >= 0)[None, :, None], h, 0.0)


def forward(p, x1, x10, t0):
    """x1 (B, L, 7) seconds t0+1..t0+L, x10 (B, T10, 5) from t = 0 -> fast out (B, L, 3), slow out (B, T10, 3)."""
    so, hs = slow_states(p["s"], x10)
    x = jnp.concatenate([x1, held(hs, t0, x1.shape[1])], -1)
    out, _ = gru_run(p["f"], x)
    return out, so


def train(steps=1500, lam_phys=0.05, seed=3, Lwin=3600):
    D = load("train")
    F10 = features_cached("train", D)
    S, T = D["dv"].shape[:2]
    T10 = T // W10
    g = D["k"] // NP; logR = np.log10(D["R"]).astype(np.float32)
    tt = np.arange(1, T + 1); t10 = (np.arange(T10) + 1) * W10
    p = init()
    opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0),
                                            optax.adam(optax.cosine_decay_schedule(1.5e-3, steps, 0.05))), 20)
    st = opt.init(p)

    def loss_fn(p, x1, x10, t0, yb, y10, lr, ib, mpos, dvb, icb):
        out, so = forward(p, x1, x10, t0)
        logit, rhat, lhat = out[..., 0], out[..., 1], jax.nn.softplus(out[..., 2]) * 0.1
        w = 1.0 + 4.0 * yb
        bce = jnp.mean(w * optax.sigmoid_binary_cross_entropy(logit, yb))
        mask = yb * mpos[:, None]
        reg = jnp.sum(mask * optax.huber_loss(rhat - lr[:, None], delta=0.5)) / (jnp.sum(mask) + 1)
        leak = jnp.mean((jnp.log1p(lhat / 0.01) - jnp.log1p(ib / 0.01)) ** 2)
        phys = jnp.mean(physics_residual(dvb, lhat, icb) * mpos)
        bce10 = jnp.mean((1.0 + 4.0 * y10) * optax.sigmoid_binary_cross_entropy(so[..., 0], y10))
        return bce + 0.3 * reg + 0.1 * leak + lam_phys * phys + 0.3 * bce10, (bce, reg, leak, phys, bce10)

    @jax.jit
    def update(p, st, x1, x10, t0, *b):
        (l, aux), gr = jax.value_and_grad(loss_fn, has_aux=True)(p, x1, x10, t0, *b)
        upd, st = opt.update(gr, st, p)
        return optax.apply_updates(p, upd), st, l, aux

    rng = np.random.default_rng(seed)
    nsc, nother = 32, 3
    t_0 = time.time(); hist = []
    for it in range(steps):
        sidx = np.sort(rng.integers(0, S, nsc))
        F1 = rls_features_1hz(D["dv"][sidx], D["I"][sidx])
        t_s = int(rng.integers(0, (T - Lwin) // W10 + 1)) * W10      # window start on a 10 s boundary
        sl = slice(t_s, t_s + Lwin)
        cols = [[] for _ in range(9)]
        for b, s in enumerate(sidx):
            gs = [g[s]] if D["has"][s] else []
            others = [q for q in rng.permutation(NS) if q not in gs][:nother + (1 - len(gs))]
            for q in gs + others:
                pos = float(D["has"][s] and q == g[s])
                for c, v in zip(cols, (F1[b, sl, q], np.asarray(F10[s, :, q]), pos * (tt[sl] >= D["ton"][s]),
                                       pos * (t10 >= D["ton"][s]), logR[s], D["isc"][s, sl] * pos, pos,
                                       np.clip(D["dv"][s, sl, q], -20, 20), D["I"][s, sl] / CAP)):
                    c.append(v)
        x1, x10, yb, y10, lr, ib, mpos, dvb, icb = [jnp.asarray(np.array(c, np.float32)) for c in cols]
        p, st, l, aux = update(p, st, x1, x10, t_s, yb, y10, lr, ib, mpos, dvb, icb)
        if it % 50 == 0 or it == steps - 1:
            hist.append([it, float(l)] + [float(a) for a in aux])
            print(f"mr it {it} loss {float(l):.4f} bce {float(aux[0]):.4f} reg {float(aux[1]):.3f} leak {float(aux[2]):.3f} "
                  f"phys {float(aux[3]):.3f} bce10 {float(aux[4]):.4f}  {time.time()-t_0:.0f}s", flush=True)
            save(p)
    save(p)
    (OUT / "det_mr_hist.json").write_text(json.dumps(hist))
    return p


def save(p):
    np.savez(OUT / "det_mr_params.npz", **{f"{a}_{k}": np.asarray(v) for a in ("s", "f") for k, v in p[a].items()})


def load_params():
    z = np.load(OUT / "det_mr_params.npz")
    return {a: {k[2:]: jnp.asarray(z[k]) for k in z.files if k.startswith(a + "_")} for a in ("s", "f")}


def run(p, D, split):
    """p_short (S, T, 23) every second over the whole scenario."""
    F10 = np.asarray(features_cached(split, D))
    S, T = D["dv"].shape[:2]
    f = jax.jit(lambda x1, x10: forward(p, x1, x10, 0)[0])
    P_ = np.zeros((S, T, NS), np.float32); R_ = np.zeros((S, T, NS), np.float32)
    for s0 in range(0, S, 8):
        F1 = rls_features_1hz(D["dv"][s0:s0 + 8], D["I"][s0:s0 + 8])
        n = F1.shape[0]
        o = np.asarray(f(jnp.asarray(F1.transpose(0, 2, 1, 3).reshape(-1, T, 7)),
                         jnp.asarray(F10[s0:s0 + n].transpose(0, 2, 1, 3).reshape(-1, F10.shape[1], 5))))
        o = o.reshape(n, NS, T, 3).transpose(0, 2, 1, 3)
        P_[s0:s0 + n] = 1 / (1 + np.exp(-o[..., 0])); R_[s0:s0 + n] = o[..., 1]
    return P_, R_


def evaluate(fa_target=0.02):
    """Same protocol as evaluate_combined: rule (K, thr) chosen on validation for the most shorts found at
    <= fa_target false alarms per module-hour, then the 512 test scenarios."""
    p = load_params()
    V = load("val"); Pv, _ = run(p, V, "val")
    hours = len(V["k"]) * V["dv"].shape[1] / 3600
    best = None
    for K in (5, 15, 30, 60, 120):
        for th in (0.95, 0.98, 0.99, 0.995, 0.998):
            fa, found = _fa_and_found(_first(Pv, th, K, 1), V)
            print(f"  val K={K:4d} thr={th:.3f} false alarms/h {fa/hours:.4f} found {found}", flush=True)
            if fa / hours <= fa_target and (best is None or found > best[2]):
                best = (K, th, found)
    K, thr = best[0], float(best[1])
    del V, Pv
    D = load("test"); P_, R_ = run(p, D, "test"); g = D["k"] // NP
    first = _first(P_, thr, K, 1)
    fa, _ = _fa_and_found(first, D)
    rows = []
    for s in range(len(g)):
        if not D["has"][s]:
            continue
        ok = bool(np.isfinite(first[s, g[s]]) and first[s, g[s]] >= D["ton"][s])
        rows.append(dict(R=float(D["R"][s]), dur=float(D["dur"][s]), detected=ok,
                         delay=float(first[s, g[s]] - D["ton"][s]) if ok else float("nan"),
                         Rhat=float(10 ** R_[s, -1, g[s]])))
    res = dict(k_consec=K, thr=thr, fa_per_module_hour_test=fa / (len(g) * D["dv"].shape[1] / 3600), rows=rows)
    (OUT / "det_eval_mr.json").write_text(json.dumps(res))
    print("two-rate GRU rule: K", K, "thr", thr, "false alarms per module-hour (test):", round(res["fa_per_module_hour_test"], 4))
    comb = json.loads((OUT / "det_eval_combined.json").read_text())["rows"]
    for lo, hi in zip([0.3, 1, 3, 10, 30], [1, 3, 10, 30, 3000]):
        for kind, sel in (("permanent", lambda r: r["dur"] > 1e8), ("transient", lambda r: r["dur"] < 1e8)):
            out = []
            for name, rr0 in (("two-rate", rows), ("pair", comb)):
                rr = [r for r in rr0 if lo <= r["R"] < hi and sel(r)]
                d = [r for r in rr if r["detected"]]
                out.append(f"{name} {100*len(d)/max(len(rr),1):5.1f} % / {np.median([r['delay'] for r in d]) if d else float('nan'):6.0f} s")
            print(f"R {lo:>4}-{hi:<5} {kind:9s} n={len(rr):3d}  " + "   ".join(out), flush=True)
    return res


def export():
    p = np.load(OUT / "det_mr_params.npz")
    pc = dict(np.load(OUT / "det_cell1hz_params.npz"))
    E = json.loads((OUT / "det_eval_mr.json").read_text()); rows = E["rows"]

    def rate(lo, hi):
        rr = [r for r in rows if lo <= r["R"] < hi and r["dur"] > 1e8]
        return 100 * np.mean([r["detected"] for r in rr]) if rr else float("nan")
    limits = (f"Test set, permanent shorts within 2 h: {rate(0.3, 3):.0f} % found at 0.3-3 ohm, {rate(3, 30):.0f} % at 3-30 ohm, "
              f"{rate(30, 300):.0f} % at 30-300 ohm; {E['fa_per_module_hour_test']:.3f} false alarms per module-hour.")
    W = dict(version="tworate", tau_rls=TAU_RLS, limits=limits, w10=W10, k_consec=E["k_consec"], thr=E["thr"],
             slow={k[2:]: p[k].tolist() for k in p.files if k.startswith("s_")},
             fast={k[2:]: p[k].tolist() for k in p.files if k.startswith("f_")},
             cell=dict(params={k: v.tolist() for k, v in pc.items()}, p_min=0.2))
    (HERE.parent / "module_twin" / "detector_weights.json").write_text(json.dumps(W))
    print("exported two-rate;", limits)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    elif cmd == "eval":
        evaluate()
    elif cmd == "export":
        export()

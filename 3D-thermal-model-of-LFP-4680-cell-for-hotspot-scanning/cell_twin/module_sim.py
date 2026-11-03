"""Paper 1.2 - fast 23s9p module simulator for training the short detector.

Batched over scenarios (numpy, shape (B, 207)). The same equations run in the
browser twin (module_twin_3d.html), with the same parameters (sim_params.json),
so the trained detector sees in the twin what it saw in training.

Cell model: OCV(SOC) - R0 i - V1 (one RC branch), Arrhenius in temperature;
two thermal nodes (core, can) cooled by serpentine ribbons; cell-to-cell spread
in capacity and resistance. The 9 cells of a group are solved in parallel.

Internal short, from the 3-D electrode sheet (fingerprint.py): each cell is a
two-port, terminal and short,

    I_sc = (U - Z_tr i) / (R_sc + Z_th),   V = U - Z i - Z_tr I_sc,

where Z_th and Z_tr depend on the position of the short along the electrode.
They enter here as fractions of the cell's DC resistance, f_th = Z_th/Z_cell
and f_tr = Z_tr/Z_cell, drawn from the fingerprint library.

What the BMS measures: the 23 group voltages (noise, resolution), the module
current, 2 NTCs on the can of two cells at the coolant inlet side, and - at
cell level - an impedance-based temperature per cell (noise, calibration bias).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TWIN = HERE.parent / "module_twin"
SIMP = json.loads((TWIN / "sim_params.json").read_text())
P = SIMP["P"]

D, NS, NP = 46.3, 23, 9
N = NS * NP
GAPX, RIB = 1.5, 2.5
PX, PY = D + GAPX, D * np.sqrt(3) / 2 + RIB * 0.9


def _layout():
    cx = np.zeros(N); cy = np.zeros(N); grp = np.zeros(N, int); row = np.zeros(N, int)
    for i in range(NS):
        for j in range(NP):
            k = i * NP + j
            cx[k] = i * PX + (PX / 2 if j % 2 else 0.0); cy[k] = j * PY; grp[k] = i; row[k] = j
    nb = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a + 1, N):
            if np.hypot(cx[a] - cx[b], cy[a] - cy[b]) < PX * 1.08:
                nb[a].append(b); nb[b].append(a)
    NR = NP - 1
    rib = []
    for r in range(NR):
        lst = [k for k in range(N) if row[k] in (r, r + 1)]
        lst.sort(key=lambda k: cx[k])
        if r % 2 == 1:
            lst.reverse()
        rib.append(np.array(lst))
    return cx, cy, grp, row, nb, rib


CX, CY, GRP, ROW, NB, RIBS = _layout()
NTC1, NTC2 = 0, NP - 1
# neighbour matrix for the can-to-can exchange
NBM = np.zeros((N, N))
for a in range(N):
    for b in NB[a]:
        NBM[a, b] = 1.0
NBDEG = NBM.sum(1)


def ocv(s):
    s = np.clip(s, 0.001, 0.999)
    return (3.25 + 0.08 * s + 0.12 * np.exp(-(1 - s) / 0.02) - 0.35 * np.exp(-s / 0.03)
            + 0.02 * np.tanh((s - 0.6) / 0.05))


# ------------------------------------------------------------------ fingerprint library
def load_fingerprint(cases=("A_literal", "B_calibrated")):
    """Positions with f_th = Z_th/Z_cell and f_tr = Z_tr/Z_cell at DC. Both foil cases are
    pooled by default: the real cell lies between them, so the detector must work for both."""
    F = json.loads((HERE / "out" / "fingerprint.json").read_text())
    f_th, f_tr, pos = [], [], []
    for case in cases:
        c = F["cases"][case]
        Zc = c["Z_cell"][0][0]
        f_th.append(np.array(c["Z_th"][0][0]) / Zc)
        f_tr.append(np.array(c["Z_tr"][0][0]) / Zc)
        pos += [dict(p, case=case) for p in F["positions"]]
    return dict(f_th=np.concatenate(f_th), f_tr=np.concatenate(f_tr), pos=pos)


# ------------------------------------------------------------------ load profiles
def twin_profile(kind, crate, t):
    c = crate * P["Q_Ah"]
    if kind == "const":
        return np.full_like(t, c)
    if kind == "rest":
        return np.zeros_like(t)
    if kind == "pulse":
        return np.where((np.floor(t / 10) % 2) == 0, c, 0.0)
    u = t % 600
    f = 0.30 + 0.12 * np.sin(2 * np.pi * u / 97) + 0.08 * np.sin(2 * np.pi * u / 31)
    f = np.where(((u > 60) & (u < 75)) | ((u > 310) & (u < 330)), 1.0, f)
    f = np.where(((u > 140) & (u < 160)) | ((u > 420) & (u < 440)), -0.45, f)
    f = np.where((u > 500) & (u < 560), 0.0, f)
    return c * f


def random_profile(rng, t):
    """Piecewise profile: rest, constant, drive-like segments of random length."""
    out = np.zeros_like(t)
    i = 0
    while i < len(t):
        L = int(rng.integers(20, 600))
        kind = rng.choice(["rest", "const", "drive", "pulse", "regen"], p=[0.25, 0.2, 0.35, 0.1, 0.1])
        c = rng.uniform(0.1, 3.0)
        tt = t[i:i + L]
        if kind == "rest":
            v = np.zeros_like(tt)
        elif kind == "const":
            v = np.full_like(tt, c * P["Q_Ah"] * rng.choice([1, 1, 1, -0.5]))
        elif kind == "regen":
            v = np.full_like(tt, -c * 0.5 * P["Q_Ah"])
        else:
            v = twin_profile(kind, c, tt + rng.uniform(0, 600))
        out[i:i + L] = v
        i += L
    return out


# ------------------------------------------------------------------ scenarios
def sample_scenarios(rng, B, T, dt, p_short=0.6, fp=None, R_range=(0.3, 3e3)):
    """B scenarios of length T [s]. Returns dict of arrays (per scenario / per cell)."""
    fp = fp or load_fingerprint()
    nt = int(round(T / dt))
    t = np.arange(nt) * dt
    I = np.zeros((B, nt))
    for b in range(B):
        if rng.random() < 0.4:
            kind = rng.choice(["const", "drive", "pulse", "rest"], p=[0.2, 0.5, 0.2, 0.1])
            I[b] = twin_profile(kind, rng.uniform(0.2, 3.0), t + rng.uniform(0, 600)) * NP
        else:
            I[b] = random_profile(rng, t) * NP
    # keep the module inside its usable window: after the state of charge would leave
    # [0.12, 0.97], the module rests (a BMS would stop it there)
    soc0 = rng.uniform(0.25, 0.95, B)
    for b in range(B):
        soc = soc0[b] - np.cumsum(I[b]) * dt / (NP * P["Q_Ah"] * 3600)
        bad = np.nonzero((soc < 0.12) | (soc > 0.97))[0]
        if len(bad):
            I[b, bad[0]:] = 0.0
    sc = dict(t=t, I=I, dt=dt,
              Tin=rng.uniform(15, 35, B), flow=rng.uniform(1.0, 10.0, B),
              soc0=soc0,
              cap=P["Q_Ah"] * (1 + P["sd_cap"] * rng.standard_normal((B, N))),
              res=P["R0"] * (1 + P["sd_res"] * rng.standard_normal((B, N))),
              soc_spread=0.005 * rng.standard_normal((B, N)),
              imp_bias=1.0 * rng.standard_normal((B, N)))
    has = rng.random(B) < p_short
    k = rng.integers(0, N, B)
    R = np.exp(rng.uniform(np.log(R_range[0]), np.log(R_range[1]), B))
    ton = rng.uniform(30, 0.6 * T, B)
    dur = np.where(rng.random(B) < 0.5, 1e9, np.exp(rng.uniform(np.log(1.0), np.log(900.0), B)))
    # intermittent: on/off with a random period, for 20 % of the transient ones
    inter = (dur < 1e8) & (rng.random(B) < 0.2)
    period = rng.uniform(5, 120, B)
    ip = rng.integers(0, len(fp["f_th"]), B)
    sc.update(has=has, k=k, R=R, ton=ton, dur=dur, inter=inter, period=period,
              f_th=fp["f_th"][ip], f_tr=fp["f_tr"][ip], ipos=ip)
    return sc


def short_on(sc, ti):
    """(B,) bool: is the short conducting at time ti."""
    on = sc["has"] & (ti >= sc["ton"]) & (ti < sc["ton"] + np.where(sc["inter"], 3 * sc["period"] * 10, sc["dur"]))
    ph = ((ti - sc["ton"]) % sc["period"]) < 0.5 * sc["period"]
    return np.where(sc["inter"], on & ph, on)


# ------------------------------------------------------------------ simulation
def simulate(sc, rng, record_every=1, timp_every=10, keep_groups=None, state_every=None):
    """Run the batch. Returns measurements at 1/(record_every*dt) Hz and the truth."""
    B, nt = sc["I"].shape
    dt = sc["dt"]
    Tc = np.repeat(sc["Tin"][:, None], N, 1).astype(float)
    Ts = Tc.copy()
    soc = np.clip(sc["soc0"][:, None] + sc["soc_spread"], 0.02, 0.99)
    V1 = np.zeros((B, N))
    cap, res = sc["cap"], sc["res"]
    bidx = np.arange(B)
    mdot = (sc["flow"] / 60) * 1.05 / (NP - 1)
    NR = len(RIBS)
    beta = P["G_rib"] / (mdot * P["cp_cool"])                      # (B,)
    W, W0 = [], []
    nrib = np.zeros(N)
    for r in range(NR):
        L = len(RIBS[r]); nrib[RIBS[r]] += 1
        n_ = np.arange(L)
        W0.append((1 - beta[:, None]) ** n_[None, :])                # weight of the inlet temperature
        e = n_[:, None] - 1 - n_[None, :]                             # n-1-m
        Wr = np.where(e[None] >= 0, beta[:, None, None] * (1 - beta[:, None, None]) ** np.maximum(e, 0)[None], 0.0)
        W.append(Wr)
    nrec = nt // record_every
    Vg_rec = np.zeros((B, nrec, NS)); I_rec = np.zeros((B, nrec)); ntc_rec = np.zeros((B, nrec, 2))
    timp_rec = np.zeros((B, nrec // timp_every, N), np.float32); acc_T = np.zeros((B, N)); isc_rec = np.zeros((B, nrec)); Psc_rec = np.zeros((B, nrec))
    timp1 = np.zeros((B, nrec, keep_groups.shape[1], NP), np.float16) if keep_groups is not None else None
    nst = nt // state_every if state_every else 0
    Tst = np.zeros((B, nst, N), np.float32); Sst = np.zeros((B, nst, N), np.float32); Ist = np.zeros((B, nst), np.float32)
    acc_V = np.zeros((B, NS)); acc_I = np.zeros(B); acc_isc = np.zeros(B); acc_P = np.zeros(B)
    for n in range(nt):
        ti = sc["t"][n]
        Im = sc["I"][:, n]                                   # module current (+ discharge)
        fT = np.exp(P["Ea_R"] * (1 / (Ts + 273.15) - 1 / 298.15))
        R = res * fT
        U = ocv(soc) - V1
        on = short_on(sc, ti)
        Rsc = np.full((B, N), np.inf)
        Rsc[bidx[on], sc["k"][on]] = sc["R"][on]
        Zth = sc["f_th"][:, None] * (P["R0"] + P["R1"])
        Ztr = sc["f_tr"][:, None] * (P["R0"] + P["R1"])
        a = np.where(np.isfinite(Rsc), Ztr / (Rsc + Zth), 0.0)
        Up = U * (1 - a); Zp = R - Ztr * a
        Upg = Up.reshape(B, NS, NP); Zpg = Zp.reshape(B, NS, NP)
        V = ((Upg / Zpg).sum(2) - Im[:, None]) / (1 / Zpg).sum(2)     # (B, NS)
        i = ((Upg - V[:, :, None]) / Zpg).reshape(B, N)
        Isc = np.where(np.isfinite(Rsc), (U - Ztr * i) / np.where(np.isfinite(Rsc), Rsc + Zth, 1.0), 0.0)
        ea = np.exp(-dt / P["tau1"])
        V1 = V1 * ea + P["R1"] * i * (1 - ea)
        soc = soc - dt * (i + Isc) / (cap * 3600)
        # thermal
        Q = i * i * R + i * V1 + Isc * Isc * np.where(np.isfinite(Rsc), Rsc, 0.0)
        Tcool = np.zeros((B, N))
        for r in range(NR):
            lst = RIBS[r]
            # coolant temperature met by each cell along the ribbon (linear recurrence, closed form)
            Tcool[:, lst] += W0[r] * sc["Tin"][:, None] + np.einsum("bnm,bm->bn", W[r], Ts[:, lst])
        qcs = (Tc - Ts) / P["R_cs"]
        qnb = P["G_nb"] * (Ts * NBDEG - Ts @ NBM)
        qrib = P["G_rib"] * nrib * (Ts - Tcool / np.maximum(1, nrib))
        Tc = Tc + dt * (Q - qcs) / P["C_core"]
        Ts = Ts + dt * (qcs - qrib - qnb) / P["C_can"]
        # measurements
        Vm = np.round((V + P["v_noise"] * rng.standard_normal((B, NS))) / P["v_lsb"]) * P["v_lsb"]
        acc_V += Vm; acc_I += Im; acc_isc += Isc.sum(1); acc_P += (Isc * Isc * np.where(np.isfinite(Rsc), Rsc, 0.0)).sum(1)
        if state_every and (n + 1) % state_every == 0 and (n + 1) // state_every - 1 < nst:
            j = (n + 1) // state_every - 1
            Tst[:, j] = 0.5 * (Tc + Ts); Sst[:, j] = soc; Ist[:, j] = Im
        if (n + 1) % record_every == 0:
            m = (n + 1) // record_every - 1
            Vg_rec[:, m] = acc_V / record_every; I_rec[:, m] = acc_I / record_every
            isc_rec[:, m] = acc_isc / record_every; Psc_rec[:, m] = acc_P / record_every
            ntc_rec[:, m, 0] = Ts[:, NTC1] + 0.05 * rng.standard_normal(B)
            ntc_rec[:, m, 1] = Ts[:, NTC2] + 0.05 * rng.standard_normal(B)
            tnow = 0.5 * (Tc + Ts) + sc["imp_bias"] + P.get("imp_noise", 0.5) * rng.standard_normal((B, N))
            acc_T += tnow
            if keep_groups is not None:
                for j in range(keep_groups.shape[1]):
                    for c in range(NP):
                        timp1[:, m, j, c] = tnow[np.arange(B), keep_groups[:, j] * NP + c]
            if (m + 1) % timp_every == 0 and (m + 1) // timp_every - 1 < timp_rec.shape[1]:
                timp_rec[:, (m + 1) // timp_every - 1] = acc_T / timp_every; acc_T[:] = 0
            acc_V[:] = 0; acc_I[:] = 0; acc_isc[:] = 0; acc_P[:] = 0
    return dict(Vg=Vg_rec, I=I_rec, ntc=ntc_rec, timp=timp_rec, isc=isc_rec, Psc=Psc_rec, timp1=timp1, T_state=Tst, soc_state=Sst, I_state=Ist,
                t=(np.arange(nrec) + 1) * record_every * dt)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    t0 = time.time()
    sc = sample_scenarios(rng, 64, 1200.0, 1.0)
    out = simulate(sc, rng)
    print("64 x 1200 s in", round(time.time() - t0, 1), "s")
    b = int(np.nonzero(sc["has"] & (sc["dur"] > 1e8))[0][0])
    g = sc["k"][b] // NP
    dv = (out["Vg"][b] - np.median(out["Vg"][b], 1, keepdims=True)) * 1e3
    print("R", sc["R"][b], "group", g, "ton", sc["ton"][b], "f_th", sc["f_th"][b], "f_tr", sc["f_tr"][b])
    print("dv of the shorted group, mV, every 100 s:", np.round(dv[::100, g], 2))

  // Stage-1 detector (train_det.py). One observer per group on the M7 every second feeds two GRUs:
  // the fast path every second (rls_features_1hz) and the slow path on 10 s means (rls_features).
  // Alarm when either path fires. The cell-level GRU runs every second on the impedance temperatures.
  // On the S32K5: observer, features and alarm logic on the M7; the GRU steps on the NPU.
  function makeDetector(NNW) {
    const G = NNW.fast, SL = NNW.slow, C = NNW.cell, CAP = NP * P.Q_Ah;
    function gru(p) {
      const H = p.Wh.length, F = p.Wx.length, O = p.bo.length;
      return {
        H, step(h, x) {
          const xg = new Float64Array(3 * H), hg = new Float64Array(3 * H);
          for (let j = 0; j < 3 * H; j++) {
            let a = p.b[j]; for (let f = 0; f < F; f++) a += x[f] * p.Wx[f][j]; xg[j] = a;
            let c = 0; for (let i = 0; i < H; i++) c += h[i] * p.Wh[i][j]; hg[j] = c;
          }
          const hn = new Float64Array(H);
          for (let i = 0; i < H; i++) {
            const z = 1 / (1 + Math.exp(-(xg[i] + hg[i])));
            const r = 1 / (1 + Math.exp(-(xg[H + i] + hg[H + i])));
            const n = Math.tanh(xg[2 * H + i] + r * hg[2 * H + i]);
            hn[i] = (1 - z) * n + z * h[i];
          }
          return hn;
        },
        out(h) { const o = new Float64Array(O); for (let k = 0; k < O; k++) { let a = p.bo[k]; for (let i = 0; i < H; i++) a += h[i] * p.Wo[i][k]; o[k] = a; } return o; },
      };
    }
    const gg = gru(G.params), gs = gru(SL.params), gc = gru(C.params);
    const lam = 1 - 1 / NNW.tau_rls, L1 = 300, L2 = 1800, W10 = SL.w10;
    const med = a => { const b = Array.from(a).sort((x, y) => x - y); return b[(b.length - 1) >> 1]; };
    const cl = (v, m) => Math.max(-m, Math.min(m, v));
    let hG, hC, pG, Rg, run, alarmG, alarmT, pC, tNow, th, Pm, Chist, C0, Iprev, n, Tbuf, X0hist, X00;
    let hS, pS, runS, cacc, eacc, iacc, n1, chS, cS0, nS, byG;
    function reset() {
      hG = Array.from({ length: NS }, () => new Float64Array(gg.H));
      hC = Array.from({ length: N }, () => new Float64Array(gc.H));
      pG = new Float64Array(NS); Rg = new Float64Array(NS);
      run = new Int32Array(NS); alarmG = new Int8Array(NS); alarmT = new Float64Array(NS).fill(NaN);
      pC = new Float64Array(N).fill(1 / NP); tNow = 0;
      th = Array.from({ length: NS }, () => [0, 0]); Pm = Array.from({ length: NS }, () => [100, 0, 100]);
      Chist = []; C0 = null; Iprev = null; n = 0; Tbuf = []; X0hist = []; X00 = null;
      hS = Array.from({ length: NS }, () => new Float64Array(gs.H)); pS = new Float64Array(NS); runS = new Int32Array(NS);
      cacc = new Float64Array(NS); eacc = new Float64Array(NS); iacc = 0; n1 = 0; chS = []; cS0 = null; nS = 0;
      byG = Array(NS).fill("");
    }
    reset();
    function observe(t, Imod, Vm, Timp) {
      tNow = t;
      // ---- M7: observer (RLS on dv = r Ic + c) per group
      const m0 = med(Vm), Ic = Imod / CAP;
      const c = new Float64Array(NS), e = new Float64Array(NS);
      for (let g = 0; g < NS; g++) {
        const dv = cl((Vm[g] - m0) * 1e3, 20);
        const T_ = th[g], Q = Pm[g];
        const err = dv - (T_[0] * Ic + T_[1]);
        const px0 = Q[0] * Ic + Q[1], px1 = Q[1] * Ic + Q[2];
        const den = lam + Ic * px0 + px1, k0 = px0 / den, k1 = px1 / den;
        T_[0] += k0 * err; T_[1] += k1 * err;
        Q[0] = (Q[0] - k0 * px0) / lam; Q[1] = (Q[1] - k0 * px1) / lam; Q[2] = (Q[2] - k1 * px1) / lam;
        c[g] = T_[1]; e[g] = err;
      }
      if (!C0) C0 = c;
      Chist.push(c); if (Chist.length > L2 + 1) Chist.shift();
      const lag = L => (n >= L ? Chist[Chist.length - 1 - L] : C0);
      const c1 = lag(L1), c2 = lag(L2);
      const d1 = c.map((v, g) => v - c1[g]), d2 = c.map((v, g) => v - c2[g]);
      const m1 = med(d1), m2 = med(d2), m3 = med(e), m6 = med(d1.map(Math.abs)), m7 = med(e.map(Math.abs));
      const dIc = Iprev === null ? 0 : Ic - Iprev; Iprev = Ic;
      // ---- NPU: GRU step for the 23 groups
      for (let g = 0; g < NS; g++) {
        hG[g] = gg.step(hG[g], [cl(d1[g] - m1, 30), cl(d2[g] - m2, 30), cl(e[g] - m3, 30), cl(Math.abs(Ic), 30), cl(dIc, 30), cl(m6, 30), cl(m7, 30)]);
        const o = gg.out(hG[g]);
        pG[g] = 1 / (1 + Math.exp(-o[0])); if (!(Rg[g] > 0)) Rg[g] = Math.pow(10, o[1]);
        run[g] = pG[g] > G.thr ? run[g] + 1 : 0;
        if (!alarmG[g] && run[g] >= G.k_consec) { alarmG[g] = 1; alarmT[g] = t; byG[g] = "fast path"; }
      }
      // ---- slow path: 10 s means of the observer, GRU step every 10 s
      for (let g = 0; g < NS; g++) { cacc[g] += c[g] / W10; eacc[g] += e[g] / W10; }
      iacc += Math.abs(Ic) / W10;
      if (++n1 === W10) {
        const cb = Float64Array.from(cacc);
        if (!cS0) cS0 = cb;
        chS.push(cb); if (chS.length > 181) chS.shift();
        const lagS = L => (nS >= L ? chS[chS.length - 1 - L] : cS0);
        const s30 = lagS(30), s180 = lagS(180);
        const e1 = cb.map((v, g) => v - s30[g]), e2 = cb.map((v, g) => v - s180[g]);
        const q1 = med(e1), q2 = med(e2), q3 = med(eacc), q5 = med(e1.map(Math.abs));
        for (let g = 0; g < NS; g++) {
          hS[g] = gs.step(hS[g], [cl(e1[g] - q1, 30), cl(e2[g] - q2, 30), cl(eacc[g] - q3, 30), cl(iacc, 30), cl(q5, 30)]);
          const o = gs.out(hS[g]);
          pS[g] = 1 / (1 + Math.exp(-o[0])); Rg[g] = Math.pow(10, o[1]);
          runS[g] = pS[g] > SL.thr ? runS[g] + 1 : 0;
          if (!alarmG[g] && runS[g] >= SL.k_consec) { alarmG[g] = 1; alarmT[g] = t; byG[g] = "slow path"; }
        }
        cacc.fill(0); eacc.fill(0); iacc = 0; n1 = 0; nS++;
      }
      // ---- M7: impedance temperature, 10 s moving mean minus the group mean, change over 60 s
      Tbuf.push(Float64Array.from(Timp)); if (Tbuf.length > 10) Tbuf.shift();
      const x0 = new Float64Array(N);
      for (let g = 0; g < NS; g++) {
        let s = 0;
        for (let j = 0; j < NP; j++) { let a = 0; for (const b of Tbuf) a += b[g * NP + j]; x0[g * NP + j] = a / Tbuf.length; s += x0[g * NP + j]; }
        s /= NP; for (let j = 0; j < NP; j++) x0[g * NP + j] -= s;
      }
      if (!X00) X00 = x0;
      X0hist.push(x0); if (X0hist.length > 61) X0hist.shift();
      const xp = n >= 60 ? X0hist[X0hist.length - 61] : X00;
      // ---- NPU: GRU step for the 207 cells; softmax over the 9 cells of each group
      for (let g = 0; g < NS; g++) {
        const lg = new Float64Array(NP); let mx = -Infinity;
        for (let j = 0; j < NP; j++) {
          const k = g * NP + j;
          hC[k] = gc.step(hC[k], [cl(x0[k], 10), cl(x0[k] - xp[k], 5)]); lg[j] = gc.out(hC[k])[0]; mx = Math.max(mx, lg[j]);
        }
        let s = 0; for (let j = 0; j < NP; j++) { lg[j] = Math.exp(lg[j] - mx); s += lg[j]; }
        for (let j = 0; j < NP; j++) pC[g * NP + j] = lg[j] / s;
      }
      n++;
    }
    function flaggedGroups() { const o = []; for (let g = 0; g < NS; g++) if (alarmG[g]) o.push(g); return o; }
    function bestCell(g) { let jb = 0; for (let j = 1; j < NP; j++) if (pC[g * NP + j] > pC[g * NP + jb]) jb = j; return { k: g * NP + jb, p: pC[g * NP + jb] }; }
    function flaggedCells(level) {
      const o = [];
      for (const g of flaggedGroups()) {
        const b = bestCell(g);
        if (level === "cell" && b.p >= C.p_min) o.push(b.k);
        else for (let j = 0; j < NP; j++) o.push(g * NP + j);
      }
      return o;
    }
    function worstInside(Pw, tau) {
      let mx = 0; for (const e of LIB.entries) mx = Math.max(mx, riseAt(e, Math.max(tau, 60)));
      return Pw * mx * (LIB.mult["1.0"] || 1);
    }
    function verdict(level) {
      const fg = flaggedGroups();
      if (!fg.length) {
        let gmax = 0; for (let g = 1; g < NS; g++) if (Math.max(pG[g], pS[g]) > Math.max(pG[gmax], pS[gmax])) gmax = g;
        return { alarm: false, text: "No short detected", sub: `Scanning 23 groups and 207 cells every second (fast path) and on 10 s means (slow path). Highest short probability: group ${gmax + 1}, ${(100 * Math.max(pG[gmax], pS[gmax])).toFixed(0)} %. ${NNW.limits}` };
      }
      const g = fg[0], R = Rg[g], Pw = Math.pow(P.V_nom, 2) / R, b = bestCell(g);
      let where = `group ${g + 1} (one of its 9 cells)`;
      if (level === "cell") where = b.p >= C.p_min ? `cell: group ${g + 1}, row ${b.k % NP + 1} (probability ${(100 * b.p).toFixed(0)} %, chance 11 %)` : `group ${g + 1}; cell not resolved yet (best ${(100 * b.p).toFixed(0)} %, chance 11 %)`;
      const worst = worstInside(Pw, tNow - alarmT[g]);
      return { alarm: true, group: g, t: alarmT[g], text: `ALARM: internal short in ${where}`,
        sub: `Detected at t = ${fmtT(alarmT[g])} by the ${byG[g]} · estimated R ≈ ${R < 10 ? R.toFixed(1) : R.toFixed(0)} Ω (within a factor of ~2; ≈ ${Pw.toFixed(1)} W) · ` +
             (R < 1 ? "estimate below 1 Ω: may be beyond soft-short scope" : `worst case inside ≈ ${worst.toFixed(0)} K above the can (3-D library, 1 mm short, linear upper bound)`) +
             (fg.length > 1 ? ` · ${fg.length} groups flagged` : "") + " · next step: 1 kHz impedance scan of the suspect cell" };
    }
    return { reset, observe, flaggedGroups, flaggedCells, verdict, probs: () => Array.from(pG), probsSlow: () => Array.from(pS), cellp: () => Array.from(pC) };
  }

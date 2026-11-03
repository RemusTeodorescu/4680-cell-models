  // Stage-1 detector, 1 Hz, as trained in train_det.py (same features, same weights).
  function makeDetector(NNW) {
    const G = NNW.group, C = NNW.cell, CAP = NP * P.Q_Ah;
    function gru(p) {
      const H = p.Wh.length, F = p.Wx.length, O = p.bo.length;
      return {
        H, step(h, x) {                                    // h: Float64Array(H), x: array(F) -> new h
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
    const gg = gru(G.params), gc = gru(C.params);
    const TAU = G.tau_rls, W10 = G.w10, lam = 1 - 1 / TAU;
    let hG, hC, pG, Rg, leakG, run, alarmG, alarmT, acc, nacc, dTbuf, pC, tNow;
    let th, Pm, cacc, eacc, iacc, n1, chist, c0;
    const med = a => { const b = Array.from(a).sort((x, y) => x - y); return b[(b.length - 1) >> 1]; };
    function reset() {
      hG = Array.from({ length: NS }, () => new Float64Array(gg.H));
      hC = Array.from({ length: N }, () => new Float64Array(gc.H));
      pG = new Float64Array(NS); Rg = new Float64Array(NS); leakG = new Float64Array(NS);
      run = new Int32Array(NS); alarmG = new Int8Array(NS); alarmT = new Float64Array(NS).fill(NaN);
      acc = new Float64Array(N); nacc = 0; dTbuf = []; pC = new Float64Array(N).fill(1 / NP); tNow = 0;
      th = Array.from({ length: NS }, () => [0, 0]); Pm = Array.from({ length: NS }, () => [100, 0, 100]);
      cacc = new Float64Array(NS); eacc = new Float64Array(NS); iacc = 0; n1 = 0; chist = []; c0 = null;
    }
    reset();
    function observe(t, Imod, Vm, Timp) {
      tNow = t;
      // ---- physics-based observer (RLS on dv = r Ic + c), every second
      const m0 = med(Vm), Ic = Imod / CAP;
      for (let g = 0; g < NS; g++) {
        const dv = Math.max(-20, Math.min(20, (Vm[g] - m0) * 1e3));
        const T_ = th[g], Q = Pm[g];
        const e = dv - (T_[0] * Ic + T_[1]);
        const px0 = Q[0] * Ic + Q[1], px1 = Q[1] * Ic + Q[2];
        const den = lam + Ic * px0 + px1, k0 = px0 / den, k1 = px1 / den;
        T_[0] += k0 * e; T_[1] += k1 * e;
        Q[0] = (Q[0] - k0 * px0) / lam; Q[1] = (Q[1] - k0 * px1) / lam; Q[2] = (Q[2] - k1 * px1) / lam;
        cacc[g] += T_[1] / W10; eacc[g] += e / W10;
      }
      iacc += Math.abs(Ic) / W10;
      if (++n1 === W10) {
        const c = Float64Array.from(cacc);
        if (!c0) c0 = c;
        chist.push(c); if (chist.length > 181) chist.shift();
        const lag = L => chist.length > L ? chist[chist.length - 1 - L] : (chist.length === 181 ? chist[0] : c0);
        const c30 = lag(30), c180 = lag(180);
        const d1 = c.map((v, g) => v - c30[g]), d2 = c.map((v, g) => v - c180[g]);
        const m1 = med(d1), m2 = med(d2), m3 = med(eacc), m5 = med(d1.map(Math.abs));
        const cl = v => Math.max(-30, Math.min(30, v));
        for (let g = 0; g < NS; g++) {
          hG[g] = gg.step(hG[g], [cl(d1[g] - m1), cl(d2[g] - m2), cl(eacc[g] - m3), cl(iacc), cl(m5)]);
          const o = gg.out(hG[g]);
          pG[g] = 1 / (1 + Math.exp(-o[0])); Rg[g] = Math.pow(10, o[1]); leakG[g] = Math.log1p(Math.exp(o[2])) * 0.1;
          run[g] = pG[g] > G.thr ? run[g] + 1 : 0;
          if (!alarmG[g] && run[g] >= G.k_consec) { alarmG[g] = 1; alarmT[g] = t; }
        }
        cacc.fill(0); eacc.fill(0); iacc = 0; n1 = 0;
      }
      // cell level: impedance temperature, 10 s means, minus the group mean
      for (let k = 0; k < N; k++) acc[k] += Timp[k];
      if (++nacc === 10) {
        const m = new Float64Array(N);
        for (let g = 0; g < NS; g++) { let s = 0; for (let j = 0; j < NP; j++) s += acc[g * NP + j] / 10; s /= NP; for (let j = 0; j < NP; j++) m[g * NP + j] = acc[g * NP + j] / 10 - s; }
        dTbuf.push(m); if (dTbuf.length > 7) dTbuf.shift();
        const old = dTbuf[0];
        for (let g = 0; g < NS; g++) {
          const lg = new Float64Array(NP); let mx = -Infinity;
          for (let j = 0; j < NP; j++) {
            const k = g * NP + j;
            const x0 = Math.max(-10, Math.min(10, m[k])), x1 = Math.max(-5, Math.min(5, dTbuf.length === 7 ? m[k] - old[k] : 0));
            hC[k] = gc.step(hC[k], [x0, x1]); lg[j] = gc.out(hC[k])[0]; mx = Math.max(mx, lg[j]);
          }
          let s = 0; for (let j = 0; j < NP; j++) { lg[j] = Math.exp(lg[j] - mx); s += lg[j]; }
          for (let j = 0; j < NP; j++) pC[g * NP + j] = lg[j] / s;
        }
        acc.fill(0); nacc = 0;
      }
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
    function verdict(level) {
      const fg = flaggedGroups();
      if (!fg.length) {
        let gmax = 0; for (let g = 1; g < NS; g++) if (pG[g] > pG[gmax]) gmax = g;
        return { alarm: false, text: "No short detected", sub: `Watching 23 groups at 1 Hz (observer) and every 10 s (network). Highest short probability: group ${gmax + 1}, ${(100 * pG[gmax]).toFixed(0)} %. ${G.limits}` };
      }
      const g = fg[0], R = Rg[g], Pw = Math.pow(P.V_nom, 2) / R, b = bestCell(g);
      let where = `group ${g + 1} (one of its 9 cells)`;
      if (level === "cell") where = b.p >= C.p_min ? `cell: group ${g + 1}, row ${b.k % NP + 1} (${(100 * b.p).toFixed(0)} % sure)` : `group ${g + 1}; cell not resolved yet (best ${(100 * b.p).toFixed(0)} %)`;
      const worst = worstInside(Pw, tNow - alarmT[g]);
      return { alarm: true, text: `ALARM: internal short in ${where}`,
        sub: `Detected at t = ${fmtT(alarmT[g])} · estimated R ≈ ${R < 10 ? R.toFixed(1) : R.toFixed(0)} Ω (within a factor of ~2; ≈ ${Pw.toFixed(1)} W) · ` +
             (R < 1 ? "beyond soft-short scope: treat as hard short" : `worst case inside ≈ ${worst.toFixed(0)} K above the can (3-D library, 1 mm short, linear upper bound)`) +
             (fg.length > 1 ? ` · ${fg.length} groups flagged` : "") + " · next step: 1 kHz impedance scan of the suspect cell" };
    }
    function worstInside(Pw, tau) {
      let mx = 0; for (const e of LIB.entries) mx = Math.max(mx, riseAt(e, Math.max(tau, 60)));
      return Pw * mx * (LIB.mult["1.0"] || 1);
    }
    return { reset, observe, flaggedGroups, flaggedCells, verdict, probs: () => Array.from(pG) };
  }

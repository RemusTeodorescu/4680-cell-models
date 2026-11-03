  // ---------------- impedance detector (BMA7418_EIS_TWIN): one scan of all 23 groups every 60 s
  // Same model and detector as model/eis.py (cell impedance Z(f, T, SOC), short two-port,
  // BMA7418 measurement errors, robust cross-group detrend, per-group baseline, one-sided score).
  function makeEIS(E) {
    const F = E.freqs, nF = F.length, C = E.cell, B = E.bma, RG = 8.314;
    const arr = (Ea, T) => Math.exp(Ea / RG * (1 / T - 1 / C.T_ref));
    const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
    const r = mulberry(91);
    const gv = Float64Array.from({ length: NS }, () => 1 + B.gain_v * gauss(r)), gi = 1 + B.gain_i * gauss(r);
    const Nsmp = Math.round(B.d * B.fs), I1 = B.I_exc * (B.waveform === "square" ? 4 / Math.PI : 1);
    const sv = Math.sqrt(B.sig_v ** 2 + B.lsb_v ** 2 / 12) * Math.sqrt(2 / Nsmp), si = B.sig_i * Math.sqrt(2 / Nsmp);
    function zcell(f, T, soc, scale) {
      const w = 2 * Math.PI * f;
      const socf = 1 + 0.6 * Math.exp(-soc / 0.08) + 0.3 * Math.exp(-(1 - soc) / 0.05);
      const act = arr(C.Ea_ct, T) * socf;
      const zarc = (R, Cc, a) => { const x = Math.pow(w * R * Cc, a); return cdiv([R, 0], [1 + x * Math.cos(a * Math.PI / 2), x * Math.sin(a * Math.PI / 2)]); };
      const za = zarc(C.Rct_a * act, C.C_a, C.a_a), zc = zarc(C.Rct_c * act, C.C_c, C.a_c);
      const zw = C.sig_w * arr(C.Ea_w, T) * socf / Math.sqrt(w);
      return [scale * (C.Rs * arr(C.Ea_s, T) + za[0] + zc[0] + zw), scale * (w * C.L + za[1] + zc[1] - zw)];
    }
    // detrend: robust quadratic over the group index, per frequency
    const xg = Array.from({ length: NS }, (_, g) => -1 + 2 * g / (NS - 1));
    function quadfit(y, use) {
      const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], v = [0, 0, 0];
      for (let g = 0; g < NS; g++) if (use[g]) { const a = [1, xg[g], xg[g] * xg[g]]; for (let i = 0; i < 3; i++) { v[i] += a[i] * y[g]; for (let j = 0; j < 3; j++) M[i][j] += a[i] * a[j]; } }
      for (let i = 0; i < 3; i++) { let p = i; for (let k = i + 1; k < 3; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k; [M[i], M[p]] = [M[p], M[i]]; [v[i], v[p]] = [v[p], v[i]];
        for (let k = i + 1; k < 3; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j < 3; j++) M[k][j] -= f * M[i][j]; v[k] -= f * v[i]; } }
      const c = [0, 0, 0]; for (let i = 2; i >= 0; i--) { let s = v[i]; for (let j = i + 1; j < 3; j++) s -= M[i][j] * c[j]; c[i] = s / M[i][i]; }
      return y.map((yy, g) => yy - (c[0] + c[1] * xg[g] + c[2] * xg[g] * xg[g]));
    }
    function detrend(Zg) {                     // Zg[g][f] complex -> e[g][f]
      const e = Array.from({ length: NS }, () => new Float64Array(nF));
      for (let f = 0; f < nF; f++) {
        const y = Zg.map(z => Math.log(Math.hypot(z[f][0], z[f][1])));
        const use = Array(NS).fill(true);
        let res = quadfit(y, use);
        const worst = res.map((v, g) => [Math.abs(v), g]).sort((a, b) => b[0] - a[0]).slice(0, 2);
        for (const [, g] of worst) use[g] = false;
        res = quadfit(y, use);
        for (let g = 0; g < NS; g++) e[g][f] = res[g];
      }
      return e;
    }
    const wsum = Math.sqrt(E.w.reduce((a, b) => a + b * b, 0));
    const score1 = d => -d.reduce((a, v, f) => a + v / E.sigma[f] * E.w[f], 0) / wsum;
    let hist, base, sbuf, alarm, alarmT, last, nscan, latest;
    function reset() { hist = []; base = null; sbuf = []; alarm = new Int8Array(NS); alarmT = new Float64Array(NS).fill(NaN); last = -1e9; nscan = 0; latest = new Float64Array(NS); }
    reset();
    function scan(t) {
      // ---- measurement: per cell impedance at its temperature and SOC, the short, 9p group, BMA errors
      const Rdc = P.R0 + P.R1;
      const Zg = [];
      const ni = Array.from({ length: nF }, () => [si * gauss(rng), si * gauss(rng)]);
      for (let g = 0; g < NS; g++) {
        const row = [];
        for (let fi = 0; fi < nF; fi++) {
          const f = F[fi]; let Y = [0, 0];
          for (let j = 0; j < NP; j++) {
            const k = g * NP + j;
            let z = zcell(f, 0.5 * (Tc[k] + Ts[k]) + 273.15, soc[k], resK[k] / P.R0);
            if (Risc[k] > 0 && shortOn(k)) {
              const m = meta.get(k), sh = m.ftr * (0.55 + 0.45 / (1 + f / 200));
              const ztr = [sh * z[0], sh * z[1]], c = cdiv(cmul(ztr, ztr), [Risc[k] + m.fth * Rdc, 0]);
              z = [z[0] - c[0], z[1] - c[1]];
            }
            const y = cdiv([1, 0], z); Y = [Y[0] + y[0], Y[1] + y[1]];
          }
          let Z = cdiv([1, 0], Y);
          const skew = B.skew_std * gauss(rng), ph = -2 * Math.PI * f * skew;
          Z = cmul(Z, [gv[g] / gi * Math.cos(ph), gv[g] / gi * Math.sin(ph)]);
          const nv = [sv * gauss(rng), sv * gauss(rng)];
          row.push(cdiv([Z[0] * I1 - nv[0], Z[1] * I1 - nv[1]], [I1 + ni[fi][0], ni[fi][1]]));
        }
        Zg.push(row);
      }
      // ---- detector
      const e = detrend(Zg);
      nscan++;
      if (nscan <= E.n_init) { hist.push(e); if (nscan === E.n_init) { base = Array.from({ length: NS }, (_, g) => Float64Array.from({ length: nF }, (_, f) => hist.reduce((a, h) => a + h[g][f], 0) / E.n_init)); for (const h of hist) sbuf.push(h.map((eg, g) => score1(eg.map((v, f) => v - base[g][f])))); } return; }
      const s = e.map((eg, g) => score1(eg.map((v, f) => v - base[g][f])));
      for (let g = 0; g < NS; g++) if (s[g] < 0.5 * E.thr) for (let f = 0; f < nF; f++) base[g][f] += (e[g][f] - base[g][f]) / E.tau_scans;
      sbuf.push(s); if (sbuf.length > E.n_smooth) sbuf.splice(0, sbuf.length - E.n_smooth);
      for (let g = 0; g < NS; g++) {
        let m = 0; for (const b of sbuf) m += b[g]; m /= sbuf.length; latest[g] = m;
        if (!alarm[g] && m > E.thr) { alarm[g] = 1; alarmT[g] = t; }
      }
    }
    function observe(t) { if (t - last >= E.scan_s) { last = t; scan(t); } }
    const flaggedGroups = () => { const o = []; for (let g = 0; g < NS; g++) if (alarm[g]) o.push(g); return o; };
    function verdict() {
      const fg = flaggedGroups();
      if (nscan < E.n_init) return { alarm: false, text: "Learning the baseline", sub: `First ${E.n_init} scans (${E.n_init} min) set each group's reference impedance. ${E.limits}` };
      if (!fg.length) {
        let gm = 0; for (let g = 1; g < NS; g++) if (latest[g] > latest[gm]) gm = g;
        return { alarm: false, text: "No short detected", sub: `Impedance scan of 23 groups at 10, 100 and 1000 Hz every minute. Highest score: group ${gm + 1}, ${latest[gm].toFixed(1)} (alarm above ${E.thr.toFixed(1)}). ${E.limits}` };
      }
      const g = fg[0];
      return { alarm: true, group: g, t: alarmT[g], text: `internal short in group ${g + 1}`,
               sub: `Impedance of group ${g + 1} fell below the others (score ${latest[g].toFixed(1)}, alarm above ${E.thr.toFixed(1)}): its shorted cell runs warmer, and its charge-transfer resistance drops.` };
    }
    return { reset, observe, flaggedGroups, verdict };
  }

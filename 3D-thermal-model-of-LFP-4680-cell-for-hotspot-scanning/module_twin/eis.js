  // ---------------- stage 2: BMA7418 EIS scan of the 9 cells of the group the GRU flagged
  // (model/eis_cell.py). Per cell, Z at 10 Hz and 1 kHz with the BMA7418 errors; index
  // x = ln|Z(10 Hz)| - ln|Z(1 kHz)|, where the cell-to-cell resistance spread cancels and
  // temperature does not: the shorted cell runs warmer, so its x is the lowest of the nine.
  // Robust z-score against the other eight over N scans; the cell is named when z < threshold.
  function makeEIS(E) {
    const C = E.cell, B = E.bma, RG = 8.314, F2 = [10, 1000];
    const arr = (Ea, T) => Math.exp(Ea / RG * (1 / T - 1 / C.T_ref));
    const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
    const r0 = mulberry(91);
    const gv = Float64Array.from({ length: N }, () => 1 + B.gain_v * gauss(r0)), gi = 1 + B.gain_i * gauss(r0);
    const Nsmp = Math.round(B.d * B.fs), I1 = B.I_exc * (B.waveform === "square" ? 4 / Math.PI : 1);
    const sv = Math.sqrt(B.sig_v ** 2 + B.lsb_v ** 2 / 12) * Math.sqrt(2 / Nsmp), si = B.sig_i * Math.sqrt(2 / Nsmp);
    function zcell(f, T, socv, scale) {
      const w = 2 * Math.PI * f;
      const socf = 1 + 0.6 * Math.exp(-socv / 0.08) + 0.3 * Math.exp(-(1 - socv) / 0.05);
      const act = arr(C.Ea_ct, T) * socf;
      const zarc = (R, Cc, a) => { const x = Math.pow(w * R * Cc, a); return cdiv([R, 0], [1 + x * Math.cos(a * Math.PI / 2), x * Math.sin(a * Math.PI / 2)]); };
      const za = zarc(C.Rct_a * act, C.C_a, C.a_a), zc = zarc(C.Rct_c * act, C.C_c, C.a_c);
      const zw = C.sig_w * arr(C.Ea_w, T) * socf / Math.sqrt(w);
      return [scale * (C.Rs * arr(C.Ea_s, T) + za[0] + zc[0] + zw), scale * (w * C.L + za[1] + zc[1] - zw)];
    }
    function measure(k) {                      // x of cell k from one scan at 10 Hz and 1 kHz
      const Rdc = P.R0 + P.R1, mag = [];
      for (const f of F2) {
        let z = zcell(f, 0.5 * (Tc[k] + Ts[k]) + 273.15, soc[k], resK[k] / P.R0);
        if (Risc[k] > 0 && shortOn(k)) {
          const m = meta.get(k), sh = m.ftr * (0.55 + 0.45 / (1 + f / 200));
          const c = cdiv(cmul([sh * z[0], sh * z[1]], [sh * z[0], sh * z[1]]), [Risc[k] + m.fth * Rdc, 0]);
          z = [z[0] - c[0], z[1] - c[1]];
        }
        const ph = -2 * Math.PI * f * B.skew_std * gauss(rng);
        z = cmul(z, [gv[k] / gi * Math.cos(ph), gv[k] / gi * Math.sin(ph)]);
        const zm = cdiv([z[0] * I1 - sv * gauss(rng), z[1] * I1 - sv * gauss(rng)], [I1 + si * gauss(rng), si * gauss(rng)]);
        mag.push(Math.hypot(zm[0], zm[1]));
      }
      return Math.log(mag[0]) - Math.log(mag[1]);
    }
    const med = a => { const b = Array.from(a).sort((x, y) => x - y); return b.length % 2 ? b[(b.length - 1) / 2] : 0.5 * (b[b.length / 2 - 1] + b[b.length / 2]); };
    let state, group, scans, tStart, last, result, z, nTot;
    function reset() { state = "idle"; group = -1; scans = []; tStart = NaN; last = -1e9; result = null; z = null; nTot = 0; }
    reset();
    function observe(t, flaggedGroups) {
      if (state === "idle" && flaggedGroups.length) { state = "scan"; group = flaggedGroups[0]; tStart = t; last = -1e9; scans = []; nTot = 0; }
      if (state === "idle" || t - last < E.dt_scan) return;
      last = t;
      scans.push(Array.from({ length: NP }, (_, j) => measure(group * NP + j)));
      if (scans.length > E.n_scan) scans.shift();
      if (scans.length < E.n_scan) return;
      const x = Array.from({ length: NP }, (_, j) => scans.reduce((a, s) => a + s[j], 0) / scans.length);
      const m = med(x);
      let jb = 0; for (let j = 1; j < NP; j++) if (x[j] < x[jb]) jb = j;
      z = m - x[jb];                            // drop of the lowest cell below the group median
      nTot++;
      if (state === "scan" && (z > E.cell_thr || nTot >= 3 * E.n_scan))
        { state = "found"; result = { k: group * NP + jb, z, t, sure: z > E.cell_thr }; }
    }
    function faultedCell() { return state === "found" ? result.k : -1; }
    function verdict() {
      if (state === "idle") return { alarm: false, text: "Waiting for a GRU alarm", sub: `When the GRU flags a group, the BMA7418 scans its 9 cells at 10 Hz and 1 kHz (${E.n_scan} scans, ${E.dt_scan} s apart) and names the warmest one. ${E.cell_limits}` };
      if (state === "scan") {
        const zs = z !== null ? ` · warmest cell ${(100 * z).toFixed(1)} % below the group median (named above ${(100 * E.cell_thr).toFixed(1)} %, or after ${3 * E.n_scan} scans)` : ` · scan ${scans.length} of ${E.n_scan}`;
        return { alarm: false, scanning: true, text: `Scanning the 9 cells of group ${group + 1}`, sub: `Started at t = ${fmtT(tStart)}${zs}` };
      }
      return { alarm: true, group, cell: result.k, t: result.t, text: `faulted cell: group ${group + 1}, row ${result.k % NP + 1}`,
               sub: `${result.sure ? "Named" : "Most likely cell (below the confidence threshold)"} at t = ${fmtT(result.t)}, ${((result.t - tStart) / 60).toFixed(1)} min after the GRU alarm · its 10 Hz / 1 kHz impedance ratio is ${(100 * result.z).toFixed(1)} % below the group median: it runs warmer` };
    }
    return { reset, observe, verdict, faultedCell, flaggedGroups: () => (state === "found" ? [group] : []) };
  }

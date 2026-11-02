"""Build module_twin.html v2: link the 23s9p module twin to the 3-D cell response library.

    python patch_twin.py   (reads ../cell_twin/out/ (or ../model/out/)response_library.json and patch_study.json)
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = next(p for p in (HERE.parent / "cell_twin" / "out", HERE.parent / "model" / "out") if p.exists())
src = (HERE / "module_twin_v1.html").read_text()
lib = json.loads((OUT / "response_library.json").read_text())
patch = json.loads((OUT / "patch_study.json").read_text())

# --------------------------------------------------------------- library payload
t = np.array(lib["t"])
ents = []
for e in lib["entries"]:
    if e["angle"] != "away_from_tab":
        continue
    rise = (np.array(e["peak"]) - np.array(e["side_mean"])).round(4)
    ents.append(dict(r=e["r"], h=e["height"], rise=rise.tolist(), side=np.array(e["side_mean"]).round(4).tolist()))
# footprint multiplier: refined-patch peak at 60 s over the library peak at 60 s (r = 12 mm, mid height)
ref = next(e for e in lib["entries"] if e["r"] == 12.0 and e["height"] == "mid" and e["angle"] == "away_from_tab")
lib60 = float(np.interp(60, t, ref["peak"]))
best = {}
for p in patch:
    w = p["w_mm"]
    if w not in best or p["h_mm"] < best[w]["h_mm"]:
        best[w] = p
mult = {str(w): round(p["peak_60s"] / lib60, 3) for w, p in sorted(best.items())}
payload = dict(t=t.round(2).tolist(), radii=lib["radii_mm"], heights=lib["heights"], entries=ents, mult=mult,
               note="unit-step (1 W) response of the 3-D cell twin, paper 1.2")

# --------------------------------------------------------------- controls
heights_lbl = {"base_end": "2 mm from the base end", "lower": "lower quarter", "mid": "mid-height",
               "upper": "upper quarter", "cap_end": "2 mm from the cap end"}
ctl = f'''      <div class="ctl">
        <label for="sloc">Where in the cell (radius)</label>
        <select id="sloc">{''.join(f'<option value="{i}"{" selected" if r == 12.0 else ""}>r = {r:g} mm{" (core)" if i == 0 else " (next to the can)" if i == len(lib["radii_mm"]) - 1 else ""}</option>' for i, r in enumerate(lib["radii_mm"]))}</select>
      </div>
      <div class="ctl">
        <label for="shgt">Where in the cell (height)</label>
        <select id="shgt">{''.join(f'<option value="{h}"{" selected" if h == "mid" else ""}>{heights_lbl[h]}</option>' for h in lib["heights"])}</select>
      </div>
      <div class="ctl">
        <label for="stype">Short type (sets the layers; thermally the same within one wrap)</label>
        <select id="stype"><option>anode–cathode</option><option selected>Al–anode</option><option>Cu–cathode</option><option>Al–Cu</option></select>
      </div>
      <div class="ctl">
        <label for="sfoot">Footprint of the short</label>
        <select id="sfoot">{''.join(f'<option value="{w}"{" selected" if w in ("1.0", "1") else ""}>{float(w):g} × {float(w):g} mm</option>' for w in mult)}</select>
      </div>
      <div class="ctl">
        <label for="sdur">Duration</label>
        <select id="sdur"><option value="1">1 s (flash)</option><option value="10">10 s</option><option value="60">60 s</option><option value="1e9" selected>permanent</option></select>
      </div>
'''
anchor = '''      <div class="row">
        <button type="button" id="clear">Clear all shorts</button>'''
assert anchor in src
src = src.replace(anchor, ctl + anchor)
src = src.replace('<label for="risc">Soft short for the next click <span></span></label>',
                  '<label for="risc">Soft short for the next click: resistance <span></span></label>')

# tile and trace
src = src.replace('''    <div class="tile"><div class="k">Module current</div>''',
                  '''    <div class="tile" id="t-hp"><div class="k">Hottest point inside a shorted cell</div><div class="v" id="hp">–</div><div class="s" id="hp-at">3-D cell twin library</div></div>
    <div class="tile"><div class="k">Module current</div>''')
src = src.replace('''        <span><i style="background:var(--ink)"></i>hottest surface</span>''',
                  '''        <span><i style="background:var(--ink)"></i>hottest surface</span>
        <span><i style="background:#8e44ad"></i>hottest point inside (3-D)</span>''')

# --------------------------------------------------------------- script
src = src.replace('''  // ---------------- parameters''', '''  // ---------------- 3-D cell twin: unit-short response library (paper 1.2)
  const LIB = ''' + json.dumps(payload, separators=(",", ":")) + ''';
  const LT = LIB.t;
  function libEntry(ri, h) { const r = LIB.radii[ri]; return LIB.entries.find(e => e.r === r && e.h === h); }
  function riseAt(e, tau) {            // K per W, hottest point inside above the can mean, tau s after a 1 W step
    if (tau <= 0) return 0;
    if (tau <= LT[0]) return e.rise[0] * tau / LT[0];
    if (tau >= LT[LT.length - 1]) return e.rise[e.rise.length - 1];
    let i = 1; while (LT[i] < tau) i++;
    const f = (Math.log(tau) - Math.log(LT[i - 1])) / (Math.log(LT[i]) - Math.log(LT[i - 1]));
    return e.rise[i - 1] + f * (e.rise[i] - e.rise[i - 1]);
  }
  const meta = new Map();              // cell -> {R, ri, h, type, w, ton, dur, steps:[{t, dP}], P}
  function shortPower(k, s) {
    const m = meta.get(k); if (!m) return 0;
    return (t >= m.ton && t < m.ton + m.dur) ? Math.pow(ocv(s), 2) / m.R : 0;
  }
  function hotspot(k) {                // internal peak by superposition of the power steps
    const m = meta.get(k); if (!m) return null;
    const e = libEntry(m.ri, m.h), mult = LIB.mult[m.w] || 1;
    let d = 0; for (const s of m.steps) d += s.dP * riseAt(e, t - s.t);
    return Ts[k] + mult * d;
  }

  // ---------------- parameters''')
src = src.replace("Q_Ah: 15.4, R_cell: 0.005,", "Q_Ah: 15.4, R_cell: 0.009,")
# heat and leakage through meta
src = src.replace('''      if (Risc[k] > 0) Q += Math.pow(ocv(socG[grp[k]]), 2) / Risc[k];''',
                  '''      if (Risc[k] > 0) {
        const Pk = shortPower(k, socG[grp[k]]); Q += Pk;
        const m = meta.get(k);
        if (m && Math.abs(Pk - m.P) > Math.max(0.01 * m.P, 1e-4)) { m.steps.push({ t, dP: Pk - m.P }); m.P = Pk; }
      }''')
src = src.replace('''if (Risc[k] > 0) leak += ocv(socG[g]) / Risc[k]; }''',
                  '''if (Risc[k] > 0) leak += Math.sqrt(shortPower(k, socG[g]) * Risc[k]) / Risc[k]; }''')
# click: create meta
src = src.replace('''  cv.addEventListener("click", ev => { const k = pick(ev); if (k >= 0) Risc[k] = Risc[k] > 0 ? 0 : +ui.risc.value; });''',
'''  function addShort(k, R) {
    Risc[k] = R;
    meta.set(k, { R, ri: +$("sloc").value, h: $("shgt").value, type: $("stype").value, w: $("sfoot").value,
                  ton: t, dur: +$("sdur").value, steps: [], P: 0 });
  }
  cv.addEventListener("click", ev => { const k = pick(ev); if (k < 0) return; if (Risc[k] > 0) { Risc[k] = 0; meta.delete(k); } else addShort(k, +ui.risc.value); });''')
src = src.replace('''  $("clear").addEventListener("click", () => { Risc.fill(0); });''',
                  '''  $("clear").addEventListener("click", () => { Risc.fill(0); meta.clear(); });''')
# tooltip
src = src.replace('''      (Risc[k] > 0 ? `<br>short ${Risc[k]} Ω · ${(Math.pow(ocv(socG[grp[k]]), 2) / Risc[k]).toFixed(2)} W` : "<br>click to add a short");''',
'''      (Risc[k] > 0 ? (() => { const m = meta.get(k); const r = LIB.radii[m.ri];
        const big = shortPower(k, socG[grp[k]]) > 4 ? "<br><b>beyond soft-short scope: escalation not modelled</b>" : "";
        return `<br>${m.type} short ${m.R} Ω · ${shortPower(k, socG[grp[k]]).toFixed(2)} W${m.dur < 1e8 ? ` for ${m.dur} s` : ""}` +
               `<br>at r = ${r} mm, ${m.h.replace("_", " ")} · inside ${hotspot(k).toFixed(1)} °C` + big; })() : "<br>click to add a short");''')
# readouts
src = src.replace('''    $("tout").textContent = tout.toFixed(2) + " °C";''',
'''    $("tout").textContent = tout.toFixed(2) + " °C";
    let hp = -Infinity, hk = -1;
    for (const k of meta.keys()) { const v = hotspot(k); if (v > hp) { hp = v; hk = k; } }
    $("hp").textContent = hk >= 0 ? hp.toFixed(1) + " °C" : "no short";
    $("hp-at").textContent = hk >= 0 ? cellName(hk) + `, r = ${LIB.radii[meta.get(hk).ri]} mm` : "3-D cell twin library";
    $("t-hp").classList.toggle("alarm", hp > 80);
    let big = 0; for (const k of meta.keys()) big = Math.max(big, shortPower(k, socG[grp[k]]));
    if (hp > 130) $("hp-at").textContent = cellName(hk) + ": above ~130 °C the separator melts; linear model no longer valid";
    if (big > 4) $("hp-at").textContent = `short of ${big.toFixed(1)} W: beyond soft-short scope (escalation not modelled)`;''')
src = src.replace('''    return { ms: Ts[ks], mc: Tc[kc] };''', '''    return { ms: Ts[ks], mc: Tc[kc], hp: hk >= 0 ? hp : NaN };''')
# history: add hp
src = src.replace('''hist.push({ t, n1: Ts[NTC1], n2: Ts[NTC2], ms: m.ms, mc: m.mc }); lastHist = t; } }
  function frame''', '''hist.push({ t, n1: Ts[NTC1], n2: Ts[NTC2], ms: m.ms, mc: m.mc, hp: m.hp }); lastHist = t; } }
  function frame''')
src = src.replace('''const m = { ms: Math.max(...Ts), mc: Math.max(...Tc) };''',
                  '''let hpv = NaN; for (const k of meta.keys()) { const v = hotspot(k); if (!(v <= hpv)) hpv = v; }
          const m = { ms: Math.max(...Ts), mc: Math.max(...Tc), hp: hpv };''')
src = src.replace('''    for (const p of hist) for (const v of [p.n1, p.n2, p.ms, p.mc]) { mn = Math.min(mn, v); mx = Math.max(mx, v); }''',
                  '''    for (const p of hist) for (const v of [p.n1, p.n2, p.ms, p.mc, p.hp]) { if (!isFinite(v)) continue; mn = Math.min(mn, v); mx = Math.max(mx, v); }''')
src = src.replace('''      hist.forEach((p, n) => n ? trx.lineTo(X(p.t), Y(p[key])) : trx.moveTo(X(p.t), Y(p[key])));''',
                  '''      let pen = false;
      hist.forEach(p => { const v = p[key]; if (!isFinite(v)) { pen = false; return; } pen ? trx.lineTo(X(p.t), Y(v)) : trx.moveTo(X(p.t), Y(v)); pen = true; });''')
src = src.replace('''    line("mc", css("--warn"), 2.5);''', '''    line("hp", "#8e44ad", 2.5); line("mc", css("--warn"), 2.5);''')
# example short through meta
src = src.replace('''  Risc[11 * NP + 4] = 3;''', '''  addShort(11 * NP + 4, 3);''')
# move addShort before use: it is defined in the controls section, which precedes the loop
# note
src = src.replace('''This is not the 3-D model of paper 1.2. It shows what the module's sensors can and cannot see.</p>''',
'''<b>Linked to the 3-D cell twin.</b> When you place a short, choose where it sits in the cell (radius, height), its type, footprint and duration. The hottest point inside the shorted cell is then computed from the unit-short response library of the 3-D model (superposition of the short's power history) on top of the cell's can temperature from this module model, with the footprint factor of the refined-patch study. The library assumes the short's heat stays in the shorted cell over the time shown. It shows what the module's sensors can and cannot see, and what the cell's interior does meanwhile.</p>''')
src = src.replace("Heat is ohmic (I²R, R = 5 mΩ per cell)", "Heat is ohmic (I²R, R = 9 mΩ per cell, the measured 10 s pulse resistance)")
src = src.replace('<option value="1">1 Ω (≈ 11 W)</option>', '<option value="1">1 Ω (≈ 11 W, beyond soft-short scope)</option>')
(HERE / "module_twin.html").write_text(src)
print("written", len(src), "bytes; footprint multipliers", mult)

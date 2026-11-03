"""Build module_twin_3d.html from twin3d_src.html.

Injects: the unit-short response library of the 3-D cell twin (LIB), the module
model parameters shared with the training simulator (SIM), and, once trained,
the detector network (NN + DETECTOR code).

    python build_twin3d.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = next(p for p in (HERE.parent / "cell_twin" / "out", HERE.parent / "model" / "out") if p.exists())

# library: reuse what patch_twin.py puts into the 2-D twin (rise = peak - side_mean, footprint factors)
v2 = (HERE / "module_twin.html").read_text()
m = re.search(r"const LIB = (\{.*?\});\n", v2)
LIB = json.loads(m.group(1))
raw = json.loads((OUT / "response_library.json").read_text())
LIB["z"] = {e["height"]: round(e["z"], 2) for e in raw["entries"]}

SIM_PATH = HERE / "sim_params.json"
SIM = json.loads(SIM_PATH.read_text())

NN_PATH = HERE / "detector_weights.json"
DET_JS = HERE / "detector.js"
NN = json.loads(NN_PATH.read_text()) if NN_PATH.exists() else None

src = (HERE / "twin3d_src.html").read_text()
src = src.replace("/*LIB*/null", json.dumps(LIB, separators=(",", ":")))
src = src.replace("/*SIM*/null", json.dumps(SIM, separators=(",", ":")))
F = json.loads((OUT / "fingerprint.json").read_text())
sel = [i for i, q in enumerate(F["positions"]) if q["side"] == "inner"]
FP = dict(r=[round(F["positions"][i]["r"], 3) for i in sel], z=[F["positions"][i]["z"] for i in sel],
          s=[round(F["positions"][i]["s"], 1) for i in sel])
for case, key in (("A_literal", "A"), ("B_calibrated", "B")):
    C = F["cases"][case]; Zc = C["Z_cell"][0][0]
    FP[key] = dict(th=[round(C["Z_th"][0][0][i] / Zc, 4) for i in sel], tr=[round(C["Z_tr"][0][0][i] / Zc, 4) for i in sel])
src = src.replace("/*FP*/null", json.dumps(FP, separators=(",", ":")))
src = src.replace("/*NN*/null", json.dumps(NN, separators=(",", ":")) if NN else "null")
src = src.replace("/*DETECTOR*/", DET_JS.read_text() if (NN and DET_JS.exists()) else "function makeDetector() { return null; }")
(HERE / "module_twin_3d.html").write_text(src)
print("written", len(src), "bytes; detector", "on" if NN else "off")

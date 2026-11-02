#!/usr/bin/env python3
"""k5twin self-test against Arnold Sequeira's board results (Aug-Sep 2026).

usage: python3 tests/selftest.py <path to unzipped 05_S32K5_Temperature_estimation_200_cells>

1. The 200-row CPU bundle, executed on the emulated Cortex-M7, must give the
   board's float32 results (CPU_vs_NPU_Report_V1, Table 2) to 4 decimals.
2. The stored calibration must reproduce its own measured points within the
   leave-one-out error it reports.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import k5twin  # noqa: E402

BOARD_CPU_DEGC = [14.5108, 23.3124, 22.0816, 23.2740, 21.0552]   # report Table 2
T_SCALE, T_MIN = 19.455988, 7.582792

root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
ok = True
if root:
    hdr = next(root.rglob("data/cpu_glow/models/*.h"))
    b = k5twin.Bundle(hdr)
    inp = k5twin.read_inputs(HERE / "examples" / "hev2_5x40_batch200.json")
    res = k5twin.run(b, inp, Path(tempfile.mkdtemp()), trace=False)
    t = res["outputs"]["temperature"].ravel()[:5] * T_SCALE + T_MIN
    d = np.abs(np.round(t, 4) - BOARD_CPU_DEGC)
    print("emulated M7:", np.round(t, 4).tolist())
    print("board      :", BOARD_CPU_DEGC)
    print("max |diff| :", d.max(), "degC")
    ok &= bool(d.max() <= 1e-4)

cal = json.loads((HERE / "calibration" / "m7_200MHz.json").read_text())
c = cal["cpu"]
worst = max(abs(p["predicted_us"] - p["measured_us"]) / p["measured_us"] for p in c["points"])
print(f"calibration CPU: worst fit error {100 * worst:.1f} %, leave-one-out {100 * c['loo_max_rel_err']:.1f} %")
ok &= worst <= c["loo_max_rel_err"] + 1e-9
print("SELFTEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

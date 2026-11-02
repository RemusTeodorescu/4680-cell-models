#!/usr/bin/env python3
"""k5twin GUI - a local web interface to the S32K566 M7 + eIQ Auto twin.

    python3 k5twin_gui.py [--root "<folder with eIQ Auto projects>"] [--port 8765]

then open http://localhost:8765 in a browser (from Windows when running in
WSL). Everything runs locally; nothing is sent anywhere.
"""
import argparse
import json
import re
import subprocess
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import k5twin

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
DEBUG_PROCS = []


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def scan(root: Path):
    bundles, inputs, tflites = [], [], []
    for h in sorted(root.rglob("*.h")):
        if "/build" in h.as_posix() or "CMakeFiles" in h.as_posix():
            continue
        o, w = h.with_suffix(".o"), h.with_name(h.stem + ".weights.txt")
        if o.exists() and w.exists() and "Model name" in h.read_text(errors="ignore")[:4000]:
            try:
                b = k5twin.Bundle(h)
            except BaseException:
                continue
            bundles.append(dict(path=str(h), name=b.name, neutron=b.neutron,
                                rel=str(h.relative_to(root)), rows=b.placeholders[0].shape[1]
                                if len(b.placeholders[0].shape) > 1 else 1,
                                guess=guess_companions(h, b)))
    for p in list((HERE / "examples").glob("*.json")) + list(root.rglob("*.npz")):
        inputs.append(str(p))
    for t in sorted(root.rglob("*.tflite")):
        tflites.append(str(t))
    return dict(root=str(root), bundles=bundles, inputs=inputs, tflites=tflites)


def guess_companions(h: Path, b):
    """Board ELF/.map, eiqa.log, int8 TFLite and layer count, from the usual
    eIQ Auto project layout: <proj>/eiqa_runtime_app/data/<variant>/models/x.h"""
    g = {}
    parts = h.parts
    if "eiqa_runtime_app" in parts:
        app = Path(*parts[:parts.index("eiqa_runtime_app") + 1])
        proj = app.parent
        variant = h.parent.parent.name
        for bdir in sorted(app.glob(f"build_{variant}*")):
            for f in sorted((bdir / "bin").glob("*")):
                if f.suffix == "" and f.with_name(f.name + ".map").exists():
                    g["board_elf"], g["board_map"] = str(f), str(f) + ".map"
                    break
            if "board_elf" in g:
                break
        log = proj / "eiqa_offline" / "eiqa.log"
        if log.exists() and b.neutron:
            g["neutron_log"] = str(log)
        if b.neutron:
            for t in sorted(proj.rglob("*.tflite")):
                if re.search(r"_p0\.tflite$", t.name) and "transform" in t.parts:
                    g["tflite"] = str(t)
                    break
        # layers: count the FC kernels in a CPU bundle of the same project
        for o in sorted(app.glob("data/*/models/*.o")):
            r = subprocess.run([k5twin.GCC + "nm", str(o)], capture_output=True, text=True)
            if "eiqa_neutron_graph_operator" in r.stdout:
                continue
            n = len(re.findall(r"libjit_fc_f_\d+", r.stdout))
            if n:
                g["layers"] = n
                break
    return g


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
def do_run(req):
    t0 = time.time()
    extra = [str(HERE / "opt" / "fast_expf.c")] if req.get("fast_expf") else []
    layers = int(req["layers"]) if str(req.get("layers") or "").strip() else None
    r = k5twin.analyze(req["bundle"], inputs=req.get("inputs") or None,
                       board_elf=req.get("board_elf") or None, board_map=req.get("board_map") or None,
                       layers=layers, neutron_log=req.get("neutron_log") or None,
                       tflite=req.get("tflite") or None, extra_sources=extra)
    cal = k5twin.load_cal(k5twin.CAL_DEFAULT) or {}
    r["calibration"] = dict(cpu=cal.get("cpu", {}), npu={k: v for k, v in cal.get("npu", {}).items()
                                                          if k != "points"})
    if req.get("inputs") and str(req["inputs"]).endswith(".json"):
        j = json.loads(Path(req["inputs"]).read_text())
        if "_truth_degC" in j:
            r["truth"] = j["_truth_degC"]
    r["wall_s"] = round(time.time() - t0, 2)
    r["label"] = (Path(req["bundle"]).stem + (" + fast expf" if extra else ""))
    return r


def do_debug(req):
    port = int(req.get("port") or 1234)
    work = Path(req.get("work") or (HERE / "k5twin_debug"))
    b = k5twin.Bundle(req["bundle"])
    elf = k5twin.build(b, work)
    k5twin.pack_inputs(b, k5twin.read_inputs(req["inputs"]) if req.get("inputs") else {}).tofile(work / "k5in.bin")
    for p in DEBUG_PROCS:
        p.terminate()
    DEBUG_PROCS.clear()
    DEBUG_PROCS.append(subprocess.Popen(k5twin.qemu_cmd(elf.name, gdb=port), cwd=work,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return dict(gdb=f"gdb-multiarch {elf} -ex 'target remote :{port}' -ex 'break {b.name}' "
                    f"-ex 'break k5_mark_end' -ex continue", port=port)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, (HERE / "gui" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/scan":
            root = Path(parse_qs(u.query).get("root", [str(ROOT)])[0]).expanduser()
            if not root.is_dir():
                return self._send(400, json.dumps(dict(error=f"not a folder: {root}")))
            return self._send(200, json.dumps(scan(root)))
        self._send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/run":
                return self._send(200, json.dumps(do_run(req)))
            if self.path == "/api/debug":
                return self._send(200, json.dumps(do_debug(req)))
        except BaseException as e:  # SystemExit from k5twin carries the message
            msg = str(e) if str(e) else traceback.format_exc()
            return self._send(500, json.dumps(dict(error=msg[-4000:])))
        self._send(404, "{}")


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder to search for eIQ Auto bundles")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    ROOT = Path(a.root).expanduser().absolute()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://localhost:{a.port}"
    print(f"k5twin GUI on {url}  (root: {ROOT})  Ctrl-C to stop")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in DEBUG_PROCS:
            p.terminate()


if __name__ == "__main__":
    main()

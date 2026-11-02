#!/usr/bin/env python3
"""k5twin - a software twin of the S32K566 Cortex-M7 + eIQ Auto (Glow) target.

It takes the compiled eIQ Auto Glow bundle (the .h/.o/.weights.txt that
graph_build writes for target m7) and, before anything is flashed:

  run       executes the bundle's own machine code on an emulated Cortex-M7
            (QEMU mps2-an500) and returns the outputs        -> functional test
  debug     the same, stopped at reset with a GDB server       -> debugging
  memory    places the bundle into the real S32K566 M7_0 memory map, using a
            board firmware ELF/.map as the baseline            -> memory budget
  time      counts the instructions the bundle executes and converts them to
            board microseconds with a calibration fitted on measured runs
  calibrate fits that calibration from bundles with measured board times
  npu       NPU (Neutron) bundles: runs the CPU part, counts NPU dispatches,
            predicts time; int8 numerics from the quantised TFLite model

Requirements (WSL Ubuntu or Linux): qemu-system-arm, an arm-none-eabi GCC with
newlib (rdimon), python3 with numpy. Optional: tflite-runtime or tensorflow
for 'npu --tflite'.

What it does not model: the NPU's internal execution (no public model of the
Neutron microcode), caches and MRAM wait states (absorbed by the calibration),
peripherals, the eIQ Session/runtime layer (its fixed cost is in the
calibration intercept), and the R52 cores.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "harness"
CAL_DEFAULT = HERE / "calibration" / "m7_200MHz.json"

GCC = os.environ.get("K5_GCC_PREFIX", "arm-none-eabi-")
QEMU = os.environ.get("K5_QEMU", "qemu-system-arm")
# Board firmware links the hard-float single-precision newlib (v7e-m+fp/hard),
# see the board .map; use the same multilib so expf etc. match.
CPU_FLAGS = ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-sp-d16", "-mfloat-abi=hard"]

# S32K566 Cortex-M7_0 regions, from the board linker map (eIQ Auto 3.14.1,
# S32K5 PREMIUM startup, k566_m7). Bytes.
S32K566_M7_0 = {
    "int_itcm_cm7":  (0x00000000, 0x8000),
    "int_dtcm_cm7":  (0x20000000, 0xF000),
    "int_cmram_cm7_c0": (0x08000800, 0x3FF800),
    "int_sram_cm7_c0":  (0x21000000, 0x80000),
    "int_sram_no_cacheable_cm7_c0": (0x21080000, 0x7FF00),
}


# --------------------------------------------------------------------------
# Bundle header parsing
# --------------------------------------------------------------------------
class Placeholder:
    def __init__(self, name, dtype, scale, zero, shape, nbytes, offset):
        self.name, self.dtype, self.scale, self.zero = name, dtype, scale, zero
        self.shape, self.nbytes, self.offset = shape, nbytes, offset

    @property
    def np_dtype(self):
        return {"float": np.float32, "i8": np.int8, "u8": np.uint8,
                "i16": np.int16, "i32": np.int32}[self.dtype]

    def to_dict(self):
        return dict(name=self.name, dtype=self.dtype, scale=self.scale,
                    zero_point=self.zero, shape=self.shape, bytes=self.nbytes,
                    offset=self.offset)


class Bundle:
    """One Glow bundle: <name>.h, <name>.o, <name>.weights.txt in one folder."""

    def __init__(self, header):
        self.h = Path(header)
        if self.h.is_dir():
            hs = sorted(self.h.glob("*.h"))
            if len(hs) != 1:
                raise SystemExit(f"{self.h}: expected exactly one bundle header, found {len(hs)}; pass the .h file")
            self.h = hs[0]
        txt = self.h.read_text()
        self.name = re.search(r'Model name: "([^"]+)"', txt).group(1)
        self.obj = self.h.with_name(self.name + ".o")
        self.weights = self.h.with_name(self.name + ".weights.txt")
        for p in (self.obj, self.weights):
            if not p.exists():
                raise SystemExit(f"missing {p}")
        up = self.name.upper()

        def macro(m):
            r = re.search(rf"#define\s+{up}_{m}\s+(\d+)", txt)
            return int(r.group(1)) if r else None
        self.const_size = macro("CONSTANT_MEM_SIZE")
        self.mut_size = macro("MUTABLE_MEM_SIZE")
        self.act_size = macro("ACTIVATIONS_MEM_SIZE")
        self.align = macro("MEM_ALIGN") or 64
        self.neutron = ("NeutronOpConfig" in txt or "eiqa_neutron_graph_operator" in txt
                        or "eiqa_neutron_graph_operator" in sh([GCC + "nm", str(self.obj)]).stdout)
        self.placeholders = []
        for blk in re.finditer(
                r'//\s+Name: "([^"]+)"\s*\n(?://[^\n]*\n)*?//\s+Type: ([^\n]+)\n'
                r'//\s+Size: \d+ \(elements\)\s*\n//\s+Size: (\d+) \(bytes\)\s*\n'
                r'//\s+Offset: (\d+) \(bytes\)', txt):
            name, typ, nbytes, off = blk.group(1), blk.group(2), int(blk.group(3)), int(blk.group(4))
            shape = [int(x) for x in re.search(r"<([^>]+)>", typ).group(1).split(" x ")]
            m = re.match(r"(\w+)(?:\[S:([-\d.eE]+) O:(-?\d+)\])?", typ.strip())
            dtype = m.group(1)
            scale = float(m.group(2)) if m.group(2) else None
            zero = int(m.group(3)) if m.group(3) else None
            self.placeholders.append(Placeholder(name, dtype, scale, zero, shape, nbytes, off))
        if not self.placeholders:
            raise SystemExit(f"{self.h}: no placeholders found in header")

    def summary(self):
        return dict(bundle=self.name, header=str(self.h), neutron=self.neutron,
                    constant_bytes=self.const_size, mutable_bytes=self.mut_size,
                    activation_bytes=self.act_size, align=self.align,
                    placeholders=[p.to_dict() for p in self.placeholders])


# --------------------------------------------------------------------------
# Build and run
# --------------------------------------------------------------------------
def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(" ".join(map(str, cmd)) + "\n" + r.stdout + r.stderr)
        raise SystemExit(f"command failed ({r.returncode})")
    return r


def build(b: Bundle, work: Path, linker=HARNESS / "an500.ld", extra_objs=(), extra_sources=()):
    """extra_sources: C files linked ahead of libm, e.g. opt/fast_expf.c to
    replace a library routine the bundle calls (as it would be in the board
    firmware's CMake sources)."""
    work.mkdir(parents=True, exist_ok=True)
    extra_objs = list(extra_objs)
    for i, src in enumerate(extra_sources):
        o = work / f"extra{i}.o"
        sh([GCC + "gcc", *CPU_FLAGS, "-O2", "-g", "-c", str(src), "-o", o])
        extra_objs.append(o)
    cfg = work / "k5_config.h"
    cfg.write_text(
        "#define GLOW_MEM_ALIGN(n) __attribute__((aligned(n)))\n"
        f"#define K5_ENTRY {b.name}\n"
        f"#define K5_CONSTANT_MEM_SIZE {b.const_size}\n"
        f"#define K5_MUTABLE_MEM_SIZE {b.mut_size}\n"
        f"#define K5_ACTIVATIONS_MEM_SIZE {b.act_size}\n"
        f"#define K5_MEM_ALIGN {b.align}\n"
        f'#define K5_WEIGHTS_FILE "{b.weights.resolve()}"\n'
        '#define K5_IN_FILE "k5in.bin"\n#define K5_OUT_FILE "k5out.bin"\n'
        + ("#define K5_NEUTRON 1\n" if b.neutron else "")
        + "int K5_ENTRY(unsigned char*, unsigned char*, unsigned char*, unsigned char*);\n")
    o1, o2, elf = work / "startup.o", work / "harness.o", work / "twin.elf"
    sh([GCC + "gcc", *CPU_FLAGS, "-O2", "-g", "-c", HARNESS / "startup.c", "-o", o1])
    sh([GCC + "gcc", *CPU_FLAGS, "-O2", "-g", f"-I{work}", "-c", HARNESS / "harness.c", "-o", o2])
    sh([GCC + "gcc", *CPU_FLAGS, "--specs=rdimon.specs", "-nostartfiles", "-T", linker,
        "-Wl,--gc-sections", "-Wl,--no-warn-execstack", f"-Wl,-Map={work / 'twin.map'}",
        o1, o2, b.obj, *extra_objs, "-lm", "-o", elf])
    return elf


def _match(name, inputs):
    """Placeholder names change through onnx2tf/Neutron conversion
    ('space' -> 'serving_default_space_0'); accept the original name too."""
    if name in inputs:
        return name
    base = re.sub(r"^serving_default_|_\d+$", "", name)
    return base if base in inputs else None


def _std_round(x):
    """C++ std::round: half away from zero (as in main_neutron.cpp)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def pack_inputs(b: Bundle, inputs: dict):
    buf = np.zeros(b.mut_size, np.uint8)
    for p in b.placeholders:
        key = _match(p.name, inputs)
        if key is None:
            continue
        v = np.asarray(inputs[key], np.float64).ravel()
        n = int(np.prod(p.shape))
        if v.size == 1:
            v = np.full(n, v[0])
        if v.size != n:
            raise SystemExit(f"input '{p.name}': {v.size} values, bundle expects {n} {p.shape}")
        if p.dtype == "float":
            raw = v.astype(np.float32)
        else:
            if p.scale is None:
                raw = v.astype(p.np_dtype)
            else:
                info = np.iinfo(p.np_dtype)
                raw = np.clip(_std_round(v / p.scale) + p.zero, info.min, info.max).astype(p.np_dtype)
        buf[p.offset:p.offset + raw.nbytes] = raw.view(np.uint8)
    return buf


def unpack(b: Bundle, buf, names=None):
    out = {}
    for p in b.placeholders:
        if names and p.name not in names:
            continue
        raw = buf[p.offset:p.offset + p.nbytes].view(p.np_dtype).reshape(p.shape)
        if p.dtype != "float" and p.scale is not None:
            out[p.name] = (raw.astype(np.float64) - p.zero) * p.scale
        else:
            out[p.name] = raw.astype(np.float64)
    return out


def qemu_version():
    r = subprocess.run([QEMU, "--version"], capture_output=True, text=True)
    m = re.search(r"version (\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def qemu_cmd(elf, trace_log=None, gdb=False):
    cmd = [QEMU, "-M", "mps2-an500", "-cpu", "cortex-m7", "-nographic",
           "-monitor", "none", "-serial", "none",
           "-semihosting-config", "enable=on,target=native", "-kernel", str(elf)]
    if trace_log:
        if qemu_version() >= (8, 1):
            cmd += ["-accel", "tcg,one-insn-per-tb=on"]
        else:
            cmd += ["-singlestep"]
        cmd += ["-d", "exec,nochain", "-D", str(trace_log)]
    if gdb:
        cmd += ["-S", "-gdb", f"tcp::{gdb}"]
    return cmd


def symbols(elf):
    syms = []
    r = sh([GCC + "nm", "-n", "-S", "-C", str(elf)])
    for line in r.stdout.splitlines():
        p = line.split(maxsplit=3)
        if len(p) == 4 and p[2] in "tTwW":
            syms.append((int(p[0], 16), int(p[1], 16), p[3]))
    return syms


CLASSES = ("fp_div", "fp_arith", "fp_mem", "mem", "branch", "int")


def classify(mn):
    m = mn.split(".")[0]
    if m in ("vdiv", "vsqrt"):
        return "fp_div"
    if m in ("vldr", "vstr", "vldmia", "vldmdb", "vstmia", "vstmdb", "vldm", "vstm", "vpush", "vpop"):
        return "fp_mem"
    if m.startswith("v"):
        return "fp_arith"
    if re.match(r"(ldr|str|ldm|stm|push|pop|ldrex|strex|pld)", m):
        return "mem"
    if re.match(r"(b|bl|blx|bx|cbz|cbnz|tbb|tbh)(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?(\.[nw])?$", mn):
        return "branch"
    return "int"


def disasm_classes(elf):
    r = sh([GCC + "objdump", "-d", "--no-show-raw-insn", str(elf)])
    cls = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(\S+)", line)
        if m:
            cls[int(m.group(1), 16)] = classify(m.group(2))
    return cls


def count_trace(log, elf):
    syms = symbols(elf)
    addr = {n: a for a, s, n in syms}
    a0, a1 = addr["k5_mark_begin"], addr["k5_mark_end"]
    rx = re.compile(rb"\[[0-9a-f]+/([0-9a-f]+)/")
    pcs = {}
    inside, n = False, 0
    with open(log, "rb") as f:
        for line in f:
            m = rx.search(line)
            if not m:
                continue
            pc = int(m.group(1), 16)
            if pc == a0:
                inside = True
                continue
            if pc == a1 and inside:
                break
            if inside:
                n += 1
                pcs[pc] = pcs.get(pc, 0) + 1
    cmap = disasm_classes(elf)
    by_class = {c: 0 for c in CLASSES}
    for pc, c in pcs.items():
        by_class[cmap.get(pc, "int")] += c
    by_fn = {}
    starts = [s[0] for s in syms]
    import bisect
    for pc, c in pcs.items():
        i = bisect.bisect_right(starts, pc) - 1
        name = syms[i][2] if i >= 0 and pc < syms[i][0] + max(syms[i][1], 1) else "?"
        by_fn[name] = by_fn.get(name, 0) + c
    return n, dict(sorted(by_fn.items(), key=lambda kv: -kv[1])), by_class


def run(b: Bundle, inputs: dict, work: Path, trace=False, extra_sources=()):
    elf = build(b, work, extra_sources=extra_sources)
    pack_inputs(b, inputs).tofile(work / "k5in.bin")
    log = work / "exec.log" if trace else None
    r = subprocess.run(qemu_cmd(elf, log), cwd=work, capture_output=True, text=True, timeout=1800)
    con = r.stdout + r.stderr
    if "K5TWIN_FAULT" in con or "K5TWIN_ERROR" in con or r.returncode != 0:
        raise SystemExit(f"target run failed (rc={r.returncode}):\n{con}")
    buf = np.fromfile(work / "k5out.bin", np.uint8)
    res = dict(
        outputs=unpack(b, buf),
        rc=int(re.search(r"K5TWIN_RC (-?\d+)", con).group(1)),
        stack_bytes=int(re.search(r"K5TWIN_STACK_BYTES (\d+)", con).group(1)),
        console=con,
        batch_rows=b.placeholders[0].shape[1] if len(b.placeholders[0].shape) > 1 else 1,
    )
    m = re.search(r"K5TWIN_NEUTRON_CALLS (\d+)", con)
    if m:
        res["neutron_calls"] = int(m.group(1))
        res["neutron_ops"] = [dict(zip(("type", "opts", "nin", "nout", "bytes_in", "bytes_out"),
                                       (g[0], *map(int, g[1:]))))
                              for g in re.findall(r"K5TWIN_NEUTRON_CALL \d+ type=(\S+) opts=(\d+) in=(\d+) "
                                                  r"out=(\d+) bytes_in=(\d+) bytes_out=(\d+)", con)]
    if trace:
        res["instructions"], res["by_function"], res["by_class"] = count_trace(log, elf)
        log.unlink()
    return res


# --------------------------------------------------------------------------
# Timing model
# --------------------------------------------------------------------------
def load_cal(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def npu_bytes(res):
    return sum(o["bytes_in"] for o in res.get("neutron_ops", []))


def neutron_log_estimate_us(path):
    """NPU-only latency estimate printed by neutron-converter in eiqa.log
    ('Latency estimation = 0.044 (ms) (@ 400.000 MHz) (NPU only)'); last one."""
    if not path:
        return None
    v = re.findall(r"Latency estimation\s*=\s*([\d.]+)\s*\(ms\)[^\n]*NPU only", Path(path).read_text(errors="ignore"))
    return 1000.0 * float(v[-1]) if v else None


def predict_us(cal, res, neutron, layers=None, npu_est_us=None):
    """Board-time prediction (us) and a short note on how it was obtained.

    CPU: t = a + sum_k c_k n_k, n_k = executed instructions of class k.
    NPU: t = a_npu + sum_k c_k n_k (CPU-side ops) + b * bytes handed to the
         NPU + d * layers (if the layer count is given).
    """
    if not cal or "cpu" not in cal:
        return None
    c = cal["cpu"]
    cls = res["by_class"]
    cpu_part = sum(c["per_class_us"][k] * cls.get(k, 0) for k in CLASSES)
    if not neutron:
        return c["intercept_us"] + cpu_part, dict(note=c["note"], loo_max_rel_err=c["loo_max_rel_err"])
    n = cal.get("npu")
    if not n:
        return None
    m = n["with_layers"] if layers is not None else n["bytes_only"]
    t = m["intercept_us"] + cpu_part + m["us_per_byte"] * npu_bytes(res)
    if layers is not None:
        t += m["us_per_layer"] * layers
    note = n["note"] + (" (layer count given)" if layers is not None else " (no layer count: bytes-only model)")
    rows = res.get("batch_rows", 1)
    if rows > 1:
        if npu_est_us is not None:
            t += npu_est_us
            note += f"; batch {rows}: + {npu_est_us:.1f} us NPU compute from the neutron-converter estimate"
        else:
            note += (f"; WARNING batch {rows}: calibration had batch 1, NPU compute for the batch is not "
                     f"included - pass --neutron-log eiqa.log")
    if res.get("neutron_calls", 1) > 1:
        note += f"; WARNING {res['neutron_calls']} NPU dispatches, calibration only had 1 per inference"
    return t, dict(note=note, loo_max_rel_err=m["loo_max_rel_err"])


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def board_baseline(board_elf, board_map):
    """Bytes used per S32K566 region by a board firmware, and the bytes of its
    own GLOW_WEIGHTS / GLOW_IO sections (to swap in a new bundle)."""
    r = sh([GCC + "readelf", "-S", "-W", str(board_elf)])
    used = {k: 0 for k in S32K566_M7_0}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+\S+\s+(\S*)", line)
        if not m or "A" not in m.group(5):
            continue
        addr, size = int(m.group(3), 16), int(m.group(4), 16)
        for k, (o, l) in S32K566_M7_0.items():
            if o <= addr < o + l:
                used[k] += size
    # load images of initialised RAM sections also sit in MRAM
    r2 = sh([GCC + "objdump", "-h", "-w", str(board_elf)])
    for line in r2.stdout.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0].isdigit():
            size, vma, lma = int(p[2], 16), int(p[3], 16), int(p[4], 16)
            flags = " ".join(p[6:])
            if vma != lma and "LOAD" in flags:
                o, l = S32K566_M7_0["int_cmram_cm7_c0"]
                if o <= lma < o + l:
                    used["int_cmram_cm7_c0"] += size
    mt = Path(board_map).read_text(errors="ignore")
    gw = sum(int(x, 16) for x in re.findall(r"^\s*GLOW_WEIGHTS\s+0x[0-9a-f]+\s+0x([0-9a-f]+)", mt, re.M))
    gio = sum(int(x, 16) for x in re.findall(r"^\s*GLOW_IO\s+0x[0-9a-f]+\s+0x([0-9a-f]+)", mt, re.M))
    return used, gw, gio


def aligned(n, a):
    return (n + a - 1) // a * a


def memory_report(b: Bundle, board_elf=None, board_map=None, stack=None):
    rep = dict(bundle=b.name,
               constant_weights=b.const_size, mutable_io=b.mut_size,
               activations=b.act_size, stack_peak=stack)
    if board_elf and board_map:
        used, gw, gio = board_baseline(board_elf, board_map)
        new_w = aligned(b.const_size, b.align)
        new_io = aligned(b.mut_size, b.align) + aligned(b.act_size, b.align)
        used = dict(used)
        used["int_cmram_cm7_c0"] += new_w - gw
        used["int_sram_cm7_c0"] += new_io - gio
        rep["board_baseline"] = dict(elf=str(board_elf), glow_weights_replaced=gw, glow_io_replaced=gio)
        rep["regions"] = {k: dict(used=v, size=S32K566_M7_0[k][1],
                                  pct=round(100.0 * v / S32K566_M7_0[k][1], 1),
                                  ok=v <= S32K566_M7_0[k][1])
                          for k, v in used.items()}
    return rep


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def analyze(bundle, inputs=None, board_elf=None, board_map=None, layers=None, neutron_log=None,
            tflite=None, extra_sources=(), cal=CAL_DEFAULT, trace=True):
    """One complete twin run, returned as a JSON-serialisable dict (used by the GUI)."""
    b = Bundle(bundle)
    inp = read_inputs(inputs) if inputs else {}
    work = Path(tempfile.mkdtemp(prefix="k5an_"))
    try:
        res = run(b, inp, work, trace=trace, extra_sources=extra_sources)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    t = (predict_us(load_cal(cal), res, b.neutron, layers, neutron_log_estimate_us(neutron_log))
         if trace else None)
    mem = memory_report(b, board_elf, board_map, res["stack_bytes"])
    out = dict(summary=b.summary(), rc=res["rc"], stack_bytes=res["stack_bytes"],
               batch_rows=res["batch_rows"], memory=mem,
               instructions=res.get("instructions"), by_class=res.get("by_class"),
               by_function=dict(list(res.get("by_function", {}).items())[:12]),
               neutron_calls=res.get("neutron_calls"), neutron_ops=res.get("neutron_ops"),
               npu_bytes=npu_bytes(res) if b.neutron else None,
               predicted_us=t[0] if t else None, time_note=t[1] if t else None,
               outputs={k: v.ravel().tolist() for k, v in res["outputs"].items()},
               extra_sources=[str(x) for x in extra_sources])
    if tflite and b.neutron:
        out["tflite_outputs"] = {k: v.ravel().tolist() for k, v in tflite_int8(tflite, b, inp).items()}
    return out


def read_inputs(path):
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".npz":
        return dict(np.load(p))
    if p.suffix == ".json":
        return {k: np.asarray(v) for k, v in json.loads(p.read_text()).items()}
    raise SystemExit("inputs: use .npz or .json  {placeholder_name: values}")


def fmt_report(b, res, mem, t):
    L = [f"bundle      {b.name}  ({'NPU/Neutron' if b.neutron else 'CPU/float32'})",
         f"rc          {res['rc']}"]
    for k, v in res["outputs"].items():
        pass
    if "instructions" in res:
        L.append(f"instructions {res['instructions']:,} executed by the bundle")
        top = list(res["by_function"].items())[:6]
        L += [f"   {n:<45s} {c:>10,}" for n, c in top]
    if "neutron_calls" in res:
        L.append(f"NPU dispatches {res['neutron_calls']}, bytes handed to the NPU {npu_bytes(res):,}")
        if "by_class" in res:
            L.append(f"CPU-side instructions (ops not offloaded) {res['instructions']:,}")
    if t:
        L.append(f"predicted board time  {t[0]:,.1f} us   (+/- about {100 * t[1]['loo_max_rel_err']:.0f} %, "
                 f"leave-one-out on the calibration set)")
        L.append(f"   calibration: {t[1]['note']}")
    L.append(f"stack peak  {res['stack_bytes']} B (bundle call only)")
    L.append(f"weights {b.const_size:,} B   io {b.mut_size:,} B   activations {b.act_size:,} B")
    if mem and "regions" in mem:
        L.append("S32K566 M7_0 memory with this bundle swapped into the board firmware:")
        for k, r in mem["regions"].items():
            if r["used"]:
                L.append(f"   {k:<30s} {r['used']:>9,} / {r['size']:>9,} B  {r['pct']:5.1f} %  {'ok' if r['ok'] else 'OVERFLOW'}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_info(a):
    print(json.dumps(Bundle(a.bundle).summary(), indent=2))


def cmd_run(a):
    b = Bundle(a.bundle)
    work = Path(a.work or tempfile.mkdtemp(prefix="k5twin_"))
    res = run(b, read_inputs(a.inputs), work, trace=not a.no_time, extra_sources=a.link)
    t = (predict_us(load_cal(a.cal), res, b.neutron, a.layers, neutron_log_estimate_us(a.neutron_log))
         if "instructions" in res else None)
    mem = memory_report(b, a.board_elf, a.board_map, res["stack_bytes"])
    print(fmt_report(b, res, mem, t))
    if a.out:
        np.savez(a.out, **res["outputs"])
        print(f"outputs -> {a.out}")
    if a.json:
        j = {k: v for k, v in res.items() if k not in ("outputs", "console")}
        j["outputs"] = {k: v.ravel().tolist() for k, v in res["outputs"].items()}
        j["predicted_us"] = t[0] if t else None
        j["memory"] = mem
        Path(a.json).write_text(json.dumps(j, indent=1))
        print(f"report -> {a.json}")
    if not a.work:
        shutil.rmtree(work, ignore_errors=True)


def cmd_debug(a):
    b = Bundle(a.bundle)
    work = Path(a.work or "k5twin_debug")
    elf = build(b, work)
    pack_inputs(b, read_inputs(a.inputs)).tofile(work / "k5in.bin")
    print(f"ELF with symbols: {elf.resolve()}\n"
          f"In a second terminal:\n"
          f"  gdb-multiarch {elf.resolve()} -ex 'target remote :{a.port}' "
          f"-ex 'break {b.name}' -ex 'break k5_mark_end' -ex continue\n"
          f"Outputs are written to {work.resolve()}/k5out.bin when main returns.\n"
          f"Starting QEMU (halted at reset, Ctrl-A X to quit) ...")
    os.chdir(work)
    os.execvp(QEMU, qemu_cmd(elf.name, gdb=a.port))


def cmd_memory(a):
    b = Bundle(a.bundle)
    print(json.dumps(memory_report(b, a.board_elf, a.board_map), indent=2))


def _nnls(X, y):
    try:
        from scipy.optimize import nnls
        return nnls(X, y)[0]
    except ImportError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def _loo(X, y):
    e = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        e.append((X[i] @ _nnls(X[m], y[m]) - y[i]) / y[i])
    return float(np.max(np.abs(e)))


def cmd_calibrate(a):
    """measured.json: [{"bundle": ".../x.h", "us": measured, "inputs": ..., "layers": optional}, ...]"""
    base = Path(a.measured).resolve().parent
    items = json.loads(Path(a.measured).read_text())
    cpu, npu = [], []
    for it in items:
        bp = Path(it["bundle"])
        b = Bundle(bp if bp.is_absolute() else base / bp)
        ip = it.get("inputs")
        inputs = read_inputs(ip if not ip or Path(ip).is_absolute() else base / ip)
        work = Path(tempfile.mkdtemp(prefix="k5cal_"))
        res = run(b, inputs, work, trace=True)
        shutil.rmtree(work, ignore_errors=True)
        row = dict(bundle=b.name, us=it["us"], cls=res["by_class"], layers=it.get("layers"),
                   bytes=npu_bytes(res), calls=res.get("neutron_calls"))
        (npu if b.neutron else cpu).append(row)
        print(f"{b.name:<36s} {'NPU' if b.neutron else 'CPU'} insn {res['instructions']:>10,} "
              f"measured {it['us']:>9.1f} us")
    cal = dict(created_from=str(Path(a.measured).resolve()), note=a.note)
    # CPU: intercept + per-class cost, non-negative least squares
    X = np.array([[1.0] + [r["cls"][k] for k in CLASSES] for r in cpu])
    y = np.array([r["us"] for r in cpu])
    coef = _nnls(X, y)
    pred = X @ coef
    cal["cpu"] = dict(intercept_us=float(coef[0]),
                      per_class_us={k: float(v) for k, v in zip(CLASSES, coef[1:])},
                      cycles_per_class_at_200MHz={k: round(float(v) * 200, 2) for k, v in zip(CLASSES, coef[1:])},
                      fit_max_rel_err=float(np.max(np.abs(pred - y) / y)), loo_max_rel_err=_loo(X, y),
                      n=len(cpu), note=a.note,
                      points=[dict(bundle=r["bundle"], measured_us=r["us"], predicted_us=round(float(p), 1))
                              for r, p in zip(cpu, pred)])
    print(f"CPU  intercept {coef[0]:.1f} us; cycles/insn @200 MHz "
          f"{cal['cpu']['cycles_per_class_at_200MHz']}; fit max {100*cal['cpu']['fit_max_rel_err']:.1f} %, "
          f"leave-one-out max {100*cal['cpu']['loo_max_rel_err']:.1f} %")
    if npu:
        cpu_part = np.array([sum(cal["cpu"]["per_class_us"][k] * r["cls"][k] for k in CLASSES) for r in npu])
        yn = np.array([r["us"] for r in npu]) - cpu_part
        out = {}
        Xb = np.array([[1.0, r["bytes"]] for r in npu])
        cb = _nnls(Xb, yn)
        out["bytes_only"] = dict(intercept_us=float(cb[0]), us_per_byte=float(cb[1]),
                                 loo_max_rel_err=_loo(Xb, yn + cpu_part) if False else _loo(Xb, yn))
        if all(r["layers"] is not None for r in npu):
            Xl = np.array([[1.0, r["bytes"], r["layers"]] for r in npu])
            cl = _nnls(Xl, yn)
            out["with_layers"] = dict(intercept_us=float(cl[0]), us_per_byte=float(cl[1]),
                                      us_per_layer=float(cl[2]), loo_max_rel_err=_loo(Xl, yn))
        else:
            out["with_layers"] = out["bytes_only"] | dict(us_per_layer=0.0)
        out["note"] = a.note
        out["points"] = [dict(bundle=r["bundle"], measured_us=r["us"], bytes_to_npu=r["bytes"], layers=r["layers"])
                         for r in npu]
        cal["npu"] = out
        for k in ("bytes_only", "with_layers"):
            print(f"NPU  {k}: {json.dumps({kk: round(v, 6) for kk, v in out[k].items()})}")
    Path(a.cal).parent.mkdir(parents=True, exist_ok=True)
    Path(a.cal).write_text(json.dumps(cal, indent=1))
    print(f"calibration -> {a.cal}")


def cmd_npu(a):
    """NPU bundle: CPU-side code on the emulated M7 (Neutron dispatches stubbed
    and counted) + int8 numerics from the quantised TFLite model."""
    b = Bundle(a.bundle)
    if not b.neutron:
        raise SystemExit("not a Neutron bundle; use 'run'")
    inputs = read_inputs(a.inputs)
    work = Path(tempfile.mkdtemp(prefix="k5npu_"))
    res = run(b, inputs, work, trace=True)
    shutil.rmtree(work, ignore_errors=True)
    t = predict_us(load_cal(a.cal), res, True, a.layers, neutron_log_estimate_us(a.neutron_log))
    print(fmt_report(b, res, None, t))
    if a.tflite:
        out = tflite_int8(a.tflite, b, inputs)
        for k, v in out.items():
            print(f"int8 TFLite output '{k}' (dequantised): {np.round(v.ravel()[:10], 6).tolist()}{' ...' if v.size > 10 else ''}")
        if a.out:
            np.savez(a.out, **out)
            print(f"outputs -> {a.out}")


def tflite_int8(path, b: Bundle, inputs: dict):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    def norm(n):
        return re.sub(r"^serving_default_|(:\d+|_\d+)$", "", n)
    for d in it.get_input_details():
        key = next((k for k in inputs if norm(k) == norm(d["name"])), None)
        if key is None:
            raise SystemExit(f"no input given for TFLite tensor {d['name']}; given {list(inputs)}")
        v = np.asarray(inputs[key], np.float64).reshape(d["shape"])
        sc, zp = d["quantization"]
        if d["dtype"] in (np.int8, np.uint8) and sc:
            info = np.iinfo(d["dtype"])
            v = np.clip(_std_round(v / sc) + zp, info.min, info.max)
        it.set_tensor(d["index"], v.astype(d["dtype"]))
    it.invoke()
    out = {}
    for d in it.get_output_details():
        v = it.get_tensor(d["index"]).astype(np.float64)
        sc, zp = d["quantization"]
        out[d["name"]] = (v - zp) * sc if sc else v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("info", help="print what the bundle header declares")
    p.add_argument("bundle")
    p.set_defaults(f=cmd_info)

    for name, fn, hlp in (("run", cmd_run, "execute on the emulated M7: outputs, instructions, time, memory"),
                          ("npu", cmd_npu, "Neutron bundle: CPU part + dispatch count + int8 numerics")):
        p = sp.add_parser(name, help=hlp)
        p.add_argument("bundle", help="bundle .h (or a folder holding one .h/.o/.weights.txt)")
        p.add_argument("--inputs", help=".npz/.json {placeholder: values} (unquantised floats)")
        p.add_argument("--cal", default=str(CAL_DEFAULT), help="timing calibration JSON")
        p.add_argument("--out", help="save outputs to .npz")
        p.add_argument("--layers", type=int, help="number of FC layers (improves the NPU time model)")
        p.add_argument("--neutron-log", help="eiqa.log of the offline build (NPU compute estimate, for batch > 1)")
        if name == "run":
            p.add_argument("--json", help="save the full report as JSON")
            p.add_argument("--board-elf", help="board firmware ELF (memory baseline)")
            p.add_argument("--board-map", help="board firmware .map (memory baseline)")
            p.add_argument("--no-time", action="store_true", help="skip the instruction trace (faster)")
            p.add_argument("--work", help="keep build files here")
            p.add_argument("--link", action="append", default=[], help="extra C file to link (e.g. opt/fast_expf.c)")
        else:
            p.add_argument("--tflite", help="int8 TFLite model before Neutron conversion")
        p.set_defaults(f=fn)

    p = sp.add_parser("debug", help="build and start QEMU halted, with a GDB server")
    p.add_argument("bundle")
    p.add_argument("--inputs")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--work")
    p.set_defaults(f=cmd_debug)

    p = sp.add_parser("memory", help="memory use against the S32K566 M7_0 regions")
    p.add_argument("bundle")
    p.add_argument("--board-elf")
    p.add_argument("--board-map")
    p.set_defaults(f=cmd_memory)

    p = sp.add_parser("calibrate", help="fit instruction count -> board us from measured runs")
    p.add_argument("measured", help='JSON list [{"bundle": ".../x.h", "us": 116.4}, ...]')
    p.add_argument("--cal", default=str(CAL_DEFAULT))
    p.add_argument("--note", default="S32K566 M7_0 @ 200 MHz, eIQ Auto 3.14.1, Session::Execute wall clock")
    p.set_defaults(f=cmd_calibrate)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

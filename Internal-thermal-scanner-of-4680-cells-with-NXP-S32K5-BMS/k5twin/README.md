# k5twin: software twin of the S32K566 M7 + eIQ Auto target

k5twin tests a compiled eIQ Auto network before it is flashed to the board.
It does three things:

- **Runs the network's own machine code.** It takes the `.o` that `graph_build` writes for target `m7`, the same object that is linked into the board firmware, and runs it on an emulated Cortex-M7 (QEMU `mps2-an500`).
- **Supports debugging.** A GDB server lets you set breakpoints and inspect registers and buffers.
- **Reports memory and time.** Memory is checked against the real S32K566 M7_0 memory map. Time is a microsecond prediction calibrated on Arnold Sequeira's 28 measured board runs.

## Checks against the board

| Check | Result |
|---|---|
| 200-row CPU build (batch 200), outputs vs board, CPU_vs_NPU_Report Table 2 | identical to 4 decimals (`tests/selftest.py`) |
| CPU time, 14 builds (width and depth sweeps), leave-one-out | max error 5.6 % |
| NPU time, 14 builds, leave-one-out (with layer count) | max error 2.9 % (8.5 % without layer count) |
| NPU numerics, int8 TFLite vs board NPU results | 0–3 output steps (≤ 0.2 °C). The board values are from the batch-1 build, so this needs re-checking against a board run of the batch-200 build. |

## Predictions for the existing 200-cell builds (`05_..._200_cells`), not yet measured on the board

| Build | Instructions | Predicted time | Stack | SRAM (M7_0, 512 KiB) | MRAM (4 MiB) |
|---|---|---|---|---|---|
| CPU float32, batch 200 | 976,969 | 9.7 ms ± 6 % | 92 B | 57.7 % | 11.7 % |
| NPU int8, batch 200 | 3,357 CPU-side + 1 NPU dispatch | 180 µs ± 3 % + uncertainty in the NPU compute (see below) | 56 B | 52.9 % | 12.3 % |

For a 2 s update of 200 cells, the CPU build would use about 0.5 % of one M7 core and the NPU build about 0.01 %.

## Install (WSL, Ubuntu 22.04 or 24.04)

Ubuntu 20.04's QEMU (4.2) has no Cortex-M7 board, so use a newer distribution than the one used for the eIQ tools. Two distributions can live side by side:

```bash
sudo apt install qemu-system-arm gcc-arm-none-eabi libnewlib-arm-none-eabi gdb-multiarch python3-numpy python3-scipy
pip install ai-edge-litert        # only for 'npu --tflite'
```

Set `K5_GCC_PREFIX` to use NXP's GCC 11.4 instead, which is the compiler of the board firmware.

## GUI

```bash
python3 k5twin_gui.py --root "<folder with your eIQ Auto projects>"
# then open http://localhost:8765 in Windows
```

The GUI finds every bundle under the folder. For each one it fills in the board ELF/.map, `eiqa.log`, the int8 TFLite model and the layer count where they exist. One click runs the twin and shows:

- the predicted time, and where the time goes by instruction class;
- memory use per S32K566 region;
- the most expensive functions;
- the outputs compared with the reference values in the input file;
- a comparison of all runs in the session.

There is also a what-if panel for memory cost, number of cores and the fixed per-call cost. It gives estimates outside the calibration. The **Start debugger** button starts QEMU with a GDB server.

## Code optimisation test: `opt/fast_expf.c`

Glow's tanh calls `expf` once per element. `opt/fast_expf.c` replaces newlib's version with a polynomial version (max relative error 2.5e-7). Link it with `--link opt/fast_expf.c`, or tick the box in the GUI. On the 200-cell build it changes the outputs by less than 4e-6 °C and cuts executed instructions by 22 %. The calibrated time model predicts only 4 % less board time, because floating-point loads and stores take 93 % of the predicted time.

## Use

A bundle is the folder (or the `.h`) that holds `<name>.h`, `<name>.o` and `<name>.weights.txt`. These are the files that `graph_build` writes into `eiqa_runtime_app/data/<variant>/models/`.

Inputs are given as `.json` or `.npz` files in the form `{placeholder: values}`. Values are unquantised floats in the normalised units the network was trained on. The original ONNX names (`space`, `time`, `measurement`) are accepted for NPU bundles too.

```bash
# describe the bundle
python3 k5twin.py info  .../data/cpu_glow/models

# functional run + time + memory (board ELF/.map = memory baseline)
python3 k5twin.py run   .../data/cpu_glow/models --inputs examples/hev2_5x40_batch200.json \
        --board-elf .../build_cpu_glow/bin/<elf> --board-map .../build_cpu_glow/bin/<elf>.map \
        --out out.npz --json report.json

# NPU bundle: CPU-side part on the emulated M7, NPU dispatches counted,
# int8 numerics from the quantised TFLite (the one before Neutron conversion)
python3 k5twin.py npu   .../data/neutron/models --inputs examples/hev2_5x40_batch200.json \
        --layers 3 --neutron-log .../eiqa_offline/eiqa.log \
        --tflite .../glow_model_data_test/subgraph_0/transform/onnx2tf_saved_model_p0.tflite

# debugging: QEMU halted at reset, GDB server on :1234
python3 k5twin.py debug .../data/cpu_glow/models --inputs examples/hev2_sample1.json
gdb-multiarch k5twin_debug/twin.elf -ex 'target remote :1234' -ex 'break k5_mark_end' -ex continue
#   (gdb) x/200fw mutableWeight+<output offset>     # outputs, offsets from 'info'

# self-test against the board results
python3 tests/selftest.py "<NXP>/02_Project Folders/05_S32K5_Temperature_estimation_200_cells"
```

## How time is predicted

QEMU runs one instruction per translation block and logs each one. k5twin counts the instructions executed between two markers placed around the network call, by instruction class:

- floating-point arithmetic;
- floating-point loads and stores;
- integer loads and stores;
- branches;
- other integer instructions;
- division and square root.

The calibration (`calibration/m7_200MHz.json`) converts these counts to board microseconds with a non-negative least-squares fit on measured `Session::Execute()` times. Per class, at 200 MHz:

| Instruction class | Effective cycles |
|---|---|
| Floating-point load/store | 7.7 |
| Floating-point arithmetic | 0.37 |
| Other integer | 0.09 |
| Fixed cost per call (eIQ runtime) | 63 µs |

The large load/store cost is consistent with the weights being read from MRAM. These figures are effective values for this kind of code, not datasheet latencies.

For the NPU, the time is modelled as:

$t = 123\,\mu s + 0.74\,\text{ns per byte handed to the NPU} + 2.3\,\mu s \text{ per layer} + \text{CPU-side ops (CPU model)}$

For a batch larger than 1, the NPU's own compute is taken from neutron-converter's estimate in `eiqa.log`, because all calibration runs used batch 1.

To recalibrate after a new board run, add the build and its measured µs to `calibration/measured_sweeps.json` and run `python3 k5twin.py calibrate calibration/measured_sweeps.json`. The sweep paths in that file assume the sweep project (`04_S32K5_Temperature_estimation_scaling`, not public) is unzipped into `k5twin/private/`, which git ignores.

## What it does not model

- **NPU internals.** There is no public model of the Neutron microcode. The NPU call is replaced by a stub that records the operator type, operand count and bytes, so NPU outputs come from the int8 TFLite model and NPU time from the calibration.
- **The eIQ runtime layer** (`eiqa::Session`, `LoadModelRaw`, tensor handles). Its fixed cost is in the calibration intercept.
- **Platform initialisation, peripherals and the Cortex-R52 cores.**
- **Caches and memory wait states.** These are absorbed by the calibration, so a change in memory placement (e.g. weights moved to TCM) needs a new calibration point.
- **Floating-point exceptions and denormal timing** differ from the real core.
- **Library differences.** `expf` (used for tanh) comes from the toolchain's newlib. The board uses the same `__ieee754_expf` routine from NXP GCC 11.4, but in a different build.

For full-firmware emulation including peripherals, NXP's S32K5 VDK (built on Synopsys Virtualizer) is the matching tool. eIQ Auto already supports it (`-DTARGET_PLATFORM=vdk`, `init_platform_VDK()`), but it needs an NXP/Synopsys licence.

## Files

```
k5twin.py                 the tool (info, run, npu, debug, memory, calibrate)
k5twin_gui.py, gui/       local web GUI
opt/fast_expf.c           faster expf for Glow tanh (optional link)
harness/startup.c         Cortex-M7 reset, FPU on, fault report via semihosting
harness/harness.c         static buffers as in main_*.cpp, file I/O, markers, Neutron stub
harness/an500.ld          emulator memory layout (sizes are checked separately against the S32K566 map)
calibration/              fitted calibration + the measured runs it came from
examples/                 HEV2 inputs: one sample (batch 1) and 5 samples x 40 (batch 200)
tests/selftest.py         reproduces the board results
```

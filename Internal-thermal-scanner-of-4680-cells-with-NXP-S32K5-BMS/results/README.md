# Results: Arnold Sequeira's 200-cell builds on k5twin (2 November 2026)

## What was run

The builds are from `05_S32K5_Temperature_estimation_200_cells` (A. Sequeira, September 2026). The PINN core-temperature estimator (6-16-16-1, 401 parameters) is compiled for the S32K566 Cortex-M7 with a fixed batch of 200 rows. Two versions were run:

- float32 on the CPU (`onnx_glow`);
- int8 on the NPU (`tflite_neutron_glow`).

**Inputs:** the five HEV2 samples of the deployment report, each repeated 40 times (`k5twin/examples/hev2_5x40_batch200.json`).

**Twin setup:**
- QEMU 8.2.2 (`mps2-an500`, Cortex-M7);
- arm-none-eabi GCC 13.2 with newlib 4.4 (v7e-m+fp/hard);
- calibration `k5twin/calibration/m7_200MHz.json`.

## Results

| Build | Instructions executed | Predicted board time | Stack peak | SRAM M7_0 (512 KiB) | MRAM (4 MiB) |
|---|---|---|---|---|---|
| CPU float32 | 976,969 | 9.69 ms ± 6 % | 92 B | 57.7 % | 11.7 % |
| CPU float32 + `opt/fast_expf.c` | 759,889 | 9.30 ms | 92 B | 57.7 % | 11.7 % |
| NPU int8 | 3,357 on the CPU + 1 NPU dispatch (8,824 B) | 180 µs ± 3 % (+ NPU estimate uncertainty) | 56 B | 52.9 % | 12.3 % |

How to read these numbers:
- The ± values are the largest leave-one-out errors on the calibration set (14 CPU and 14 NPU builds, all batch 1).
- The NPU compute for the batch of 200 (44 µs) is the neutron-converter estimate, not a measurement.
- The SRAM figure includes the firmware's 256 KiB heap. The network's own buffers use 31 KiB on the CPU and 4.6 KiB on the NPU.

## Checks

- The CPU outputs on the emulated M7 are identical, to 4 decimals, to the board results in the deployment report (Table 2): 14.5108, 23.3124, 22.0816, 23.2740, 21.0552 °C.
- The int8 TFLite model, evaluated on the PC, gives 14.4098, 23.2718, 22.2215, 23.2061, 21.1055 °C. These are within 0–3 int8 output steps (≤ 0.2 °C) of the board NPU values of the batch-1 build. A board run of this batch-200 build is still needed.
- The fast `expf` changes the outputs by less than 4e-6 °C.

## Where the CPU time goes

Floating-point loads and stores take 93 % of the predicted CPU time, at about 7.7 effective cycles each. This is consistent with the weights being read from MRAM. `expf`, which Glow's tanh calls, takes 46 % of the instructions but little of the time. Estimates outside the calibration, still to be checked on the board:
- **Weights and buffers in DTCM** (33 KiB): 1.9–3.0 ms.
- **The 200 rows split over 4 M7 cores:** about 2.5 ms, or 0.5–0.8 ms with DTCM as well.

## Limits

- **Not a 4680 model yet.** The PINN is the 26650-size model of Zheng et al. (2024). The 4680 network is still to come.
- **The reference temperatures are simulated.** They are the simulated core temperatures from the HEV2 thermal model output, not measured ones.
- **No board timing for batch 200 yet.** Flashing these two builds once will check the predictions and give batch-200 calibration points.

## Files

| File | Content |
|---|---|
| `cpu_float32_batch200.{txt,json}` | twin report: outputs, instruction classes, functions, memory regions, prediction |
| `cpu_float32_batch200_fast_expf.{txt,json}` | the same, with `opt/fast_expf.c` linked |
| `npu_int8_batch200.{txt,json}` | NPU build: CPU-side part, NPU dispatch record, prediction |
| `gui_*.png` | the k5twin GUI on these runs |

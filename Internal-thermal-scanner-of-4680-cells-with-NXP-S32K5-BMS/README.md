# Internal thermal scanner of 4680 cells with NXP S32K5 BMS

AI-VOLT group, AAU Energy, Aalborg University, with NXP Semiconductors.

The aim of this work is to estimate the internal temperature field of every cell in a pack of about 200 BYD LFP 4680 cells, at roughly a 2 s update rate, on an NXP S32K5 battery-management microcontroller, using its Cortex-M7 cores and eIQ Neutron NPU.

The estimator is a physics-informed neural network. The plan is to distil it from the thermal DFN models of this repository:
- [Parameterisation of NMC and LFP 4680 cells](../Parameterisation-of-NMC-and-LFP-4680-cells/);
- [Multi-timescale thermal DFN of LFP 4680 cells](../Multi-timescale-thermal-DFN-of-LFP-4680-cells/).

The network is then compressed (distillation and int8 quantisation) and deployed with NXP eIQ Auto.

This folder holds the tools, results and, later, the papers of this work. It is work in progress.

## Status (November 2026)

| Item | State |
|---|---|
| eIQ Auto toolchain and S32K566 board bring-up | done (A. Sequeira, 2026) |
| PINN core-temperature estimator ([Zheng et al. 2024](https://doi.org/10.1109/TTE.2023.3294417)) on the M7 (float32) and on the NPU (int8) | done for a 26650-size cell; MAE 0.22–0.28 °C on 5 HEV2 samples |
| CPU vs NPU timing over network width and depth (28 builds) | done; NPU faster above about 457 parameters |
| 200 cells in one batched call (CPU and NPU builds) | built; board timing pending |
| **k5twin** software twin of the S32K566 M7 + eIQ Auto target, with GUI and Colab notebook | available in [`k5twin/`](k5twin/) |
| Physics-informed network for the BYD LFP 4680 cell, distilled from the TDFN models | planned |
| Live data input, display and fault cases (internal short) | planned |

## k5twin: test a network before flashing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RemusTeodorescu/4680-cell-models/blob/internal-thermal-scanner-s32k5/Internal-thermal-scanner-of-4680-cells-with-NXP-S32K5-BMS/notebooks/k5twin_colab.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RemusTeodorescu/4680-cell-models/tree/internal-thermal-scanner-s32k5?devcontainer_path=.devcontainer%2Fk5twin%2Fdevcontainer.json)

k5twin runs a compiled eIQ Auto network on an emulated Cortex-M7 and reports four things:
- the outputs;
- the executed instructions;
- memory use against the S32K566 memory map;
- the board time, calibrated on 28 measured board runs.

It has a GUI and a GDB debugger. Details: [`k5twin/README.md`](k5twin/README.md).

- **Colab:** use the badge above. You supply your own eIQ Auto project zips; see the notebook.
- **GitHub Codespaces:** use the Codespaces badge. It creates a cloud machine with the tools installed (first start takes a few minutes); then run `python3 k5twin/k5twin_gui.py --root /workspaces/s32k5-thermal-scanner-projects/projects` and open the forwarded port 8765. Members of the project with access to the private eIQ Auto projects repository get it cloned automatically (accept the access request when the codespace is created); others upload their own eIQ Auto project by dragging it into the file explorer.
- **Locally with an NVIDIA GPU (for training the thermal networks):** install Docker Desktop (WSL 2 backend) and VS Code with the Dev Containers extension, clone the repository, then use Reopen in Container → **k5twin-gpu**. It gives the twin plus PyTorch with CUDA 12.8 (Blackwell GPUs included). Check with `nvidia-smi` and `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
- **Locally (WSL Ubuntu 22.04/24.04 or Linux):** run `python3 k5twin/k5twin_gui.py --root <folder with eIQ Auto projects>`, then open `http://localhost:8765`.

## Results so far

The [`results/`](results/) folder has the reports and the GUI screenshots. All times below are twin predictions; none of these builds has been timed on the board yet.

| 200-cell build (one call, M7 at 200 MHz) | Predicted time | Load at 2 s update | SRAM (M7_0) | MRAM |
|---|---|---|---|---|
| CPU float32 | 9.69 ms ± 6 % | 0.48 % | 57.7 % | 11.7 % |
| CPU float32, fast `expf` | 9.30 ms | 0.47 % | 57.7 % | 11.7 % |
| NPU int8 | 180 µs ± 3 %, plus the uncertainty of the converter's NPU estimate | 0.01 % | 52.9 % | 12.3 % |

## Papers

Papers from this work will be listed in [`papers/`](papers/), each with its data and code, when submitted.

## Not included, and why

The following are not published here:
- **Compiled eIQ Auto networks and board firmware** (`.o`, `.eiqa`, ELF files). They are produced with NXP's licensed eIQ Auto toolchain and contain NXP runtime code.
- **The HEV1/HEV2 training data** of the PINN.

The twin works with any eIQ Auto project you have; the two example input files hold five normalised HEV2 samples.

## Credits

- Remus Teodorescu (AAU Energy), with
  - Arnold Sequeira (AAU Energy, until autumn 2026): board bring-up, eIQ Auto deployment, the CPU/NPU studies and the board measurements used for calibration;
  - Nicolai A. Weinreich (AAU Energy, from November 2026): implementation on the S32K5 platform.
- Supervision: Wenzel Prochazka (NXP Semiconductors Austria).

## Licence

Code (`k5twin/`, `notebooks/`) is under the MIT licence. Results and calibration data (`results/`, `k5twin/calibration/`) are under CC BY 4.0. See the repository [`LICENSE`](../LICENSE).

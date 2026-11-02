# 3-D thermal model of an LFP 4680 cell for real-time hotspot scanning (paper 1.2)

**Status:** in preparation (work in progress, November 2026). This branch holds the two digital twins of the paper. Results will change as the model develops.

[![Open the cell twin in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RemusTeodorescu/4680-cell-models/blob/paper-1.2-3d-thermal-model/3D-thermal-model-of-LFP-4680-cell-for-hotspot-scanning/notebooks/cell_twin_colab.ipynb)
[Open the module twin in the browser](https://htmlpreview.github.io/?https://github.com/RemusTeodorescu/4680-cell-models/blob/paper-1.2-3d-thermal-model/3D-thermal-model-of-LFP-4680-cell-for-hotspot-scanning/module_twin/module_twin.html)

**Cell:** BYD FC4680 LFP, the cell of the two companion papers in this repository. Geometry from the teardown by Liu et al. (2025), via `byd_parts.py` of the parameterisation paper.

![BYD FC4680, exploded](figures/fig_byd_exploded.png)

## The two twins

### 1. Cell twin (`cell_twin/`): 3-D thermal model with the electrode sheet

| File | Content |
|---|---|
| `geom3d.py` | Geometry and mesh: spiral map s ↔ (r, θ), tabs at their measured radii, collector and insulator disks, mandrel, can; anisotropic stack conductivity from the layer stack |
| `thermal3d.py` | Finite-volume model on (r, θ, z), backward Euler, AMG-preconditioned CG. Verification, micro-short study, uniform heat with and without a foam cap |
| `sheet.py` | Unrolled electrode sheet (s, z): Al and Cu foil potentials, current to the real tabs, a local model at every node, coupled to the 3-D grid. Internal shorts by position, type (anode–cathode, Al–anode, Cu–cathode, Al–Cu) and side |
| `figs.py` | Figures from `out/` |
| `fig_byd_exploded.py`, `fig_module_23s9p.py` | Drawings |

**Running it:**

```
cd cell_twin
pip install -r requirements.txt
python thermal3d.py verify     # analytic cylinder and energy balance, ~1 min
python thermal3d.py short      # micro-short study with mesh convergence, ~40 min on 2 cores
python thermal3d.py uniform    # foam-cap predictions, ~10 min
python sheet.py                # coupled sheet + 3-D runs, ~7 min
python figs.py
```

The reference results are in `cell_twin/out/`.

**Local model:** the local model in `sheet.py` is a placeholder: an LFP-like OCV with an area-specific resistance. The multi-timescale thermal DFN model of the companion paper (MT-TDFN) plugs in through the same three calls: `linearize`, `advance`, `heat`.

### 2. Module twin (`module_twin/module_twin.html`)

A 23s9p module of 207 cells, about a quarter of a 92s9p 4680 pack, with 2 NTCs at the coolant inlet and outlet. Each cell is drawn in the colour of its surface temperature.

- Click a cell to place a soft short.
- The twin shows the temperature field, the two NTC readings and the voltage deviation of every series group.
- It is a single HTML file and runs in any browser.

![23s9p module](figures/fig_module_23s9p.png)

## First results

All values below are from the model, not measurements.

- **Resolution.**
  - Radially, one cell per wrap (pitch 0.30 mm) is enough: going to 2 cells per wrap changes the short's peak by 1.3 %.
  - The layers inside a wrap equalise in about 10 ms, so one node per layer is not needed.
  - In the plane of the sheet, the peak temperature of a short converges only when the grid resolves the short's footprint. The signals at the can converge on a coarse grid.
- **Thermal signal of a soft short.**
  - A 1 W short at mid-radius raises the hottest point inside by 10–16 K after 60 s.
  - On the can directly above it, the rise is 0.13 K.
  - Detection with a few NTCs per module therefore has to be electrical.
- **Current collection.**
  - With the four cathode tabs and two live anode tabs of the teardown, the foil path alone gives 8–11 mΩ, and the current density along the cathode varies by a factor of about 20.
  - This conflicts with the measured 10 s pulse resistance (≈ 9–10 mΩ) and is being checked against the teardown.

![Micro-short response](figures/fig_short_response.png)
![Electrode sheet coupled to the 3-D grid](figures/fig_coupled_sheet.png)

## Next

1. MT-TDFN as the local model.
2. A locally refined patch at the short.
3. A response library per location and short type, linked to the module twin.
4. A GPU solver.
5. Training data for the Physics AI NN (physics-informed neural network), including transient shorts.

## Credits and licence

Remus Teodorescu, AI-VOLT group, AAU Energy, Aalborg University.

Licence: code MIT, figures and data CC BY 4.0 (see `LICENSE` at the repository root).

Contact: ret@et.aau.dk.

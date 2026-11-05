# Parameter sets and bench data for NMC and LFP 4680 cylindrical lithium-ion cells

R. Teodorescu, N. A. Weinreich, B. Bilgin, Y. Zhuang, W. Prochazka
AAU Energy, Aalborg University, and NXP Semiconductors Austria

Dataset prepared for a Frontiers FAIR² Data Article. The analysis that uses it is reported in
[Parameterisation of NMC and LFP 4680 cells for electrochemical and thermal modelling](../Parameterisation-of-NMC-and-LFP-4680-cells/).

## What the dataset contains

Measured parameters for large-format 4680 cells are scarce. This dataset gives two complete,
provenance-tagged parameter sets for an electrochemical–thermal (thermal Doyle–Fuller–Newman) model,
the measurements they were checked and calibrated against, and the record of the calibration.

| Cell | Chemistry | Capacity | Geometry from |
|---|---|---|---|
| Tesla 4680 (tabless) | NMC811 / graphite | 22.3 Ah (model, C/3) | teardown of Ank et al. (2023) |
| BYD FC4680 (multi-tab) | LiFePO₄ / graphite | 15.3 Ah (measured, 1C) | teardown of Liu et al. (2025) |

| Folder | Content | Origin |
|---|---|---|
| `data/parameters/` | Both parameter sets as tidy CSV and JSON (79 parameters per cell, each with value, unit, provenance tag and source); the NMC set also as BPX JSON; closure checks against independently measured mass, capacity and winding length; Battery Data Genome metadata tiers | this work |
| `data/bench/` | Bench test of BYD FC4680 cell c1: charge, 1C discharge to 2.0 V, 1 h rest, charge, 2C discharge, 1 h rest; current, voltage and three can temperatures; 161 626 records at 1–10 Hz | measured in this work |
| `data/published_nmc_ank2023_digitised/` | Can temperatures during 1C and 2C charges, C/20 pseudo-OCV, 10 s pulse resistances and impedance spectra of the Tesla 4680, digitised from the figures of Ank et al. (2023) | derived from published figures |
| `data/calibration/` | Transferred and calibrated values of the calibrated parameters; the optimisation logs (one row per evaluation, with the cost and its terms); the model's voltage and temperature for the transferred and calibrated sets | this work |

Each column of each file is described in `data_dictionary.csv`. The test and the extraction and calibration
methods are in `METHODS.md`. Machine-readable metadata (MLCommons Croissant 1.0) are in `croissant.json`, and
file checksums in `SHA256SUMS`.

Provenance tags: **M** measured or reported for the cell itself; **L** literature value for the same
chemistry; **D** derived from tagged quantities by a stated relation; **E** estimated, to be treated as a
fitting parameter.

## Reuse

- The NMC set loads into PyBaMM: `pybamm.ParameterValues.create_from_bpx("data/parameters/bpx_nmc4680.json")`.
- The LFP set is given as CSV and JSON. Its BPX file is not part of this dataset because its electrochemistry is
  transferred from About:Energy's BPX parameterisation under CC BY-SA 4.0; it is available, under that licence,
  in [`../Parameterisation-of-NMC-and-LFP-4680-cells/data/parameters/`](../Parameterisation-of-NMC-and-LFP-4680-cells/data/parameters/).
- The bench data serve to validate any thermal or electrochemical model of the BYD FC4680 under constant-current
  discharge, including the temperature rise, the peak after cut-off and the cooling.
- The model code that produced the calibration runs is in
  [`../Parameterisation-of-NMC-and-LFP-4680-cells/models/`](../Parameterisation-of-NMC-and-LFP-4680-cells/models/) (MIT).

## Licence

Open Data Commons Attribution License (ODC-BY 1.0), see `LICENSE.md`. The digitised data of Ank et al. are
derived from their CC BY 4.0 article; cite their work when using them.

## Citation

Cite the data article (DOI on publication) and the research article above.

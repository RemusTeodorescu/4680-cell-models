# Parameterisation of NMC and LFP 4680 cells for electrochemical and thermal modelling

R. Teodorescu, N. A. Weinreich, B. Bilgin, Y. Zhuang, W. Prochazka

Preprint: arXiv (link added on announcement). Submitted to *Frontiers in Batteries and Electrochemistry* (2026).

## Abstract

An electrochemical–thermal model of the 4680 cylindrical cell is presented: a Doyle–Fuller–Newman model of the electrode stack coupled to a two-node thermal balance of the roll and the can (a thermal DFN, TDFN), solved by an implicit finite-volume scheme. Two cells are parameterised from teardown geometry and open literature, with every value tagged by provenance: the Tesla NMC811/graphite tabless cell and the BYD FC4680 LiFePO₄ multi-tab cell. The reaction heat is evaluated at the open-circuit potential of the particle-mean concentration, which restores the heat of mixing. Both sets are calibrated in the same way: the fixture is identified from the measurement alone, and a few effective parameters are adjusted by a Nelder–Mead minimisation of a normalised least-squares cost. For the NMC cell, against the open 1C and 2C charge test of Ank et al., calibrating the cathode diffusion time, its activation energy, the anode window and a contact heat of 0.94 mΩ outside the voltage path brings the model within 0.52 and 1.46 K of the measured mean can temperature, the contact heat matching the authors’ estimate for their holder. For the LFP cell, discharges at 1C and 2C to 2.0 V were measured on the authors’ bench; the fixture conductance, 0.337 W/K, follows from the energy balance over a full cycle and the heat capacity, 277 J/K, from the cooling. The transferred open-circuit potential of the cathode lacks the end-of-lithiation drop of LiFePO₄; with the drop restored and its width identified from the test, the calibrated set follows the can temperature within 0.44 K (1C) and 0.76 K (2C) over the whole discharge and reproduces the peaks, 14.1 against 15.0 K and 26.6 against 26.6 K, and the relaxation after cut-off. What remains for both cells is what a single can node and a single representative particle do not describe: the axial gradient of the NMC can under contact heating, and a gradual decline of the LFP voltage before the knee at 2C, of up to 90 mV, which points to the distribution of particle states in the electrode.

| Cell | Chemistry | Source of geometry |
|---|---|---|
| Tesla 4680 (tabless) | NMC811 / graphite | published teardown (Ank et al. 2023) |
| BYD FC4680 (multi-tab) | LiFePO4 / graphite | published teardown (Liu et al. 2025) |

Every parameter carries a provenance tag: `M` measured, `L` literature, `D` derived, `E` estimated.

## Contents

| Folder | What it holds |
|---|---|
| `data/parameters/` | Both parameter sets in all formats: Battery Parameter eXchange JSON (`bpx_*.json`, loads directly into PyBaMM), tidy CSV and JSON with provenance tags and sources (`parameters_both_cells.*`, `parameters_nmc.csv`, `parameters_lfp.csv`), the closure checks (`closure_audits.csv`, `quality_audits.csv`) and the same content in three metadata tiers (`metadata_*.csv`, see `README_metadata.txt`) |
| `data/bench/` | Bench test of the BYD FC4680 cell c1: 1C and 2C discharges to 2.0 V with rests; voltage, current and three can temperatures |
| `data/ank2023_digitised/` | Measurements of Ank et al. (2023) on the Tesla 4680, digitised from their published figures, used for the NMC comparison |
| `data/calibration/` | Logs and final runs of the calibration of both parameter sets |
| `models/` | The thermal DFN (TDFN) model: finite-volume solver (`gw2d.py`, `ddfnt.py`) and the provenance-tagged parameter sets (`gw4680.py` NMC, `byd4680_params.py` LFP, with geometry and closure modules) |
| `examples/` | `run_discharge.py`: a constant-current discharge of either cell |

## Quick start

Run the model from this folder (Python 3.10+, `pip install -r requirements.txt`):

```bash
python examples/run_discharge.py lfp 1    # BYD FC4680, 1C, bench fixture
python examples/run_discharge.py nmc 2    # Tesla 4680, 2C
```

Each run takes one to two minutes and writes voltage, current and roll and can temperature to a CSV file.
The parameter sets carry the calibrated values of the paper; the paper's runs use the measured test
currents and a finer grid.

Or load a parameter set into PyBaMM:

```python
import pybamm
params = pybamm.ParameterValues.create_from_bpx("data/parameters/bpx_lfp_fc4680.json")
```

## Licence

Code (`models/`, `examples/`): MIT. Data and parameter sets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), except
`data/parameters/bpx_lfp_fc4680.json`, which is CC BY-SA 4.0 because its electrochemistry is transferred
from About:Energy's BPX parameterisation (see `data/parameters/LICENCE-bpx-lfp.md`).
The digitised data of Ank et al. are derived from their CC BY 4.0 article; please also cite their work.


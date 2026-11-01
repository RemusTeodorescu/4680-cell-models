# 4680 cell models: thermal DFN model, open parameter sets and data

A thermal Doyle-Fuller-Newman (TDFN) model, parameter sets and measurement data for two 4680 cylindrical lithium-ion cells, released with the paper

> R. Teodorescu, N. A. Weinreich, B. Bilgin, Y. Zhuang, W. Prochazka,
> *Parameterisation of NMC and LFP 4680 cells for electrochemical and thermal modelling*,
> preprint arXiv (link added on announcement), submitted to Frontiers in Batteries and Electrochemistry (2026).

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

Run the model (Python 3.10+, `pip install -r requirements.txt`):

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

## Citation

Please cite the paper above and this repository (`CITATION.cff`).

Contact: Remus Teodorescu, AAU Energy, Aalborg University (ret@et.aau.dk).

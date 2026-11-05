# Methods

## 1. Cells

| | Tesla 4680 | BYD FC4680 |
|---|---|---|
| Chemistry | NMC811 / graphite | LiFePO₄ / graphite |
| Current collection | tabless | multi-tab (4 Al, 3 Ni tabs) |
| Can | 46 × 80 mm | 46.3 × 80.2 mm |
| Mass | 355.9 g (Ank et al.) | 332 g (published 330–335 g) |
| Data | published teardown and tests of Ank et al. (2023) | teardown of Liu et al. (2025); bench test of this work |

## 2. Bench test of the BYD FC4680 (`data/bench/`)

- **Fixture.** One cell (c1) in a horizontal toggle clamp with four spring contacts on the positive terminal
  and one pointed contact on the negative terminal, with separate voltage-sense contacts, inside a temperature
  chamber held at 25 °C by fans. The cell settles at 23.2 °C at rest (effective ambient).
- **Temperature.** Three type-T thermocouples glued with epoxy to the can: T1 1 mm from the positive end,
  T2 at mid-height, T3 1 mm from the negative end.
- **Protocol** (after two preconditioning cycles at 1C):
  1. CCCV charge at C/3 to 3.65 V, constant voltage to C/20;
  2. rest 2 h;
  3. CC discharge at 1C, 15.29 A, to 2.0 V;
  4. rest 1 h;
  5. CCCV charge as step 1;
  6. rest 2 h;
  7. CC discharge at 2C, 30.59 A, to 2.0 V;
  8. rest 1 h.
- **Sampling.** 10 Hz in steps 3, 4, 7 and 8; 1 Hz otherwise. `test_time_s` is reconstructed from the step
  durations.
- **Result.** 15.30 Ah at 1C and 14.44 Ah at 2C to 2.0 V; peak can-temperature rise (mean of the three sensors)
  15.0 K and 26.6 K, reached 11–12 s after cut-off. The delivered charge to 2.8 V agrees within 0.2 Ah with the
  rate data of Liu et al. (2025).

## 3. Published measurements of the Tesla 4680 (`data/published_nmc_ank2023_digitised/`)

Vector digitisation of Figures 12b, 13a, 14 and 16 of Ank et al. (2023), exact to plot precision. The
thermal data are the can temperatures at cap, centre and bottom during 1C and 2C constant-current charges
from the discharged state to 4.2 V followed by 1800 s of rest, resampled to 10 s; the constant-current
durations (3039 s and 1303 s) are read from the published current profile. Raw data of the original
study: mediaTUM, doi:10.14459/2023mp1725661 (not redistributed here).

## 4. Parameter extraction (`data/parameters/`)

- **Model.** Doyle–Fuller–Newman model of one electrode stack (Cu foil, anode, separator, cathode, Al foil)
  with a quartic particle-concentration profile, Butler–Volmer kinetics, reaction heat at the open-circuit
  potential of the particle-mean concentration (which includes the heat of mixing), ohmic and reversible heat,
  and a two-node thermal balance of jelly roll and can. Finite volumes, backward Euler, Newton.
- **Sources.** Geometry and inventory from the teardowns (Ank et al. 2023; Liu et al. 2025); NMC
  electrochemistry from Chen et al. (2020) with the electrolyte of Nyman et al. (2008); LFP electrochemistry
  from About:Energy's BPX parameterisation of an LFP/graphite 18650 cell; entropic coefficients from O'Regan et al. (2022).
- **Provenance.** Every value is tagged M, L, D or E (see README) and carries its source or derivation in the
  `*_source` / `note` columns.
- **Closure checks.** Quantities measured independently of the parameter set are computed from it and compared:
  cell mass (sum of all layers), capacity (coating loading and window) and winding length (spiral of known
  pitch inside the measured can bore). For the LFP cell, quantities that no source reports (cathode porosity,
  can wall, separator thickness) are determined from the same three relations, so these checks close by
  construction (`closure_audits.csv`, `quality_audits.csv`).
- **Windows.** Electrode stoichiometry windows calibrated on the C/20 capacity (NMC) and the measured capacity (LFP).

## 5. Calibration (`data/calibration/`)

Both sets were calibrated in the same five steps:
1. transfer the set and state its prediction;
2. cross-check the measurement against other published data;
3. identify the fixture from the measurement alone;
4. calibrate the E-tagged parameters;
5. examine the residual.

- **Fixture.** NMC: hA = 0.18 W/K from the measured cooling (time constant 1735 s) with the measured mass and
  c_p = 884.6 J/kg/K. LFP: hA = 0.337 W/K from the energy balance over the full cycle and C = 277 J/K from the
  cooling after cut-off.
- **Cost.** Nelder–Mead minimisation of

  J = Σ over 1C, 2C of [ (ΔT_rms / 0.5 K)² + (ΔT_peak / 0.5 K)² + (ΔQ / 0.1 Ah)² + (ΔV_rms / 20 mV)² ]

  with ΔT the can-temperature difference, ΔQ the difference in charge at the cut-off voltage and ΔV the voltage
  difference (LFP only; no voltage trace is published for the NMC test). Each evaluation runs the model at both rates.
- **NMC.** Four parameters: cathode diffusion time, its activation energy, anode stoichiometry at full charge,
  and a contact resistance whose I²R heat enters the can node but not the voltage. Log: `nmc_opt1_log.csv`.
- **LFP.** First the same four (logs `lfp_opt_log.csv`, `lfp_opt3_log.csv`), then with the end-of-lithiation
  drop of the cathode potential added as −A[(θ − 1 + w)/w]ⁿ for θ > 1 − w, A = 1.4 V fixed, w and n free
  (`lfp_opt7_log.csv`, `lfp_opt8_log.csv`, `lfp_opt9_log.csv`; opt9 is the final search, 45 evaluations, J = 21.4).
- **Result.** `calibrated_parameters.csv` lists the transferred and calibrated values; `model_runs.csv` the
  model's can and roll temperatures and voltage for both sets at both rates. The calibrated LFP set follows the
  can temperature within 0.44 K (1C) and 0.76 K (2C) RMS; the calibrated NMC set within 0.52 and 1.46 K.

## References

- M. Ank et al., J. Electrochem. Soc. 170, 120536 (2023).
- A. Liu et al., J. Electrochem. Soc. 172, 110536 (2025).
- C.-H. Chen et al., J. Electrochem. Soc. 167, 080534 (2020).
- J. A. Nelder and R. Mead, Comput. J. 7, 308 (1965).
- A. Nyman et al., Electrochim. Acta 53, 6356 (2008).
- K. O'Regan et al., Electrochim. Acta 425, 140700 (2022).
- About:Energy, BPX parameterisation of a 2 Ah LFP/graphite 18650 cell (2023), github.com/About-Energy-OpenSource/About-Energy-BPX-Parameterisation (CC BY-SA 4.0).
- L. Ward et al., Joule 6, 2253 (2022).

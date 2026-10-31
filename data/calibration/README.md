# Calibration logs and final runs

Both parameter sets were calibrated by a Nelder-Mead minimisation of a normalised least-squares cost
(paper 1, comparison with measurement).

`lfp/` (BYD FC4680, against the bench test in `data/bench/`):
- `opt_log.csv`, `opt3_log.csv`, `opt7_log.csv`, `opt8_log.csv`, `opt9_log.csv`: optimisation logs, one row per
  evaluation (parameters, cost and its terms); `opt9` is the final search and `opt9_result.json` its result;
- `c2_b1r.json`, `c2_b2r.json`: transferred set at the measured ambient, hA and heat capacity, 1C and 2C;
- `c2_o9_44_1C.json`, `c2_o9_44_2C.json`: calibrated set, 1C and 2C (time series of voltage and temperatures).

`nmc/` (Tesla 4680, against `data/ank2023_digitised/`):
- `opt1_log.csv`: optimisation log over the cathode diffusion time, contact resistance, activation energy and
  anode stoichiometry window;
- `cn_final_1C.json`, `cn_final_2C.json`: calibrated set, 1C and 2C CC charges.
  Calibrated values: tau_c 8031 s, R_c 0.94 mOhm, E_a(D_s) 11.0 kJ/mol, anode theta0 0.700.

Licence CC BY 4.0.

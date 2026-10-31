# Bench test of the BYD FC4680 LFP/graphite cell c1

`fc4680_c1_1C_2C.csv`: the full record of the thermal test at AAU Energy, Aalborg University.

Columns: `step_index`, `step_type`, `test_time_s`, `time_in_step_s`, `current_A` (positive = charge),
`voltage_V`, `step_capacity_Ah`, `step_energy_Wh`, `T1_C`, `T2_C`, `T3_C`. The file header states the
protocol, the fixture and the sampling.

Steps: (1) CCCV charge C/3 to 3.65 V, CV to C/20; (2) rest 2 h; (3) CC discharge 1C, 15.29 A, to 2.0 V;
(4) rest 1 h; (5) CCCV charge; (6) rest 2 h; (7) CC discharge 2C, 30.59 A, to 2.0 V; (8) rest 1 h.
10 Hz on steps 3, 4, 7, 8; 1 Hz otherwise. Chamber held at 25 C by fans; the cell settles at 23.2 C.
Three type-T thermocouples glued with epoxy to the can: T1 1 mm from the positive end, T2 at
mid-height, T3 1 mm from the negative end.

Licence CC BY 4.0.

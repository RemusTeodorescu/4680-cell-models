# One file in this dataset is not CC BY 4.0

`data/parameters/bpx_lfp_fc4680.json` is released under **CC BY-SA 4.0**. Everything else
in this repository is CC BY 4.0.

## Why

That file is a Battery Parameter eXchange description of the BYD FC4680. Its
geometry, inventory and the three quantities determined by the closure system
are ours. Its **electrochemistry** — the open-circuit potentials, the
stoichiometry window, the maximum concentrations, the diffusivities, the
reaction rate constants and the entropic coefficients — is transferred from
About:Energy's BPX parameterisation of a 2 Ah LFP/graphite 18650, which is
CC BY-SA 4.0:

> About:Energy Limited, *BPX parameterisation of a 2 Ah LFP|graphite
> cylindrical 18650 cell*, https://github.com/About-Energy-OpenSource/
> About-Energy-BPX-Parameterisation, CC BY-SA 4.0.

Our position throughout this project has been that individual parameter values
are facts while a parameter *file* is a work, which is why the parameter tables
carry these values with `L` provenance tags and do not redistribute
About:Energy's file. Assembling those values back into a BPX file is a
different act. What comes out is close enough in kind to their own artefact
that share-alike is respected rather than argued around.

This is a judgement, not a legal opinion, and it is deliberately the
conservative one. It costs nothing — nobody is harmed by a share-alike
interchange file — and the alternative costs an argument with the source of
half the parameter set.

## If you want the unencumbered part

`data/parameters/parameters_both_cells.csv` contains the FC4680's geometry, inventory and
determined quantities with their provenance, and **no transferred
electrochemistry**. It is CC BY 4.0 like the rest of the dataset.

## The NMC file

`data/parameters/bpx_nmc4680.json` is CC BY 4.0. Its geometry is ours and its
electrochemistry comes from Chen et al. (2020), a published paper whose
parameter values are facts and which PyBaMM itself ships.

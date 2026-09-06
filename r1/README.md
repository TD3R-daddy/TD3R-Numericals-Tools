# TD3R reproducibility package — R1 benchmark

This directory accompanies the TD3R synthesis article. It separates directly rerunnable finite checks from archived outputs of the larger working calculation.

## Directly rerunnable scripts

- `TD3R_IVE17_R1_Higgs_audit.py`: finite charge and Higgs-content consistency checks.
- `TD3R_IVE18_R1_vacuum_mass78.py`: reconstructs the **archived 78-state gauge-boson block spectrum** and verifies rank 66, nullity 12 and positivity. It is a 4D EFT benchmark; it does not derive the Higgs vacuum from 6D.
- `TD3R_IVE19_global_stabilizer.py`: compact independent audit of the Gamma_20 kernel, Smith invariants `(1,20)`, E6 root-orbit counting and regularity of the adjoint VEV at `v=w=1`.
- `TD3R_SUSY_completion_v0_9.py`: compact analytic check of two local N=1 compensator branches. It is not a 6D microscopic completion.
- `TD3R_compensator_branch_v0_8.py`: conditional EFT compensator-branch existence/robustness check.

## Archived machine-readable outputs

The JSON/CSV files record outputs from the larger R1 working archive. They are retained for traceability and comparison with the compact reruns.

## What this repository does **not** establish

It does not close the full 6D origin of the R1 Higgs sector, the complete covariant scalar Hessian, loop stability, the global multi-root vacuum, or the BION/SPARC observational pipeline.

Official SPARC data: https://astroweb.cwru.edu/SPARC/

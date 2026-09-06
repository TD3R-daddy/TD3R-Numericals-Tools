# TD3R reproducibility package — R1 and local completion checks

This directory accompanies the TD3R synthesis article. It contains numerical/algebraic checks available in the working archive at the time of this release.

## Included scripts

- `TD3R_IVE17_R1_Higgs_audit.py`: finite charge and Higgs-content checks.
- `TD3R_IVE18_R1_vacuum_mass78.py`: explicit R1 benchmark, F/D-flatness checks and 78 x 78 gauge-boson mass matrix.
- `TD3R_IVE19_global_stabilizer.py`: global stabilizer, Smith normal form, root-orbit audit, gauge-matrix kernel and restricted singlet Hessian.
- `TD3R_IVE19_global_stabilizer_report.json`: archived report associated with the global stabilizer calculation.
- `TD3R_compensator_branch_v0_8.py`: conditional EFT compensator-branch existence and robustness scan.
- `TD3R_SUSY_completion_v0_9.py` and `.json`: local N=1 completion checks.

The scripts were rerun successfully during preparation of the companion archive. These calculations reproduce finite benchmark checks. They do **not** close the full 6D origin of the R1 Higgs sector, the complete scalar Hessian, loop stability, the global multi-root vacuum, or the BION/SPARC observational pipeline.

Official SPARC data: https://astroweb.cwru.edu/SPARC/

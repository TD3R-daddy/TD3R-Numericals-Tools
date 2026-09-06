# TD3R-Numerical-Tools

Companion reproducibility repository for the journal-style synthesis of TD3R by Yves-Pol Taburet.

**Author:** Yves-Pol Taburet  
**ORCID:** https://orcid.org/0009-0004-1698-7900

## Scope

This repository contains numerical and algebraic checks that can be rerun independently from the manuscript. It deliberately distinguishes reproducible benchmark calculations from exploratory or not-yet-closed sectors.

## 1. R1 benchmark

Directory: `r1/`

Included checks:
- Higgs-content and charge audit;
- explicit `78 x 78` gauge-boson mass matrix benchmark;
- global stabilizer and root-orbit audit;
- restricted singlet Hessian;
- local supersymmetric completion checks;
- conditional compensator-branch checks.

The manuscript uses the R1 outputs only as a verified **4D benchmark**. They do not by themselves derive the R1 vacuum from the full six-dimensional compactification.

## 2. Fibonacci discrete-scale-invariance check

Directory: `fibonacci_dsi/`

`fibonacci_dsi.py` reproduces

`Omega_T = 2*pi/ln(phi_F) = 13.057005...`

and the corresponding kinematic ratio `m/H` if one separately imposes `Omega_Theta = Omega_T`.

The numerical identity is exact under the Fibonacci module. The microscopic selection of that module from the 6D action is not claimed to be derived.

## 3. BION / SPARC status

Directory: `bion/`

The current synthesis does **not** present the old BION galaxy pilots as a closed observational validation. The complete raw pilot archive is not contained in this release. `REPRODUCIBILITY_STATUS.md` gives the required protocol for a publication-grade rerun.

Official SPARC portal: https://astroweb.cwru.edu/SPARC/

Primary SPARC reference: F. Lelli, S. S. McGaugh and J. M. Schombert, *Astronomical Journal* **152**, 157 (2016), DOI `10.3847/0004-6256/152/6/157`.

## Installation

For the R1 Python checks, a recent Python 3 installation is sufficient. Install dependencies with:

```bash
python -m pip install -r r1/requirements.txt
```

## Reproduction

```bash
python r1/TD3R_IVE18_R1_vacuum_mass78.py
python r1/TD3R_IVE19_global_stabilizer.py
python fibonacci_dsi/fibonacci_dsi.py
```

## Citation

Please cite the associated TD3R Zenodo record or the software DOI once minted. No DOI is hard-coded before it exists.

## License

The source code in this repository is released under the MIT License. See `LICENSE`.

## Current release status

**Public reproducibility repository.** Release `2026.1` is being prepared for Zenodo archiving and DOI assignment.

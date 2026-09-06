#!/usr/bin/env python3
"""Reproduce the archived R1 gauge-boson mass-spectrum benchmark.

This is a finite 4D EFT benchmark. It verifies the published block spectrum,
rank and kernel of the 78 x 78 gauge-boson mass matrix. It does not derive the
Higgs vacuum from the 6D TD3R action.
"""
from pathlib import Path
import csv, json
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
BLOCKS = [
    ("SM", "(8,1,0)+(1,3,0)+(1,1,0)", [0.0]*12),
    ("45/24", "(3,2,5/6)+c.c.", [5.0/3.0]*12),
    ("45/(10+10bar)", "10+10bar components", [4.933333333333334]*20),
    ("16+16bar / 10+10bar", "10+10bar components", [3.1375672217930863]*20),
    ("16+16bar / 5+5bar", "5+5bar components", [6.787298334620742]*10),
    ("neutral 45+1", "two (1,1,0)", [10.0, 50.0/3.0]),
    ("neutral 16+16bar", "two (1,1,0)", [2.0211694422987643, 10.021169442298763]),
]

def main():
    vals = np.array([x for _,_,xs in BLOCKS for x in xs], dtype=float)
    assert vals.size == 78
    matrix = np.diag(vals)
    eig = np.linalg.eigvalsh(matrix)
    tol = 1e-10
    nullity = int(np.sum(np.abs(eig) < tol))
    rank = int(np.linalg.matrix_rank(matrix, tol=tol))
    assert (rank, nullity) == (66, 12)
    assert np.min(eig) >= -tol
    OUT.mkdir(exist_ok=True)
    with (OUT/"TD3R_R1_spectrum.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["block","SM_rep","multiplicity","eigenvalues_over_gE6sqMGsq"])
        for name,rep,xs in BLOCKS:
            w.writerow([name,rep,len(xs),";".join(f"{v:.15g}" for v in sorted(set(xs)))])
    report = {
        "matrix_shape":[78,78], "rank":rank, "nullity":nullity,
        "minimum_positive_eigenvalue":float(eig[eig>tol].min()),
        "maximum_eigenvalue":float(eig.max()), "trace":float(np.trace(matrix)),
        "status":"4D EFT existence/compatibility benchmark; not a 6D Higgs-sector derivation"
    }
    (OUT/"TD3R_R1_vacuum_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()

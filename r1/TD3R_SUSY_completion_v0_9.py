#!/usr/bin/env python3
"""Audit de deux completions N=1 du secteur compensateur TD3R.

Branche J (jordanienne):
    W = kappa X (S I3(Phi)/Lambda^2 - mu^2)
Dans la tranche I3(Phi)=P^3, elle admet un vide F=D=W=0 et un
Hessien positif modulo le Goldstone U(1). Ce vide exige I3 != 0 et ne
preserve donc pas le plongement SM de R1.

Branche S (singulets localises):
    W = kappa X (S_- B_+ - v^2)
Elle admet aussi un vide F=D=W=0, ne brise pas E6, et peut coexister avec
R1. Elle ne derive toutefois pas l'orientation jordanienne du 27.

Le script reproduit les Hessiens canoniques par differences finies et les
compare aux spectres analytiques. Aucune donnee observationnelle n'intervient.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq


TOL = 5.0e-6


def finite_hessian(function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1.0e-4) -> np.ndarray:
    size = point.size
    result = np.zeros((size, size), dtype=float)
    center = function(point)
    for i in range(size):
        ei = np.zeros(size)
        ei[i] = step
        result[i, i] = (function(point + ei) - 2.0 * center + function(point - ei)) / step**2
        for j in range(i):
            ej = np.zeros(size)
            ej[j] = step
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * step**2)
            result[i, j] = value
            result[j, i] = value
    return result


def positive_root_for_jordan(amplitude: float, xi: float) -> tuple[float, float]:
    """Solve rho^8 + xi rho^6 - 3 amplitude^2 = 0, s=amplitude/rho^3."""

    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")

    def polynomial(rho: float) -> float:
        return rho**8 + xi * rho**6 - 3.0 * amplitude**2

    upper = max(2.0, (3.0 * amplitude**2 + abs(xi) + 1.0) ** 0.25)
    while polynomial(upper) <= 0.0:
        upper *= 2.0
    rho = brentq(polynomial, 1.0e-12, upper)
    s = amplitude / rho**3
    return float(rho), float(s)


def jordan_potential_real(
    real_fields: np.ndarray,
    *,
    kappa: float,
    gauge_g: float,
    amplitude: float,
    xi: float,
) -> float:
    # Canonical convention z=(x+i y)/sqrt(2), I3(Phi)=P^3, Lambda=1.
    p = (real_fields[0] + 1j * real_fields[1]) / math.sqrt(2.0)
    s = (real_fields[2] + 1j * real_fields[3]) / math.sqrt(2.0)
    x = (real_fields[4] + 1j * real_fields[5]) / math.sqrt(2.0)
    f_x = kappa * (s * p**3 - amplitude)
    f_p = 3.0 * kappa * x * s * p**2
    f_s = kappa * x * p**3
    d_f = abs(p) ** 2 - 3.0 * abs(s) ** 2 + xi
    return float(abs(f_x) ** 2 + abs(f_p) ** 2 + abs(f_s) ** 2 + 0.5 * gauge_g**2 * d_f**2)


def jordan_branch(kappa: float = 1.0, gauge_g: float = 0.5, amplitude: float = 1.0 / math.sqrt(3.0), xi: float = 0.0) -> dict[str, object]:
    rho, s = positive_root_for_jordan(amplitude, xi)
    point = np.array([math.sqrt(2.0) * rho, 0.0, math.sqrt(2.0) * s, 0.0, 0.0, 0.0])
    potential = lambda values: jordan_potential_real(
        values, kappa=kappa, gauge_g=gauge_g, amplitude=amplitude, xi=xi
    )
    numerical_hessian = finite_hessian(potential, point)
    numerical_eigenvalues = np.linalg.eigvalsh(numerical_hessian)

    common_norm = rho**2 + 9.0 * s**2
    mass_f_squared = kappa**2 * rho**4 * common_norm
    mass_vector_squared = 2.0 * gauge_g**2 * common_norm
    analytic_eigenvalues = np.array(
        [0.0, mass_vector_squared, mass_f_squared, mass_f_squared, mass_f_squared, mass_f_squared]
    )
    analytic_eigenvalues.sort()

    return {
        "superpotential": "kappa X (S I3(Phi)/Lambda^2 - mu^2)",
        "field_charges": {"Phi": 1, "S": -3, "X": 0},
        "vacuum": {"rho": rho, "s": s, "X": 0.0, "I3_times_S": amplitude},
        "F_terms_zero": True,
        "D_term_zero": abs(rho**2 - 3.0 * s**2 + xi) < 1.0e-10,
        "W_zero": True,
        "unique_positive_radial_solution": True,
        "analytic_mass_squared": {
            "massive_U1_vector_multiplet": mass_vector_squared,
            "two_chiral_multiplets": mass_f_squared,
            "goldstone_before_gauge_fixing": 0.0,
        },
        "chiral_fermion_mass_eigenvalues": [
            -math.sqrt(mass_f_squared),
            0.0,
            math.sqrt(mass_f_squared),
        ],
        "gaugino_higgsino_dirac_mass": math.sqrt(mass_vector_squared),
        "supertrace_M2_exact_SUSY": 0.0,
        "analytic_real_hessian_eigenvalues": analytic_eigenvalues.tolist(),
        "numerical_real_hessian_eigenvalues": numerical_eigenvalues.tolist(),
        "hessian_max_abs_difference": float(
            np.max(np.abs(numerical_eigenvalues - analytic_eigenvalues))
        ),
        "positive_after_gauge_quotient": bool(numerical_eigenvalues[1] > 0.0),
        "R1_compatibility": {
            "I3_required_nonzero": True,
            "I3_on_standard_model_singlet_subspace": 0,
            "preserves_R1_standard_model": False,
            "reason": "The 27 cubic vanishes on the two SM-singlet VEV directions.",
        },
    }


def singlet_potential_real(real_fields: np.ndarray, *, kappa: float, gauge_g: float, v: float) -> float:
    b = (real_fields[0] + 1j * real_fields[1]) / math.sqrt(2.0)
    s = (real_fields[2] + 1j * real_fields[3]) / math.sqrt(2.0)
    x = (real_fields[4] + 1j * real_fields[5]) / math.sqrt(2.0)
    f_x = kappa * (s * b - v**2)
    f_b = kappa * x * s
    f_s = kappa * x * b
    d_f = 3.0 * abs(b) ** 2 - 3.0 * abs(s) ** 2
    return float(abs(f_x) ** 2 + abs(f_b) ** 2 + abs(f_s) ** 2 + 0.5 * gauge_g**2 * d_f**2)


def singlet_branch(kappa: float = 1.0, gauge_g: float = 0.5, v: float = 1.0) -> dict[str, object]:
    point = np.array([math.sqrt(2.0) * v, 0.0, math.sqrt(2.0) * v, 0.0, 0.0, 0.0])
    potential = lambda values: singlet_potential_real(values, kappa=kappa, gauge_g=gauge_g, v=v)
    numerical_hessian = finite_hessian(potential, point)
    numerical_eigenvalues = np.linalg.eigvalsh(numerical_hessian)
    mass_f_squared = 2.0 * kappa**2 * v**2
    mass_vector_squared = 36.0 * gauge_g**2 * v**2
    analytic_eigenvalues = np.array(
        [0.0, mass_vector_squared, mass_f_squared, mass_f_squared, mass_f_squared, mass_f_squared]
    )
    analytic_eigenvalues.sort()
    return {
        "superpotential": "kappa X (S_-3 B_+3 - v^2)",
        "field_charges": {"B": 3, "S": -3, "X": 0},
        "vacuum": {"B": v, "S": v, "X": 0.0},
        "F_terms_zero": True,
        "D_term_zero": True,
        "W_zero": True,
        "analytic_mass_squared": {
            "massive_U1_vector_multiplet": mass_vector_squared,
            "two_chiral_multiplets": mass_f_squared,
            "goldstone_before_gauge_fixing": 0.0,
        },
        "chiral_fermion_mass_eigenvalues": [
            -math.sqrt(mass_f_squared),
            0.0,
            math.sqrt(mass_f_squared),
        ],
        "gaugino_higgsino_dirac_mass": math.sqrt(mass_vector_squared),
        "supertrace_M2_exact_SUSY": 0.0,
        "analytic_real_hessian_eigenvalues": analytic_eigenvalues.tolist(),
        "numerical_real_hessian_eigenvalues": numerical_eigenvalues.tolist(),
        "hessian_max_abs_difference": float(
            np.max(np.abs(numerical_eigenvalues - analytic_eigenvalues))
        ),
        "positive_after_gauge_quotient": bool(numerical_eigenvalues[1] > 0.0),
        "R1_compatibility": {
            "preserves_E6": True,
            "preserves_R1_standard_model": True,
            "derives_Jordan_27_orientation": False,
        },
    }


def robustness_scan() -> dict[str, object]:
    failures: list[dict[str, float | str]] = []
    points = 0
    for kappa in np.geomspace(0.05, 2.0, 8):
        for gauge_g in np.geomspace(0.05, 1.5, 8):
            for amplitude in (0.1, 1.0 / math.sqrt(3.0), 2.0):
                for xi in (-2.0, 0.0, 2.0):
                    points += 1
                    result = jordan_branch(
                        kappa=float(kappa),
                        gauge_g=float(gauge_g),
                        amplitude=amplitude,
                        xi=xi,
                    )
                    if (
                        not result["positive_after_gauge_quotient"]
                        or result["hessian_max_abs_difference"] > TOL
                    ):
                        failures.append(
                            {
                                "branch": "Jordan",
                                "kappa": float(kappa),
                                "gauge_g": float(gauge_g),
                                "amplitude": amplitude,
                                "xi": xi,
                            }
                        )
    return {"points": points, "failures": failures, "tolerance": TOL}


def global_group_audit() -> dict[str, object]:
    stueckelberg_order = 72
    singlet_branch_order = math.gcd(stueckelberg_order, 3)
    final_r1_order = math.gcd(stueckelberg_order, 3, 2)
    return {
        "stueckelberg_order_assumed": stueckelberg_order,
        "U1_only_with_S_minus3_VEV": f"Z_{singlet_branch_order}",
        "U1_only_with_S_minus3_and_Phi_plus1_VEVs": "trivial",
        "global_group_selected_by_R1_representations": "E6 x U(1)_F",
        "diagonal_quotient_(E6xU1)/Z3": (
            "forbidden by the neutral 27_H: the proposed quotient kernel acts as omega on 27_0"
        ),
        "jordan_branch_in_direct_product": (
            "a diagonal Z_3 can remain by combining alpha=2pi/3 with the inverse E6 center"
        ),
        "singlet_branch_with_charge_3_VEVs_only": f"Z_{singlet_branch_order}",
        "full_R1_with_charge_2_and_charge_3_VEVs": (
            "trivial" if final_r1_order == 1 else f"Z_{final_r1_order}"
        ),
        "residual_order_formula_full_R1": "gcd(72, 3, 2) = 1",
        "decision": (
            "the charge-2 Yukawa dressing and a residual Z_3^F cannot both be retained"
        ),
    }


def main() -> None:
    result = {
        "scope": "Complete local 4D N=1 phase-sector audit; not a 6D microscopic derivation.",
        "jordan_branch": jordan_branch(),
        "singlet_branch": singlet_branch(),
        "robustness_scan_jordan": robustness_scan(),
        "global_group_audit": global_group_audit(),
        "decision": {
            "jordan_branch": "SUSY Minkowski GO, but not a simultaneous final R1 vacuum.",
            "singlet_branch": "SUSY Minkowski GO compatible with R1, but no Jordan selection.",
            "factorized_R1_plus_charge2_plus_charge3": "F=D=0 is possible in global SUSY, but the U(1)_F residual is trivial.",
            "supergravity_and_6D_completion": "open",
            "single_branch_closing_both_claims": False,
        },
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

    jordan = result["jordan_branch"]
    singlet = result["singlet_branch"]
    scan = result["robustness_scan_jordan"]
    if not jordan["positive_after_gauge_quotient"] or not singlet["positive_after_gauge_quotient"]:
        raise SystemExit("a benchmark is not stable after gauge quotient")
    if jordan["hessian_max_abs_difference"] > TOL or singlet["hessian_max_abs_difference"] > TOL:
        raise SystemExit("analytic and numerical Hessians disagree")
    if scan["failures"]:
        raise SystemExit("robustness scan contains failures")


if __name__ == "__main__":
    main()

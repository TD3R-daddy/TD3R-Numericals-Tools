#!/usr/bin/env python3
"""Audit reproductible de la branche compensatrice TD3R S_{-3}.

Le programme ne derive pas les coefficients microscopiques. Il verifie :
  1. la reduction Z_72 -> Z_gcd(72,3) apres condensation de S_{-3};
  2. l'existence d'un point stationnaire radial positif pour un benchmark EFT;
  3. la positivite du Hessien radial;
  4. le rang un et le signe du Hessien de la phase gauge-invariante;
  5. une grille de robustesse predefinie, sans entree observationnelle.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class Parameters:
    lambda_phi: float = 1.0
    lambda_s: float = 1.0
    gauge_g: float = 0.5
    kappa: float = 0.05
    v_phi: float = math.sqrt(3.0)
    v_s: float = 1.0
    xi: float = 0.0
    i3_u: float = 1.0


def potential(rho: float, s: float, psi: float, p: Parameters) -> float:
    d_term = rho * rho - 3.0 * s * s + p.xi
    return (
        0.25 * p.lambda_phi * (rho * rho - p.v_phi**2) ** 2
        + 0.25 * p.lambda_s * (s * s - p.v_s**2) ** 2
        + 0.5 * p.gauge_g**2 * d_term**2
        - 2.0 * p.kappa * p.i3_u * s * rho**3 * math.cos(psi)
    )


def radial_gradient(x: np.ndarray, p: Parameters) -> np.ndarray:
    rho, s = x
    d_term = rho * rho - 3.0 * s * s + p.xi
    return np.array(
        [
            p.lambda_phi * rho * (rho * rho - p.v_phi**2)
            + 2.0 * p.gauge_g**2 * rho * d_term
            - 6.0 * p.kappa * p.i3_u * s * rho**2,
            p.lambda_s * s * (s * s - p.v_s**2)
            - 6.0 * p.gauge_g**2 * s * d_term
            - 2.0 * p.kappa * p.i3_u * rho**3,
        ],
        dtype=float,
    )


def radial_hessian(rho: float, s: float, p: Parameters) -> np.ndarray:
    h_rr = (
        p.lambda_phi * (3.0 * rho * rho - p.v_phi**2)
        + 2.0 * p.gauge_g**2 * (3.0 * rho * rho - 3.0 * s * s + p.xi)
        - 12.0 * p.kappa * p.i3_u * s * rho
    )
    h_ss = (
        p.lambda_s * (3.0 * s * s - p.v_s**2)
        - 6.0 * p.gauge_g**2 * (rho * rho - 9.0 * s * s + p.xi)
    )
    h_rs = -12.0 * p.gauge_g**2 * rho * s - 6.0 * p.kappa * p.i3_u * rho**2
    return np.array([[h_rr, h_rs], [h_rs, h_ss]], dtype=float)


def phase_hessian(rho: float, s: float, p: Parameters) -> np.ndarray:
    # psi = sigma + 3 theta.  The zero eigenvalue is the gauge direction
    # before quotient; it is not an instability.
    prefactor = 2.0 * p.kappa * p.i3_u * s * rho**3
    return prefactor * np.array([[9.0, 3.0], [3.0, 1.0]], dtype=float)


def solve_vacuum(p: Parameters) -> dict[str, object]:
    initial = np.array([p.v_phi, p.v_s], dtype=float)
    solution = root(lambda x: radial_gradient(x, p), initial)
    rho, s = (float(solution.x[0]), float(solution.x[1]))
    h_rad = radial_hessian(rho, s, p)
    h_phase = phase_hessian(rho, s, p)
    grad = radial_gradient(np.array([rho, s]), p)
    return {
        "solver_success": bool(solution.success),
        "rho": rho,
        "s": s,
        "psi": 0.0,
        "potential": potential(rho, s, 0.0, p),
        "gradient_inf_norm": float(np.linalg.norm(grad, ord=np.inf)),
        "d_term_argument": rho * rho - 3.0 * s * s + p.xi,
        "radial_hessian": h_rad.tolist(),
        "radial_eigenvalues": np.linalg.eigvalsh(h_rad).tolist(),
        "phase_hessian_theta_sigma": h_phase.tolist(),
        "phase_eigenvalues_coordinate": np.linalg.eigvalsh(h_phase).tolist(),
        "phase_hessian_rank": int(np.linalg.matrix_rank(h_phase, tol=1.0e-10)),
        "local_minimum_after_gauge_quotient": bool(
            solution.success
            and rho > 0.0
            and s > 0.0
            and np.min(np.linalg.eigvalsh(h_rad)) > 0.0
            and np.max(np.linalg.eigvalsh(h_phase)) > 0.0
        ),
    }


def robustness_scan() -> dict[str, object]:
    failures: list[dict[str, float | str]] = []
    smallest_radial_eigenvalue = math.inf
    points = 0
    for gauge_g in np.linspace(0.2, 1.0, 9):
        for kappa in np.linspace(0.005, 0.1, 20):
            p = Parameters(gauge_g=float(gauge_g), kappa=float(kappa))
            result = solve_vacuum(p)
            points += 1
            min_eigenvalue = float(min(result["radial_eigenvalues"]))
            smallest_radial_eigenvalue = min(smallest_radial_eigenvalue, min_eigenvalue)
            if not result["local_minimum_after_gauge_quotient"]:
                failures.append(
                    {
                        "gauge_g": float(gauge_g),
                        "kappa": float(kappa),
                        "reason": "no_positive_local_minimum",
                    }
                )
    return {
        "grid": {"gauge_g": [0.2, 1.0, 9], "kappa": [0.005, 0.1, 20]},
        "points": points,
        "failures": failures,
        "smallest_radial_eigenvalue": smallest_radial_eigenvalue,
    }


def main() -> None:
    p = Parameters()
    result = {
        "status_scope": (
            "EFT existence test only; coefficients and the renormalized FI term "
            "are not derived microscopically"
        ),
        "discrete_group": {
            "stueckelberg_residual_order": 72,
            "condensing_scalar_charge_absolute": 3,
            "unbroken_order": math.gcd(72, 3),
            "result": "Z_3",
        },
        "parameters": asdict(p),
        "benchmark": solve_vacuum(p),
        "robustness": robustness_scan(),
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["benchmark"]["local_minimum_after_gauge_quotient"]:
        raise SystemExit("benchmark failed")
    if result["robustness"]["failures"]:
        raise SystemExit("robustness grid contains failures")


if __name__ == "__main__":
    main()

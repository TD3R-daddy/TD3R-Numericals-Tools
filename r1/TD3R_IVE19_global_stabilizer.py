#!/usr/bin/env python3
"""Audit global et tests de robustesse de la branche TD3R IV-E19/R1.

Ce programme ferme uniquement ce qu'il calcule :

FERMÃ‰
  * noyau Gamma_20 de SU(5) x U(1)_chi x U(1)_psi -> E6 sur le 27 ;
  * invariants de Smith (1,20) de la matrice de charges des VEVs ;
  * absorption des 20 solutions toriques par Gamma_20 ;
  * exhaustivitÃ© des quatre orbites (52 racines hors SU(5)) ;
  * rÃ©gularitÃ© du VEV adjoint hors quatre murs calculÃ©s ;
  * noyau de la matrice complÃ¨te 78 x 78 quand (v,w) varient et que les
    autres VEVs restent gelÃ©s ;
  * rÃ©sidu Z_2^F produit par les VEVs de charges +/-2 ;
  * Hessienne F, restreinte aux 19 singulets SM de IV-E18.

NON FERMÃ‰ PAR CE PROGRAMME
  * origine microscopique de v=w=1 ;
  * Hessienne de tous les champs 27+27bar+351'+351'bar+78 ;
  * corrections de Coleman-Weinberg et stabilitÃ© radiative ;
  * couplage radion--VEVs et compatibilitÃ© dynamique avec le chapitre 29 ;
  * chaÃ®ne A_s,n_s -> Boltzmann -> sigma_8.

Les normalisations et le potentiel singulet proviennent de
TD3R_IVE18_R1_vacuum_mass78.py. Une variation de (v,w) avec les autres VEVs
et les couplages gelÃ©s teste les rÃ©sidus F, mais ne remplace pas une nouvelle
minimisation multi-champ.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from functools import reduce
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

import TD3R_IVE18_R1_vacuum_mass78 as r12


TOL = 1.0e-12
ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "TD3R_IVE19_outputs" / "TD3R_IVE19_global_stabilizer_report.json"


# (nom, exposant du centre SU(5), q_chi, q_psi)
WEIGHTS_27 = (
    ("10", 2, -1, 1),
    ("bar5_16", -1, 3, 1),
    ("1_16", 0, -5, 1),
    ("5_10", 1, 2, -2),
    ("bar5_10", -1, -2, -2),
    ("1_1", 0, 0, 4),
)

# VEVs c2, e1, e3 et e4. Les conjuguÃ©s portent les charges opposÃ©es.
STANDARD_VEV_CHARGES = np.asarray(
    [(-5, 1), (-10, 2), (0, 8), (0, -4)], dtype=int
)

# Les quatre orbites de racines de E6 \ SU(5).
ROOT_ORBIT_MULTIPLICITIES = {
    "45/(10+10bar)": 20,
    "(16+16bar)/(10+10bar)": 20,
    "(16+16bar)/(5+5bar)": 10,
    "(16+16bar)/(1+1)": 2,
}

# (coefficient de v, coefficient de w, prÃ©facteur positif). Les parois et
# les masses sont ainsi produites par une source unique : m^2=c(a v+b w)^2.
ROOT_ORBIT_FORMS = {
    "45/(10+10bar)": (1.0, 0.0, 4.0 / 15.0),
    "(16+16bar)/(10+10bar)": (-1.0, math.sqrt(15.0), 1.0 / 60.0),
    "(16+16bar)/(5+5bar)": (math.sqrt(3.0 / 5.0), 1.0, 1.0 / 4.0),
    "(16+16bar)/(1+1)": (math.sqrt(5.0 / 6.0), -math.sqrt(1.0 / 2.0), 1.0),
}


def close_to_one(z: complex, tol: float = TOL) -> bool:
    return abs(z - 1.0) < tol


def acts_trivially_on_27(k: int, alpha: float, beta: float) -> bool:
    """Action de (zeta^k,exp(i alpha),exp(i beta)) sur le 27."""
    zeta = cmath.exp(2j * math.pi / 5)
    return all(
        close_to_one(
            zeta ** (center_exp * k)
            * cmath.exp(1j * (q_chi * alpha + q_psi * beta))
        )
        for _, center_exp, q_chi, q_psi in WEIGHTS_27
    )


def verify_gamma20() -> dict[str, object]:
    """VÃ©rifie les 20 puissances du gÃ©nÃ©rateur Gamma_20."""
    kernel = []
    for n in range(20):
        alpha = math.pi * n / 10
        beta = math.pi * n / 2
        k = (2 * n) % 5
        if not acts_trivially_on_27(k, alpha, beta):
            raise AssertionError(f"gamma^{n} n'agit pas trivialement sur le 27")
        kernel.append((k, n % 20, n % 4))
    if len(set(kernel)) != 20:
        raise AssertionError(le generateur proposÃ© n'est pas d'ordre 20)
    return {
        "order": 20,
        "generator": "(heta^2, exp(i*pi/10), exp(i*pi/2))",
        "verified_on": [name for name, *_ in WEIGHTS_27],
    }


def smith_invariants_rank2(charges: np.ndarray) -> tuple[int, int]:
    """Invariants de Smith d'une matrice entiÃ¨re de rang deux."""
    q = np.asarray(charges, dtype=int)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("la matrice de charges doit avoir exactement deux colonnes")
    d1 = reduce(math.gcd, (abs(int(x)) for x in q.flat))
    minors = [
        abs(int(q[i, 0] * q[j, 1] - q[i, 1] * q[j, 0]))
        for i, j in combinations(range(q.shape[0]), 2)
    ]
    nonzero = [m for m in minors if m]
    if not nonzero:
        raise ValueError("la matrice de charges n'est pas de rang deux")
    d2 = reduce(math.gcd, nonzero) // d1
    return d1, d2


def verify_charge_lattice() -> dict[str, object]:
    """VÃ©rifie Smith=(1,20) et son invariance sous GL(2,Z)."""
    base = smith_invariants_rank2(STANDARD_VEV_CHARGES)
    if base != (1, 20):
        raise AssertionError(f "Smith inattendue : {base}")

    transforms = (
        np.asarray(((1, 1), (0, 1)), dtype=int),
        np.asarray(((0, 1), (1, 0)), dtype=int),
        np.asarray(((1, 0), (1, 1)), dtype=int),
        np.asarray(((-1, 0), (0, 1)), dtype=int),
    )
    tests = []
    for u in transforms:
        det = int(round(np.linalg.det(u)))
        if abs(det) != 1:
            raise AssertionError("la matrice test n'appartient pas Ã  GL(2,Z)")
        invariant = smith_invariants_rank2(STANDARD_VEV_CHARGES @ u)
        if invariant != base:
            raise AssertionError("les invariants de Smith changent sous GL(2,Z)")
        tests.append({"matrix": u.tolist(), "det": det, "smith": list(invariant)})

    return {
        "charges": STANDARD_VEV_CHARGES.tolist(),
        "smith": list(base),
        "basis_change_tests": tests,
        "interpretation": (
            "Les charges E6 ne sont pas des paramÃ¨tres libres. "
            "Un changement de base GL(2,Z) conserve la conclusion; "
            "des charges arbitrairement diffÃ©rentes dÃ©finiraient un autre modÃ¨le."
        ),
    }


def verify_torus_absorption() -> dict[str, object]:
    """VÃ©rifie les 20 solutions et leur identification Ã  Gamma_20."""
    solutions = []
    for n in range(20):
        alpha = math.pi * n / 10
        beta = math.pi * n / 2
        phases = (
            cmath.exp(1j * (-5 * alpha + beta)),
            cmath.exp(1j * (-10 * alpha + 2 * beta)),
            cmath.exp(8j * beta),
            cmath.exp(-4j * beta),
        )
        if not all(close_to_one(z) for z in phases):
            raise AssertionError(f"solution torique n={n} invalide")
        solutions.append((n % 20, n % 4))
    if len(set(solutions)) != 20:
        raise AssertionError("le tore rÃ©siduel n'a pas 20 points")
    return {
        "covering_torus_solutions": 20,
        "identified_with": "Gamma_20",
        "physical_component_group_in_E6": "trivial",
    }


def adjoint_root_masses(v: float, w: float) -> dict[str, float]:
    """Coefficients de masse du seul VEV adjoint, en unitÃ©s communes."""
    return {
        name: prefactor * (a * v + b * w) ** 2
        for name, (a, b, prefactor) in ROOT_ORBIT_FORMS.items()
    }


def wall_functions(v: float, w: float) -> dict[str, floatL]:
    """Formes linÃ©aires, dÃ©rivÃ©es des masses, nulles sur les parois."""
    labels = {
        "45/(10+10bar)": "v=0",
        "(16+16bar)/(10+10bar)": "v=sqrt(15)w",
        "(16+16bar)/(5+5bar)": "w=-sqrt(3/5)v",
        "(16+16bar)/(1+1)": "w=sqrt(5/3)tˆ°(€€€ô(€€€É•ÑÕÉ¸ì(€€€€€€€±…‰•±Ím¹…µ•tè„€¨Ø€¬ˆ€¨Ü(€€€€€€€™½È¹…µ”°€¡„°ˆ°|¤¥¸I==Q}=I	%Q}=I5L¹¥Ñ•µÌ ¤(€€€ô(()‘•˜É½½Ñ}½É‰¥Ñ}…Õ‘¥Ð¡Øè™±½…Ð°Üè™±½…Ð°Ñ½°è™±½…Ð€ôQ=0¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€€ˆˆ‰á¡…ÕÍÑ¥Ù¥Ó¤•ÐË¥Õ±…É¥Ó¤‘Ô•¹ÑÉ…±¥Í…Ñ•ÕÈ…‘©½¥¹Ð¸ˆˆˆ(€€€µ…ÍÍ•Ì€ô…‘©½¥¹Ñ}É½½Ñ}µ…ÍÍ•Ì¡Ø°Ü¤(€€€Ý…±±Ì€ôÝ…±±}™Õ¹Ñ¥½¹Ì¡Ø°Ü¤(€€€µ¥¹¥µÕ´€ôµ¥¸¡µ…ÍÍ•Ì¹Ù…±Õ•Ì ¤¤(€€€É•ÑÕÉ¸ì(€€€€€€€€‰Ù¥•Ý}Á…É…µ•Ñ•ÉÌˆèì‰ØˆèØ°€‰ÜˆèÝô°(€€€€€€€€‰µ…ÍÍ}½•™™¥¥•¹ÑÌˆèµ…ÍÍ•Ì°(€€€€€€€€‰Ý…±±}™Õ¹Ñ¥½¹ÌˆèÝ…±±Ì°(€€€€€€€€‰µ¥¹¥µÕµ}µ…ÍÍ}½•™™¥¥•¹Ðˆèµ¥¹¥µÕ´°(€€€€€€€€‰…‘©½¥¹Ñ}•¹ÑÉ…±¥é•É}É•Õ±…Èˆèµ¥¹¥µÕ´€øÑ½°°(€€€€€€€€‰MTÕ}É½½ÑÌˆè€ÈÀ°(€€€€€€€€‰½ÕÑÍ¥‘•}MTÕ}É½½ÑÌˆèÍÕ´¡I==Q}=I	%Q}5U1Q%A1%%Q%L¹Ù…±Õ•Ì ¤¤°(€€€€€€€€‰Ù}É½½ÑÍ}Ñ½Ñ…°ˆè€ÈÀ€¬ÍÕ´¡I==Q}=I	%Q}5U1Q%A1%%Q%L¹Ù…±Õ•Ì ¤¤°(€€€€€€€€‰µÕ±Ñ¥Á±¥¥Ñ¥•ÌˆèI==Q}=I	%Q}5U1Q%A1%%Q%L°(€€€ô(()‘•˜É•ÍÑÉ¥Ñ•‘}Í¥¹±•Ñ}™}¡•ÍÍ¥…¸ ¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€‰MÕ‰±¥µ”‘”¡•ÍÍ¥•¹¹”‘•Í€ÄäÍ¥¹Õ±•ÑÌM4¸ˆ(€€Œ¥µÁ½ÉÐµ…ÍÌµ…ÑÉ¥à™É½´ÈÄà‰•¹¡µ…É¬(€€À€ôÈÄà¹‰•¹¡µ…É­}Á…É…µ•Ñ•ÉÌ ¤(€€€à€ôÈÄà¹‰•¹¡µ…É­}Ù•ÙÌ ¤(€€€€Œ ½½¹‘¥Ñ¥½¸Ù¥„™¥¹¥Ñ”‘¥™™•É•¹”½˜É…‘¥•¹Ð(€€€ €ô€Å”´Ô(€€€¸€ôà¹Í¥é”(€€€ €ô¹À¹é•É½Ì ¡¸°¸¤¤(€€€™½È¤¥¸É…¹”¡¸¤è(€€€€€€€áÀ€ôà¹½Áä ¤ìáÁm¥t€¬ô (€€€€€€á´€ôà¹½Áä ¤ìáµm¥t€´ô (€€€€€€€À€ôÈÄà¹½µÁ±•á}ÍÑ•Á}É…‘¥•¹Ð¡áÀ°À¤(€€€€€€€´€ôÈÄà¹½µÁ±•á}ÍÑ•Á}É…‘¥•¹Ð¡á´°À¤(€€€€€€€!qm¤±qt€ô€¡À€´´¤¼ É ¤(€€€ €ô€À¸Ô¨¡ €¬ ¹P¤(€€€•Ù…±Ì€ô¹À¹±¥¹…±œ¹•¥Ù…±Í ¡ ¹P  ¤(€€€Ñ½°€ô€Å”´ÄÀ(€€€¹•œ€ô¥¹Ð¡¹À¹ÍÕ´¡•Ù…±Ì€ð€µÑ½°¤¤(€€€¹Õ±°€ô¥¹Ð¡¹À¹ÍÕ´¡¹À¹…‰Ì¡•Ù…±Ì¤€ðôÑ½°¤¤(€€€É…¹¬€ô¥¹Ð¡¹À¹ÍÕ´¡•Ù…±Ì€øÑ½°¤¤(€€€Á½Ì€ô•Ù…±Ím•Ù…±Ì€øÑ½±t(€€€É•ÑÕÉ¸ì(€€€€€€€€‰‘¥µ•¹Í¥½¸ˆè¸°(€€€€€€€€‰É…¹¬ˆèÉ…¹¬°(€€€€€€€€‰¹Õ±±¥Ñäˆè¹Õ±°°(€€€€€€€€‰¹•…Ñ¥Ù•}µ½‘•Ìˆè¹•œ°(€€€€€€€€‰µ¥¹¥µÕµ}Á½Í¥Ñ¥Ù•}•¥•¹Ù…±Õ”ˆè™±½…Ð¡Á½Ì¹µ¥¸ ¤¤¥˜Á½Ì¹Í¥é”•±Í”9½¹”°(€€€€€€€€‰µ…á¥µÕµ}•¥•¹Ù…±Õ”ˆè™±½…Ð¡•Ù…±Ì¹µ…à ¤¤°(€€€€€€€€‰•¥•¹Ù…±Õ•Ìˆèm™±½…Ð¡à¤™½Èà¥¸•Ù…±Ít°(€€€€€€€€‰Í•Ñ½Èˆè€ˆÄäM4µÍ¥¹±•ÐÉ•…°½½É‘¥¹…Ñ•ÌìµÑ•É´½¹±äˆ°(€€€€€€€€‰¥¹Ñ•ÉÁÉ•Ñ…Ñ¥½¸ˆè€‰1•ÌÅÕ…ÑÉ”é•É½Ì½µÁ…Ñ¥‰±•Ì…Ù•Œ±•ÌÅÕ…ÑÉ”µÕ±Ñ¥Á±•ÑÌ‘”½±‘ÍÑ½¹”¹•ÕÑÉ•Ì…ÑÑ•¹‘ÕÌ°µ…¥Ì±•ÕÈ¥‘•¹Ñ¥™¥…Ñ¥½¸•á¥”±•ÌÙ•Ñ•ÕÉÌÁÉ½ÁÉ•Ì‘”©…Õ”¸•ÑÑ”!•ÍÍ¥•¹¹”¹”™•Éµ”Á…Ì±„ÍÑ…‰¥±¥Ñ”Í…±…¥É”½µÁ³¡Ñ”¸ˆ(€€€ô(()‘•˜™Õ±±}…Õ•}µ…ÍÍ}…Õ‘¥Ð¡Øè™±½…Ð°Üè™±½…Ð¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€€ˆˆ‰½¹ÍÑÉÕ¥Ð„™Õ±°€ÜáàÜàµ…ÍÌµ…ÑÉ¥à…¹…Õ‘¥ÑÌ¥ÑÌ­•É¹•°¸ˆˆˆ(€€€à€ôÈÄà¹‰•¹¡µ…É­}Ù•ÙÌ ¤(€€€€Œ½Ù•ÉÝÉ¥Ñ”…‘©½¥¹ÐØ±Ü(€€€à€ôà¹½Áä ¤(€€€€ô‘¥Ð¡é¥À¡ÈÄà¹%1}95L°à¤¤(€€€‘l‰Ø‰t€ôØì‘l‰Ü‰t€ôÜ(€€€à€ô¹À¹…Í…ÉÉ…ä¡m‘l¹…µ•t™½È¹…µ”¥¸ÈÄà¹%1}95Mt¤(€€€ÍÁ•ÑÉÕ´€ôÈÄà¹µ…ÍÍ}ÍÁ•ÑÉÕ´¡à¤(€€€•¥Ù…±Ì€ô¹À¹…Í…ÉÉ…ä¡mè™½È‰±½¬¥¸ÍÁ•ÑÉÕ´™½Èè¥¸‰±½­l‰•¥•¹Ù…±Õ•Ì‰ut¤(€€€µ…ÑÉ¥à€ô¹À¹‘¥…œ¡•¥Ù…±Ì¤(€€€•Ø€ô¹À¹±¥¹…±œ¹•¥Ù…±Í ¡µ…ÑÉ¥à¤(€€€Ñ½°€ô€Å”´ÄÀ(€€€¹Õ±°€ô¥¹Ð¡¹À¹ÍÕ´¡¹À¹…‰Ì¡•Ø¤€ðÑ½°¤¤(€€€É…¹¬€ô¥¹Ð¡¹À¹ÍÕ´¡•Ø€øÑ½°¤¤(€€€É•ÑÕÉ¸ì(€€€€€€€€‰‘¥µ•¹Í¥½¸ˆè€Üà°(€€€€€€€€‰É…¹¬ˆèÉ…¹¬°(€€€€€€€€‰­•É¹•±}‘¥µ•¹Í¥½¸ˆè¹Õ±°°(€€€€€€€€‰M5}1¥•}…±•‰É…}­•É¹•°ˆè¹Õ±°€ôô€ÄÈ°(€€€€€€€€‰1¥•}…±•‰É…}¥™}­•É¹•±}¥Í|ÄÈˆè€‰ÍÔ Ì¤­ÍÔ È¤­Ô Ä¤ˆ°(€€€€€€€€‰¹•…Ñ¥Ù•}•¥•¹Ù…±Õ•Ìˆè¥¹Ð¡¹À¹ÍÕ´¡•Ø€ð€µÑ½°¤¤°(€€€€€€€€‰µ¥¹¥µÕµ}Á½Í¥Ñ¥Ù•}•¥•¹Ù…±Õ”ˆè™±½…Ð¡•Ùm•Ø€øÑ½±t¹µ¥¸ ¤¤¥˜¹À¹…¹ä¡•Ø€øÑ½°¤•±Í”9½¹”°(€€€€€€€€‰µ…á¥µÕµ}•¥•¹Ù…±Õ”ˆè™±½…Ð¡•Ø¹µ…à ¤¤°(€€€€€€€€‰Í½Á”ˆè€‰…±°€Üà…Õ”‰½Í½¹Ìì¹½¸µ…‘©½¥¹ÐYÙÌ™É½é•¸…Ð‰•¹¡µ…É¬ˆ°(€€€€€€€€‰Ý…É¹¥¹œˆè€‰‘¥ÍÁ±…•€¡Ø±Ü¤•¹•É…±±äÙ¥½±…Ñ•Ìµ™±…Ñ¹•ÍÌÕ¹±•ÍÌ…±°™¥•±‘Ì…É”É”µµ¥¹¥µ¥é•ìÑ¡¥Ì¥Ì„ÍÑ…‰¥±¥é•È…Õ‘¥Ð°¹½Ð„Ù…ÕÕ´Í½±ÕÑ¥½¸¸ˆ°(€€€ô()‘•˜Í…¹}ÙÝ}‰½à¡ØÀè™±½…Ð°ÜÀè™±½…Ð°™É…Ñ¥½¸è™±½…Ð°Á½¥¹ÑÌè¥¹Ð¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€€‰M…¸‰½à…É½Õ¹€¡ØÀ±ÜÀ¤™½È•¹ÑÉ…±¥é•ÈÍÑÉÕÑÕÉ”½¹±ä¸ˆ(€€€ÙÌ€ô¹À¹±¥¹ÍÁ…”¡ØÀ¨ Äµ™É…Ñ¥½¸¤°ØÀ¨ Ä­™É…Ñ¥½¸¤°Á½¥¹ÑÌ¤(€€€ÝÌ€ô¹À¹±¥¹ÍÁ…”¡ÜÀ¨ Äµ™É…Ñ¥½¸¤°ÜÀ¨ Ä­™É…Ñ¥½¸¤°Á½¥¹ÑÌ¤(€€€µ¥¹}…À€ôµ…Ñ ¹¥¹˜(€€€µ¥¹}…Ð€ô9½¹”(€€€Í¥¹Õ±…È€ô€À(€€€™½ÈØ¥¸ÙÌè(€€€€€€€™½ÈÜ¥¸ÝÌè(€€€€€€€€€€€…Õ‘¥Ð€ôÉ½½Ñ}½É‰¥Ñ}…Õ‘¥Ð¡™±½…Ð¡Ø¤°™±½…Ð¡Ü¤¤(€€€€€€€€€€€¥˜¹½Ð…Õ‘¥Ñl‰…‘©½¥¹Ñ}•¹ÑÉ…±¥é•É}É•Õ±…È‰tè(€€€€€€€€€€€€€€€Í¥¹Õ±…È€¬ô€Ä(€€€€€€€€€€€™½È´¥¸…Õ‘¥Ñl‰µ…ÍÍ}½•™™¥¥•¹ÑÌ‰t¹Ù…±Õ•Ì ¤è(€€€€€€€€€€€€€€€¥˜´€ðµ¥¹}…Àè(€€€€€€€€€€€€€€€€€€€µ¥¹}…À€ô´(€€€€€€€€€€€€€€€€€€€µ¥¹}…Ð€ô€¡™±½…Ð¡Ø¤°™±½…Ð¡Ü¤¤(€€€É•ÑÕÉ¸ì(€€€€€€€€‰‰½àˆèì‰ØˆŽˆÙ›Ø]
œÖÌJK›Ø]
œÖËLWJWKÈŽˆÙ›Ø]
ÜÖÌJK›Ø]
ÜÖËLWJWKœÚ[×Ü\—Ø^\ÈŽˆÚ[ßKˆœØ[\\ÈŽˆ[
Ú[ÊœÚ[ÊKˆœÚ[™Ý[\—ÜØ[\\ÈŽˆÚ[™Ý[\‹ˆ›Z[š[][WÛX\Ü×ØÛÙY™šXÚY[ŽˆZ[—ÙØ\ˆ›Z[š[][WØ]ŽˆZ[—Ø]ˆš[\œ™]][ÛˆŽˆ”›Ø\Ý\ÜÙHH\HHÙ[˜[\Ø]]\ˆÙ][[Y[ˆ[H™H›Ý]™H\È]YHÚ\]YHÚ[\Ý[ˆ^™[][HHÝ[Y[ˆ‹ˆB‚‚™Yˆ—Ù›]™\Ü×Ü\\˜˜][Û—Ø]Y]
[\Îˆ]\˜X›VÙ›Ø]JHOˆXÝÜÝ‹Øš™XÝN‚ˆHŒN˜™[˜ÚX\š×Ü\˜[Y]\œÊ
Bˆ˜\ÙHHŒN˜™[˜ÚX\š×Ý™]œÊ
Bˆ™\Ý[ÈH×Bˆ›Üˆ[H[ˆ[\Î‚ˆH˜\ÙK˜ÛÜJ
Bˆ›Üˆ˜[YH[ˆ
ˆ‹ÈŠN‚ˆHHŒN‘’QSÓSQTËš[™^
˜[YJBˆÚWH
ÏH[BˆÜ˜YHŒN˜ÛÛ\^ÜÝ\ÙÜ˜YY[

Bˆ™\Ý[Ë˜\[™
È™[HŽˆ[K›X^ØXœ×Ñ—ÑM—Ùœ›Þ™[—Ý™]œÈŽˆ›Ø]
œ›X^
œ˜XœÊÜ˜Y
JJ_JBˆ™]\›ˆÂˆœ™\Ý[ÈŽˆ™\Ý[Ëˆš[\œ™]][ÛˆŽˆ“›Ûž™\›È™\ÚYX[È\™H^XÝYÚ[ˆÛ›H
‹ÊH\™H\ÜXÙY[™[Ý\ˆ‘UœÈ\™Hœ›Þ™[‹ˆ\È\È›ÝH˜XÝ][HÝXš[]H\Ýˆ‹ˆB‚™YˆY\˜\˜ÚWÙXYÛ›ÜÝXÊ\ÜÚX[ŽˆXÝÜÝ‹Øš™XÝKÛÝ™\—ÓQÎˆ›Ø]›Û™K[X™RÒ×ÛÝ™\—ÓQÎˆ›Ø]›Û™K˜XÝÜŽˆ›Ø]
HOˆXÝÜÝ‹Øš™XÝN‚ˆ[Z[ˆH\ÜÚX[–È›Z[š[][WÜÜÚ]]™WÙZYÙ[˜[YH—Bˆ™]\›ˆÂˆ™]˜[X]YŽˆÛÝ™\—ÓQÈ\È›Ý›Û™H[™[X™RÒÈÝ™\—ÓQÈ\È›Ý›Û™Kˆ’ÛÝ™\—ÓQÈŽˆÛÝ™\—ÓQËˆ“[X™RÒ×ÛÝ™\—ÓQÈŽˆ[X™RÒÈÝ™\—ÓQËˆœ™\ÝšXÝYÛZ[š[][WÛX\Ü×ÛÝ™\—ÓQÈŽˆX]œÜ\
[Z[ŠHYˆ[Z[ˆ\È›Ý›Û™H[ÙH›Û™Kˆ™˜XÝÜˆŽˆ˜XÝÜ‹ˆ’ØÚXÚÈŽˆ

X]œÜ\
[Z[ŠHˆ˜XÝÜˆ
ˆÛÝ™\—ÓQÊHYˆÛÝ™\—ÓQÈ\È›Ý›Û™H[ÙH›Û™JKˆ’Ò×ØÚXÚÈŽˆ

[X™RÒ×ÛÝ™\—ÓQÈˆ˜XÝÜˆ
ˆX]œÜ\
[Z[ŠJHYˆ[X™RÒ×ÛÝ™\—ÓQÈ\È›Ý›Û™H[ÙH›Û™JKˆœØÛÜHŽˆ•\È\Ý\Ù\ÈÛ›HH™\ÝšXÝYNK\Ú[™Û]‹]\›H\ÜÚX[‹›ÝHÛÛ\]HÛÝ˜\šX[ØØ[\ˆ\ÜÚX[‹ˆ‹ˆB‚™YˆÛÜÝ\™WÛYÙ\Š
HOˆXÝÜÝ‹Øš™XÝN‚ˆ™]\›ˆÂˆ[š]\š]WØ˜\ÙWÜÚ[ŽˆÈœÝ]\ÈŽˆ˜ÛÜÙYØ]Ø™[˜ÚX\šÈ‹œ™\Ý[Žˆ•HŒH™[˜ÚX\šÈ\È‹ÑY›]È[Y\šXØ[Û\˜[˜ÙHŸKˆ™ÛØ˜[ÜÝXš[^™\ˆŽˆÈœÝ]\ÈŽˆ˜ÛÜÙYØ]Ø™[˜ÚX\šÈ‹œ™\Ý[Žˆ’Ù\›™[[Y[œÚ[ÛˆLˆ[™›È™YØ]]™HØ]YÙKX›ÜÛÛˆ]˜[Y\È]]ÏLHŸKˆœ˜YX]]™WÜÝXš[]HŽˆÈœÝ]\ÈŽˆ›Ü[ˆ‹›™YYYŽˆ™šY[Y\[™[[ÜXÝ[K™[›Ü›X[^˜][Ûˆ™\ØÜš\[Ûˆ[™ÛÛ[X[‹UÙZ[˜™\™È\ÜÚX[ˆŸKˆ˜Ú\\ŒŽWØÛÛ\]Xš[]HŽˆÈœÝ]\ÈŽˆ˜ÛÛ\]X›WØ]ÍÑQ•Û]™[ÛÛ›HŸKˆœ˜Y[Û—ØÛÝ\[™ÈŽˆÈœÝ]\ÈŽˆ›Ü[ˆŸKˆœš[[Ü™X[Ý×Ø›Û›X[›—Ý×ÜÚYÛXNŽˆÈœÝ]\ÈŽˆœÙ\\˜]WÛÜ[—Ü›ÙÜ˜[HŸKˆ›][YšY[ÒÒÒ×ÚY\˜\˜ÚHŽˆÈœÝ]\ÈŽˆœ\X[WÚ[\[Y[YŸKˆœ˜Y[Û—Ø˜\ž[Û—Ùš\Ú\—ÔXš[—ÖLLŽˆÈœÝ]\ÈŽˆœÙ\\˜]WÛÜ[—Ü›ÙÜ˜[HŸKˆÓTÔ×ÐÐSPˆŽˆÈœÝ]\ÈŽˆœÙ\\˜]WÛÜ[—Ü›ÙÜ˜[HŸKˆ›\WÜ›ÜÜ[Û˜[ÌWÜ\×ÞˆŽˆÈœÝ]\ÈŽˆœ\˜[Y]\š^™YØ™[˜ÚX\š×Û›ÝÙ[™[Y[[Ü™YXÝ[ÛˆŸKˆ™œ›Þ™[—ÐPÕÔÔÝ×ÑTÒWÔXš[—Ý\ÝŽˆÈœÝ]\ÈŽˆœ›ÝØÛÛÙYš[™YØ]Û›ÝÙ^XÝ]YŸKˆB‚‚™YˆZ[Ü™\Ü
\™ÜÎˆ\™Ü\œÙK“˜[Y\ÜXÙJHOˆXÝÜÝ‹Øš™XÝN‚ˆ\ÜÚX[ˆH™\ÝšXÝYÜÚ[™Û]Ù—Ú\ÜÚX[Š
Bˆ™]\›ˆÂˆœØÛÜHŽˆ•ÔˆU‹QLNKÔŒHÛØ˜[ÝXš[^™\ˆ[™›Ý[™Y›Ø\Ý™\ÜÈ]Y]‹ˆ™Ø[[XLŒŽˆ™\šYžWÙØ[[XLŒ

Kˆ˜Ú\™ÙWÛ]XÙHŽˆ™\šYžWØÚ\™ÙWÛ]XÙJ
KˆÜ\×ØXœÛÜœ[ÛˆŽˆ™\šYžWÝÜ\×ØXœÛÜœ[ÛŠ
Kˆœ›ÛÝÛÜ˜š]ÈŽˆ›ÛÝÛÜ˜š]Ø]Y]
\™ÜË‹\™ÜËÊKˆ™[ÙØ]YÙWÛX\Ü×ÍÎŽˆ[ÙØ]YÙWÛX\Ü×Ø]Y]
\™ÜË‹\™ÜËÊKˆ×ÜØØ[ˆŽˆØØ[—Ý×Ø›Þ
\™ÜË‹\™ÜËË\™ÜËœØØ[—Ùœ˜XÝ[Û‹\™ÜËœØØ[—ÜÚ[ÊKˆ‘—Ù›]™\Ü×Ü\\˜˜][ÛœÈŽˆ—Ù›]™\Ü×Ü\\˜˜][Û—Ø]Y]

ŒKŒKŒL
JKˆœ™\ÝšXÝYÜÚ[™Û]Ñ—Ú\ÜÚX[ˆŽˆ\ÜÚX[‹ˆšY\˜\˜ÚWÙXYÛ›ÜÝXÈŽˆY\˜\˜ÚWÙXYÛ›ÜÝXÊ\ÜÚX[‹\™ÜË’ÛÝ™\—ÓQË\™ÜË“[X™RÒ×ÛÝ™\—ÓQË\™ÜËšY\˜\˜ÚWÙ˜XÝÜŠKˆ•LQˆŽˆÈœš[Z]]™WØÚ\™ÙWÜ™\Ù[ŽˆK˜ÛÛ™[œÙYØÚ\™Ù\ÈŽˆÌ‹L—Kœ™\ÚYX[ÙÜ›Ý\Žˆ–—Ì—‘ˆŸKˆ˜ÛÜÝ\™WÛYÙ\ˆŽˆÛÜÝ\™WÛYÙ\Š
KˆB‚‚™Yˆ\œÙWØ\™ÜÊ
HOˆ\™Ü\œÙK“˜[Y\ÜXÙN‚ˆ\œÙ\ˆH\™Ü\œÙK\™Ý[Y[\œÙ\Š\ØÜš\[ÛW×ÙØ××ÊBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K]ˆ‹\OY›Ø]Y˜][LKŒ
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K]È‹\OY›Ø]Y˜][LKŒ
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\ØØ[‹Yœ˜XÝ[Ûˆ‹\OY›Ø]Y˜][LŒL
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\ØØ[‹\Ú[È‹\OZ[Y˜][LLJBˆ\œÙ\‹˜YØ\™Ý[Y[
‹KR[Ý™\‹SQÈ‹\ÝH’ÛÝ™\—ÓQÈ‹\OY›Ø]Y˜][S›Û™JBˆ\œÙ\‹˜YØ\™Ý[Y[
‹KS[X™RÒË[Ý™\‹SQÈ‹\ÝH“[X™RÒ×ÛÝ™\—ÓQÈ‹\OY›Ø]Y˜][S›Û™JBˆ\œÙ\‹˜YØ\™Ý[Y[
‹KZY\˜\˜ÚKY˜XÝÜˆ‹\OY›Ø]Y˜][LLŒ
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K[Ý]]‹\OT]Y˜][QQUSÔ‘TÔ•
Bˆ™]\›ˆ\œÙ\‹œ\œÙWØ\™ÜÊ
B‚‚™YˆXZ[Š
HOˆ›Û™N‚ˆ\™ÜÈH\œÙWØ\™ÜÊ
Bˆ™\ÜHZ[Ü™\Ü
\™ÜÊBˆ\™ÜË›Ý]]œ\™[›ZÙ\Š\™[ÏUYK^\ÝÛÚÏUYJBˆ\™ÜË›Ý]]Üš]WÝ^
œÛÛ‹™[\Ê™\Ü[™[L‹ÛÜÚÙ^\ÏUYJH
È—ˆ‹[˜ÛÙ[™ÏH]‹NŠBˆš[
‘Ø[[XWÌŒ]XœÛÜœ[Ûˆ\ÈŒÛÛ][ÛœÈˆ°ê\šYšpê\ÈŠBˆš[
”ÛZ]
WÕ‘UŠHHXYÊKŒ
K[˜\šX[ÛÝ\ÈÓ
‹ŠHŠBˆ›ÛÝÈH™\ÜÈœ›ÛÝÛÜ˜š]È—Bˆš[
ˆ“Ü˜š]\ÈH˜XÚ[™\ÈÜœÈÕJJHˆÜÝ[J“ÓÕÓÔ’UÓUSTPÒUQTË˜[Y\Ê
J_KÍLˆ°ê\šYšpêY\ÈŠBˆ[ÙØ]YÙHH™\ÜÈ™[ÙØ]YÙWÛX\Ü×ÍÎ—Bˆš[
ˆ“X]šXÙHH˜]YÙHÛÛ\0êHˆ˜[™Ï^Ù[ÙØ]YÙVÉÜ˜[šÉ×_K›ÞX]O^Ù[ÙØ]YÙVÉÚÙ\›™[Ù[Y[œÚ[Û‰×_K[ðêœ™HÓO^Ù[ÙØ]YÙVÉÔÓWÓYWØ[ÙXœ˜WÚÙ\›™[	×_HŠBˆ\ÜÚX[ˆH™\ÜÈœ™\ÝšXÝYÜÚ[™Û]Ñ—Ú\ÜÚX[ˆ—Bˆš[
ˆ’\ÜÚY[›™HˆÚ[™Ý[]Èˆ˜[™Ï^Ú\ÜÚX[–ÉÜ˜[šÉ×_K[]0êO^Ú\ÜÚX[–ÉÛ[]I×_K[Ù\È°êYØ]YœÏ^Ú\ÜÚX[–ÉÛ™YØ]]™WÛ[Ù\É×_HŠBˆYˆ[ÙØ]YÙVÈ”ÓWÓYWØ[ÙXœ˜WÚÙ\›™[—N‚ˆš[
”ÝXš[\Ø]]\ˆÛØ˜[ŒHˆÊJÊ^JŠJNÈ°ê\ÚYH˜[Z[X[ˆ—Ì—‘ˆŠBˆ[ÙN‚ˆš[
“HÚ[\Ý0êH‰ØH\ÈH›ÞX]H[™š[š]0ê\Ú[X[ÓHŠBˆš[
US•SÓˆˆÜšYÚ[™H\È‘UœË\ÜÚY[›™HÛÛ\0êK˜Y[Ûˆ]›ÝXÛ\È™\Ý[Ý]™\ÈŠBˆš[
ˆœ˜\ÜˆØ\™ÜË›Ý]]HŠB‚‚šYˆ×Û˜[YW×ÈOH—×ÛXZ[—×ÈŽ‚ˆXZ[Š
B
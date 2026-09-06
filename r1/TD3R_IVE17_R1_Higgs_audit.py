#!/usr/bin/env python3
"""Finite checks used in TD3R IV-E17 (R1-1 Higgs audit)."""

q_matter = 1
q_yukawa_higgs = -2 * q_matter
q_self_cubic_higgs = 0  # 3 q = 0 for a continuous U(1)

assert q_yukawa_higgs == -2
assert q_self_cubic_higgs == 0
assert q_yukawa_higgs != q_self_cubic_higgs

q_s_minus = -2
q_s_plus = 2
q_higgs_neutral = 0
effective_yukawa_charge = q_s_minus + 2 * q_matter + q_higgs_neutral
assert effective_yukawa_charge == 0

assert q_s_minus + q_s_plus == 0
assert q_s_minus**3 + q_s_plus**3 == 0

reference_dimensions = {
    "27": 27,
    "27bar": 27,
    "351prime": 351,
    "351primebar": 351,
    "78": 78,
}
assert sum(reference_dimensions.values()) == 834

dim_e6 = 78
dim_sm = 8 + 3 + 1
assert dim_sm == 12
assert dim_e6 - dim_sm == 66

print("Higgs charge required by FFH:", q_yukawa_higgs)
print("Higgs charge required by H^3:", q_self_cubic_higgs)
print("Dressed effective Yukawa charge:", effective_yukawa_charge)
print("Reference Higgs representation dimension:", sum(reference_dimensions.values()))
print("Gauge-boson test: massless =", dim_sm, "; massive =", dim_e6 - dim_sm)

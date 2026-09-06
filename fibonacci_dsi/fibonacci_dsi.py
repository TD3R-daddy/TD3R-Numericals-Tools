from math import sqrt, log, pi

phi_F = (1.0 + sqrt(5.0)) / 2.0
omega_T = 2.0 * pi / log(phi_F)
ratio_m_over_H = sqrt(omega_T**2 + 9.0/4.0)

print(f"phi_F = {phi_F:.15f}")
print(f"Omega_T = 2*pi/ln(phi_F) = {omega_T:.12f}")
print(f"If Omega_Theta = Omega_T, m/H = sqrt(Omega_T^2 + 9/4) = {ratio_m_over_H:.12f}")

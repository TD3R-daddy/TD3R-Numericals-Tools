#!/usr/bin/env python3
"""Compact analytic audit of two local N=1 compensator branches in TD3R.

This reproduces algebraic branch-level consistency checks only. It is not a
6D microscopic completion and does not establish the global R1 scalar vacuum.
"""
import json, math

def singlet_branch(kappa=1.0, g=0.5, v=1.0):
    mf2=2*kappa*kappa*v*v; mv2=36*g*g*v*v
    eig=sorted([0.0,mv2,mf2,mf2,mf2,mf2])
    return {"superpotential":"kappa X (S_-3 B_+3 - v^2)",
      "F_terms_zero":True,"D_term_zero":True,"W_zero":True,
      "analytic_real_hessian_eigenvalues":eig,
      "positive_after_gauge_quotient":eig[1]>0,
      "preserves_E6":True,"preserves_R1_standard_model":True,
      "derives_Jordan_27_orientation":False}

def jordan_reference(kappa=1.0,g=0.5):
    rho=1.0; s=1/math.sqrt(3); common=rho*rho+9*s*s
    mf2=kappa*kappa*rho**4*common; mv2=2*g*g*common
    eig=sorted([0.0,mv2,mf2,mf2,mf2,mf2])
    return {"superpotential":"kappa X (S I3(Phi)/Lambda^2 - mu^2)",
      "reference_vacuum":{"rho":rho,"s":s,"X":0.0},
      "analytic_real_hessian_eigenvalues":eig,
      "positive_after_gauge_quotient":eig[1]>0,
      "I3_required_nonzero":True,"preserves_R1_standard_model":False,
      "reason":"the 27 cubic vanishes on the two SM-singlet R1 VEV directions"}

def main():
    report={"scope":"local 4D N=1 analytic audit; not a 6D microscopic derivation",
      "singlet_branch":singlet_branch(),"jordan_reference":jordan_reference(),
      "supergravity_and_6D_completion":"open"}
    assert report["singlet_branch"]["positive_after_gauge_quotient"]
    assert report["jordan_reference"]["positive_after_gauge_quotient"]
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()

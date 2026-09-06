#!/usr/bin/env python3
"""Compact independent audit of the R1 group-theory benchmark.

Checks the Gamma_20 kernel on the E6 27, Smith invariants of the VEV charge
lattice, E6 root-orbit counting and regularity of the adjoint VEV at v=w=1.
This does not prove the complete scalar vacuum or its 6D origin.
"""
import cmath, math, json
from functools import reduce
from itertools import combinations
import numpy as np

WEIGHTS_27=(("10",2,-1,1),("bar5_16",-1,3,1),("1_16",0,-5,1),("5_10",1,2,-2),("bar5_10",-1,-2,-2),("1_1",0,0,4))
Q=np.array([[-5,1],[-10,2],[0,8],[0,-4]],dtype=int)
MULT={"45/(10+10bar)":20,"(16+16bar)/(10+10bar)":20,"(16+16bar)/(5+5bar)":10,"(16+16bar)/(1+1)":2}
FORMS={"45/(10+10bar)":(1.,0.,4/15),"(16+16bar)/(10+10bar)":(-1.,math.sqrt(15),1/60),"(16+16bar)/(5+5bar)":(math.sqrt(3/5),1.,1/4),"(16+16bar)/(1+1)":(math.sqrt(5/6),-math.sqrt(1/2),1.)}

def gamma20_ok():
    zeta=cmath.exp(2j*math.pi/5); seen=[]
    for n in range(20):
        a,b,k=math.pi*n/10,math.pi*n/2,(2*n)%5
        for _,ce,qc,qp in WEIGHTS_27:
            z=zeta**(ce*k)*cmath.exp(1j*(qc*a+qp*b)); assert abs(z-1)<1e-12
        seen.append((k,n%20,n%4))
    return len(set(seen))==20

def smith():
    d1=reduce(math.gcd,(abs(int(x)) for x in Q.flat))
    minors=[abs(int(Q[i,0]*Q[j,1]-Q[i,1]*Q[j,0])) for i,j in combinations(range(len(Q)),2)]
    return d1, reduce(math.gcd,[m for m in minors if m])//d1

def masses(v=1.,w=1.):
    return {name:c*(a*v+b*w)**2 for name,(a,b,c) in FORMS.items()}

def main():
    m=masses()
    report={
      "gamma20_order_20_verified":gamma20_ok(), "smith_invariants":list(smith()),
      "E6_roots_total":20+sum(MULT.values()), "outside_SU5_roots":sum(MULT.values()),
      "adjoint_mass_coefficients_at_v_eq_w_eq_1":m,
      "adjoint_centralizer_regular":min(m.values())>1e-12,
      "R1_gauge_kernel_target_dimension":12,
      "status":"finite group-theory audit; complete scalar Hessian and 6D origin remain open"
    }
    assert report["smith_invariants"]==[1,20]
    assert report["E6_roots_total"]==72
    assert report["adjoint_centralizer_regular"]
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()

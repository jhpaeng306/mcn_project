"""Outward-rounded local frequency certificate, using only Decimal + NumPy.

All model decimals in calibration.json are interpreted exactly. This proves a
local statement for that specified decimal model, not for uncertain physiology.
Interval endpoints use directed rounding at 60 digits. Decimal.exp is correctly
rounded; expanding its rounded value by one neighboring Decimal is conservative.
"""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path
import json
import numpy as np

PREC = 60

def rounded(fn, rounding):
    with localcontext() as c:
        c.prec=PREC; c.rounding=rounding
        return +fn()

class I:
    def __init__(self, lo, hi=None):
        self.lo=D(str(lo)); self.hi=D(str(lo if hi is None else hi))
        assert self.lo<=self.hi
    def __add__(self, other):
        o=as_i(other)
        return I(rounded(lambda:self.lo+o.lo,ROUND_FLOOR),
                 rounded(lambda:self.hi+o.hi,ROUND_CEILING))
    __radd__=__add__
    def __neg__(self): return I(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self, other):return self+-as_i(other)
    def __rsub__(self, other):return as_i(other)+-self
    def __mul__(self, other):
        o=as_i(other)
        lower=[rounded(lambda a=a,b=b:a*b,ROUND_FLOOR)
               for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        upper=[rounded(lambda a=a,b=b:a*b,ROUND_CEILING)
               for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        return I(min(lower),max(upper))
    __rmul__=__mul__
    def __truediv__(self,other):
        o=as_i(other)
        assert not o.lo<=0<=o.hi
        inverse=I(rounded(lambda:1/o.hi,ROUND_FLOOR),
                  rounded(lambda:1/o.lo,ROUND_CEILING))
        return self*inverse
    def __rtruediv__(self,other):return as_i(other)/self
    def exp(self):
        with localcontext() as c:
            c.prec=PREC
            return I(self.lo.exp().next_minus(),self.hi.exp().next_plus())
    def absmax(self):return max(self.lo.copy_abs(),self.hi.copy_abs())
    def bounds(self):return [str(self.lo),str(self.hi)]
    def strictly_inside(self,outer):return outer.lo<self.lo and self.hi<outer.hi

def as_i(x):return x if isinstance(x,I) else I(x)

class J:
    """Interval value and its interval derivative with respect to q."""
    def __init__(self,v,d=0):self.v=as_i(v);self.d=as_i(d)
    def __add__(self,o):
        o=as_j(o);return J(self.v+o.v,self.d+o.d)
    __radd__=__add__
    def __neg__(self):return J(-self.v,-self.d)
    def __sub__(self,o):return self+-as_j(o)
    def __rsub__(self,o):return as_j(o)+-self
    def __mul__(self,o):
        o=as_j(o);return J(self.v*o.v,self.d*o.v+self.v*o.d)
    __rmul__=__mul__

def as_j(x):return x if isinstance(x,J) else J(x)

def coefficients(record,Q):
    K=len(record['G']);q=as_i(Q)
    slope=[]
    for g,theta in zip(record['gain'],record['theta']):
        g=I(g);s=1/(1+(-g*I(theta)).exp())
        # s is logistic(+g theta); s(1-s) is symmetric under sign reversal.
        slope.append(g*s*(1-s))
    m=[]
    for i in range(K):
        row=[]
        for j in range(K):
            variance=I(record['Sigma'][i][j]);tau=I(record['tau_ms'][i])
            recurrent=I('.75')*I(record['G'][i][j])*slope[j]*(-variance*q*q/2).exp()/tau
            row.append(J(recurrent-(1/tau if i==j else 0),-q*variance*recurrent))
        m.append(row)
    a,b,c=m[0][:3];d,e,f=m[1][:3];g,_,h=m[2][:3]
    A=-a-e-h; B=a*e-b*d+h*(a+e)-g*c
    C=-h*(a*e-b*d)+g*c*e-g*b*f
    if K==3:return [J(1),A,B,C]
    j=m[2][3];r,_,s,t=m[3]
    return [J(1),A-t,B-t*A-j*s,
            C-t*B+j*s*(a+e)-j*r*c,
            -t*C-j*s*(a*e-b*d)+j*r*c*e-j*r*b*f]

def cmul(z,w):return (z[0]*w[0]-z[1]*w[1],z[0]*w[1]+z[1]*w[0])
def poly(coeff,z):
    out=(I(0),I(0))
    for c in coeff:
        re,im=cmul(out,z);out=(re+c,im)
    return out
def derivative_coeff(c):return [(len(c)-1-i)*x for i,x in enumerate(c[:-1])]

def _certify(record):
    # Rational q interval contains q0=6*pi/47.6023 = 0.395979940497...
    Q=I('.39597','.39599');center=.39598
    cs=coefficients(record,I(str(center)))
    numerical=np.array([float((x.v.lo+x.v.hi)/2) for x in cs])
    roots=np.roots(numerical)
    lam=roots[roots.imag>0][0]
    x0=[I(str(lam.real)),I(str(lam.imag))]
    X=[x0[0]+I('-.0001','.0001'),x0[1]+I('-.0001','.0001')]
    c=coefficients(record,Q); cv=[v.v for v in c];cq=[v.d for v in c]
    F=poly(cv,x0); U,V=poly(derivative_coeff(cv),X)
    jac=[[U,-V],[V,U]]
    dp=np.polyval(np.polyder(numerical),lam)
    inv=np.linalg.inv([[dp.real,-dp.imag],[dp.imag,dp.real]])
    C=[[I(str(x)) for x in row] for row in inv]
    det=C[0][0]*C[1][1]-C[0][1]*C[1][0]
    assert not det.lo<=0<=det.hi
    E=[[I(int(i==j))-sum(C[i][k]*jac[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    Kbox=[x0[i]-sum(C[i][j]*F[j] for j in range(2))+
          sum(E[i][j]*(X[j]-x0[j]) for j in range(2)) for i in range(2)]
    assert all(k.strictly_inside(x) for k,x in zip(Kbox,X)),(Kbox[0].bounds(),Kbox[1].bounds())
    contraction=max(rounded(lambda:sum(e.absmax() for e in row),ROUND_CEILING) for row in E)
    assert contraction<1
    real_boxes=[]
    for r in roots[abs(roots.imag)<1e-10].real:
        r0=I(str(r));box=r0+I('-.0001','.0001')
        inverse=I(str(1/np.polyval(np.polyder(numerical),r)))
        F0=poly(cv,(r0,I(0)))[0]
        jacr=poly(derivative_coeff(cv),(box,I(0)))[0]
        remainder=1-inverse*jacr
        enclosure=r0-inverse*F0+remainder*(box-r0)
        assert enclosure.strictly_inside(box)
        assert remainder.absmax()<1
        assert box.hi<X[0].lo
        real_boxes.append(box.bounds())
    assert len(real_boxes)==len(record['G'])-2
    if len(real_boxes)==2:
        assert D(real_boxes[0][1])<D(real_boxes[1][0]) or D(real_boxes[1][1])<D(real_boxes[0][0])
    # All K roots are accounted for in disjoint pair/real-root boxes.
    u,v=poly(derivative_coeff(cv),X);cc,dd=poly(cq,X)
    slope=(cc*v-dd*u)/(u*u+v*v)
    assert slope.lo>0
    return {'q_interval':Q.bounds(),'alpha_box':X[0].bounds(),'omega_box':X[1].bounds(),
        'newton_image':[k.bounds() for k in Kbox],
        'contraction_bound':str(contraction),'other_real_root_boxes':real_boxes,
        'positive_omega_derivative_interval':slope.bounds(),
        'precision_decimal_digits':PREC,'model_interpretation':'exact saved decimal constants'}

def certify(record):
    with localcontext() as context:
        context.prec=PREC
        return _certify(record)

if __name__=='__main__':
    # Keep Decimal unary negation and auxiliary scalar operations at >=PREC too.
    from decimal import getcontext
    getcontext().prec=PREC
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'simulations/output/v1_extension/calibration.json').read_text(),parse_float=str)
    result={str(K):certify(data[str(K)]) for K in (3,4)}
    path=root/'simulations/output/v1_extension/local_monotonicity_certificate.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

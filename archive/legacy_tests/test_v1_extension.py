"""Scientific invariants for the new V1 extension, independent of plot layout."""
import json
from pathlib import Path
import numpy as np
import pytest
from simulations.v1_extension import (make_model, LEGACY_G, linear_symbol,
    dominant, eigen_derivative, analyze_model, integrate_cases, forcing_cases)


@pytest.fixture(params=[3, 4])
def model(request):
    data=json.loads((Path(__file__).parents[1]/
        'simulations/output/v1_extension/calibration.json').read_text())
    return make_model(request.param,data[str(request.param)]['G'])


def test_preserved_mask_and_calibrated_onset(model):
    assert np.array_equal(np.sign(model.A_p), np.sign(LEGACY_G[:model.K,:model.K]))
    critical,normal,_,_,tail=analyze_model(model)
    assert abs(critical.p0-.25)<1e-9
    assert critical.mode==3
    assert abs(critical.omega0-.08)<1e-9
    assert normal.b.real<0 and (normal.b+normal.c).real<0
    assert tail.max()<1
    # 2D shells include sqrt(m^2+n^2), not merely integer 1D radii.
    radii2=np.unique(np.add.outer(np.arange(65)**2,np.arange(65)**2))
    for radius2 in radii2:
        q=2*np.pi*np.sqrt(radius2)/model.length
        if q>=8: continue  # the verified decreasing Gaussian tail covers these
        growth=dominant(linear_symbol(model,q,critical.p0)).real
        if radius2==9:
            assert abs(growth)<1e-9
        else:
            assert growth<0


def test_eigen_derivative_and_large_q(model):
    for q in [.4,.8,1.2]:
        h=1e-6
        finite=(dominant(linear_symbol(model,q+h,.25))-
                dominant(linear_symbol(model,q-h,.25)))/(2*h)
        assert abs(eigen_derivative(model,q,.25)-finite)<1e-7
    np.testing.assert_allclose(linear_symbol(model,20,.25),
                                -np.diag(1/model.tau),atol=1e-12)


def test_sparse_characteristic_polynomial(model):
    for q in [0,.4,1.1,4]:
        M=linear_symbol(model,q,.25)
        a,b,c=M[0,:3];d,e,f=M[1,:3];g,_,h=M[2,:3]
        for z in [.13+.07j,-.05+.2j]:
            H=(z-a)*(z-e)-b*d
            R=c*(z-e)+b*f
            P3=(z-h)*H-g*R
            if model.K==3: expected=P3
            else:
                j=M[2,3];r,_,s,t=M[3]
                expected=(z-t)*P3-j*(s*H+r*R)
            assert abs(np.linalg.det(z*np.eye(model.K)-M)-expected)<1e-14


def test_axis_invariance_and_1d_lift(model):
    critical,normal,*_=analyze_model(model)
    p,forcing=forcing_cases(model,normal)
    cases={'axis': {'traveling':forcing['temporal']['temporal']}}
    one=integrate_cases(model,critical,cases,p=p,dim=1,n=32,duration=50,dt=.5)
    two=integrate_cases(model,critical,cases,p=p,dim=2,n=32,duration=50,dt=.5,
                        exact_axis=('axis',))
    assert two['transverse_rms'].max()<1e-13
    np.testing.assert_allclose(two['final'][...,0],one['final'],atol=1e-12)


def test_time_step_convergence(model):
    critical,normal,*_=analyze_model(model)
    p,cases=forcing_cases(model,normal)
    coarse=integrate_cases(model,critical,cases,p=p,n=64,duration=1000,dt=1)
    fine=integrate_cases(model,critical,cases,p=p,n=64,duration=1000,dt=.5)
    rel=np.linalg.norm(coarse['final']-fine['final'])/np.linalg.norm(fine['final'])
    assert rel<2e-4


def test_interval_polynomial_and_local_increase(model):
    from decimal import Decimal, localcontext
    from simulations.v1_interval_certificate import I, coefficients, certify
    path=Path(__file__).parents[1]/'simulations/output/v1_extension/calibration.json'
    record=json.loads(path.read_text(),parse_float=str)[str(model.K)]
    # Compare independently expanded interval coefficients to the matrix's
    # characteristic polynomial, at several noncritical wavenumbers too.
    for q in ['0', '.39598', '1.1']:
        values=coefficients(record,I(q))
        computed=np.array([float((c.v.lo+c.v.hi)/2) for c in values])
        expected=np.poly(linear_symbol(model,float(q),.25))
        np.testing.assert_allclose(computed,expected,rtol=1e-12,atol=1e-15)
    proof=certify(record)  # Root inclusion, contraction and dominance asserted inside.
    lo,hi=map(float,proof['positive_omega_derivative_interval'])
    assert 0<lo<hi
    for q in np.linspace(.39597,.39599,7):
        assert lo<eigen_derivative(model,q,.25).imag<hi
    # A directed division must enclose an independently calculated 100-digit value.
    with localcontext() as context:
        context.prec=100
        exact=Decimal(1)/Decimal(7)
    interval=I(1)/7
    assert interval.lo<exact<interval.hi

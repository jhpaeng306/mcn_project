import numpy as np
import pytest
from simulations.v1_simple import make_model,coefficients,audit,integrate
from simulations.fft import linear_symbol

@pytest.fixture(params=[3,4])
def setting(request):
    model=make_model(request.param)
    return model,*coefficients(model)

def test_onset_and_stable_amplitude_patterns(setting):
    m,c,n,e=setting
    result=audit(m,c,n,e)
    assert result['square_spectral_gap']>0
    assert abs(np.vdot(c.psi,c.phi)-1)<1e-12
    assert abs(np.linalg.norm(c.phi)-1)<1e-12
    assert np.all(m.A_p==np.round(m.A_p))

def test_frequency_slope_matches_finite_difference(setting):
    m,c,n,e=setting;dq=1e-5
    def branch(q):
        v=np.linalg.eigvals(linear_symbol(m,q,c.p0))
        return v[np.argmin(abs(v-1j*c.omega0))]
    finite=(branch(c.q+dq)-branch(c.q-dq))/(2*dq)
    assert abs(finite-e['nu_q'])<1e-8

def test_traveling_axis_invariance(setting):
    m,c,n,e=setting;eps=.02;xx=2*np.pi*c.mode*np.arange(16)/16
    u=.0002*np.real(c.phi[:,None])*np.cos(xx)
    a=integrate(m,c,{'traveling':{'traveling':8}},n=16,initial=u[None],slow_duration=.02,dt=.5)
    b=integrate(m,c,{'traveling':{'traveling':8}},dim=2,n=16,initial=np.broadcast_to(u[None,:,:,None],(1,m.K,16,16)),slow_duration=.02,dt=.5)
    assert b['transverse_rms'].max()<1e-13
    np.testing.assert_allclose(a['final'],b['final'][...,0],atol=1e-12)

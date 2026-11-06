"""Checks for the current notebooks' batched field solver and spectral selection."""
import numpy as np
from simulations.main_analysis import integrate, tracked_spectrum, specification
from simulations.v1_literature import make_model, coefficients, audit, S_UM, G
from simulations.v1_simple import integrate as reference_integrate
from simulations.pipeline_2d import compute_bifurcation_2d


def test_batched_solver_agrees_with_independent_existing_fft_solver():
    model=make_model(4);cr,_,_=coefficients(model)
    cases={'none':{},'temporal':{'temporal':8}}
    actual=integrate(model,cr,cases,n=16,duration=100.,dt=.5)
    reference=reference_integrate(model,cr,cases,n=16,initial=actual['initial'],slow_duration=.04,dt=.5)
    np.testing.assert_allclose(actual['final'],reference['final'],atol=2e-14)


def test_rk4_refinement_both_boundaries():
    model=make_model(4)
    for boundary in ['periodic','reflect']:
        geo=compute_bifurcation_2d(model,boundary=boundary,p_bounds=(0,1),scan_points=31)
        kw=dict(dim=2,boundary=boundary,wavevector=geo.wavevector,n=24,duration=500.,save_ms=5.)
        cases={'temporal':{'temporal':8},'spatial':{'spatial':8}}
        coarse=integrate(model,geo.point,cases,dt=1.,**kw)
        fine=integrate(model,geo.point,cases,dt=.5,**kw)
        assert np.linalg.norm(coarse['final']-fine['final'])/np.linalg.norm(fine['final'])<2e-4
        # Reflecting cosine coefficients and periodic coefficients use their
        # respective reconstruction normalization (rather than raw DCT magnitudes).
        mode0=coarse['modes'][0,:,0,1]
        expected=.02*.02*geo.point.phi[0].real
        if boundary=='reflect': assert abs(mode0[0]-expected)<2e-6
        else: assert abs(abs(mode0[0])-abs(expected)/2)<2e-6


def test_rightmost_selection_and_cache_parameters():
    model=make_model(4);cr,_,_=coefficients(model)
    qs=np.linspace(0,4,401);branches,ids=tracked_spectrum(model,cr.p0,qs)
    selected=branches[np.arange(len(qs)),ids]
    np.testing.assert_allclose(selected.real,branches.real.max(axis=1),atol=1e-11)
    assert selected.imag.min()>-1e-11
    assert selected[0].imag>0 and selected[-1].imag==0
    args=(model,cr,{'temporal':{'temporal':8}},1,'periodic',(2,),64,1.,150000.,5.)
    old=specification(*args)
    model.Sigma[0,0]*=1.01
    assert old!=specification(*args)


def test_restored_coefficients_frequency_and_stability():
    from simulations.pipeline import v1_epvsst_vip_preset
    from simulations.fft import linear_symbol, compute_bifurcation
    from simulations.v1_literature import LENGTH
    old,_=v1_epvsst_vip_preset();m=make_model();cr,n,e=coefficients(m)
    np.testing.assert_array_equal(m.A_p,old.A_p)
    np.testing.assert_array_equal(m.Sigma,old.Sigma)
    np.testing.assert_array_equal(m.tau,old.tau)
    np.testing.assert_array_equal(m.gain,old.gain)
    np.testing.assert_array_equal(m.theta,old.theta)
    assert LENGTH==43 and cr.mode==3
    assert audit(m,cr,n,e)['square_spectral_gap']>0
    h=1e-5
    def branch(q):
        v=np.linalg.eigvals(linear_symbol(m,q,cr.p0))
        return v[np.argmin(abs(v-1j*cr.omega0))]
    derivative=(branch(cr.q+h)-branch(cr.q-h))/(2*h)
    assert abs(derivative-e['nu_q'])<1e-8 and derivative.imag>0
    assert compute_bifurcation(make_model(3,submatrix=True),p_bounds=(0,1)).mode==0


def test_three_population_variant_finite_wave_onset():
    from simulations.main_analysis import model_audit
    m=make_model(3);block=make_model(3,submatrix=True).A_p
    changed=np.argwhere(m.A_p!=block)
    assert changed.tolist()==[[0,0]] and np.all(np.sign(m.A_p)==np.sign(block))
    assert abs(m.A_p[0,0]/block[0,0]-1)<.1
    np.testing.assert_array_equal(m.Sigma,make_model(4).Sigma[:3,:3])
    cr,n,e=coefficients(m)
    assert cr.mode==3 and abs(cr.p0-.588034166)<1e-8 and e['nu_q'].imag>0
    checks=model_audit(3,m,cr,n,e)
    assert checks['spatial_branch_sign']==-1 and checks['square_spectral_gap']>0


def test_four_population_cases_unchanged():
    from simulations.main_analysis import forcing_cases, specification, same_spec
    base,controls=forcing_cases(4)
    assert base=={'none':{},'temporal':{'temporal':3},'spatial':{'spatial':8},'spatial + temporal':{'temporal':3,'spatial':2}}
    assert controls=={'weak temporal':{'temporal':1},'detuned temporal':{'temporal':3,'omega_factor':1.5},
        'detuned spatial':{'spatial':8,'space_factor':3},'detuned combined':{'temporal':3,'spatial':2,'omega_factor':1.5,'space_factor':3}}
    m=make_model(4);cr,_,_=coefficients(m)
    spec=specification(m,cr,base,1,'periodic',(3,),64,1.,150000.,5.)
    assert 'seed_shift' not in spec
    perturbed=dict(spec,kappa_c=spec['kappa_c']*(1+1e-15))
    assert same_spec(spec,perturbed) and not same_spec(spec,dict(spec,kappa_c=spec['kappa_c']*(1+1e-9)))


def test_e_only_drive_is_explicit():
    m=make_model();cr,_,_=coefficients(m)
    r=integrate(m,cr,{'E only':{'temporal':3,'drive':[1,0,0,0]}},n=16,duration=10.)
    np.testing.assert_array_equal(r['drive'],[[1,0,0,0]])

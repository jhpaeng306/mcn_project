"""Restored V1 parameterization: literature widths/motif, fixed effective gains.

No optimization toward a reference matrix. G, tau and activation parameters
are the previous effective model choices, not experimentally identified gains.
Billeh 2020 Methods e10-e11 specify exp(-r²/s²), hence Sigma=s²/2.
"""
import numpy as np
from scipy.linalg import eig
from .simulation import ModelParameters
from .fft import linear_symbol, compute_bifurcation, compute_normal_form, compute_forcing_coefficients

S_UM=np.array([[114.,95.,85.,85.],[92.,120.,120.,120.],
               [103.,120.,120.,120.],[103.,120.,120.,120.]])
G=np.array([[6.22009,-1.45455,-6.48944,0.],
            [5.09798,-.90615,-8.04066,0.],
            [2.67174,0.,0.,-16.],
            [1.68844,0.,-3.13584,0.]])
LENGTH=43.
TAU=np.array([10.,12.,35.,20.])
GAIN=np.array([3.47682761,7.34413013,3.74650527,6.])
THETA=np.array([.28483799,.12068148,.40137356,.7])


# Three-population model: principal E/PV/SST block of G with E->E strengthened
# by 8.04% (6.22009 -> 6.72). The unchanged block has a homogeneous (k=0) first
# Hopf onset; E->E is the only single entry whose small same-sign change moves the
# first crossing to a finite wave number (k0=3 on L=43, as for four populations).
# Widths, signs and zeros are unchanged. Use make_model(3,submatrix=True) for the
# unmodified block.
G3=G[:3,:3].copy();G3[0,0]=6.72


def make_model(K=4,*,length=LENGTH,submatrix=False):
    if K not in (3,4):raise ValueError('K must be 3 or 4')
    gains=G3.copy() if K==3 and not submatrix else G[:K,:K].copy()
    return ModelParameters(K=K,A=np.zeros((K,K)),A_p=gains,
        Sigma=(S_UM[:K,:K]/100)**2/2,tau=TAU[:K],gain=GAIN[:K],theta=THETA[:K],
        length=length,names=('E','PV','SST','VIP')[:K],time_unit='ms',space_unit='0.1 mm',
        metadata={'source':'Billeh 2020 widths; Pfeffer 2013 reduced motif',
          'gain_status':'fixed effective model coefficients; not inferred synaptic strengths',
          'drive_weights':[1.,1.,0.,0.][:K],
          'gain_modification':None if K==4 or submatrix else 'E->E 6.22009 -> 6.72 (+8.04%)'})


def coefficients(model,drive=None):
    cr=compute_bifurcation(model,p_bounds=(0,1),max_mode=64,scan_points=101)
    if cr.mode==0:raise ValueError('This gain matrix has a homogeneous first onset; the finite-wave reduction is not applicable.')
    nf=compute_normal_form(model,cr)
    q=cr.q;omega=cr.omega0;phi=cr.phi;psi=cr.psi
    def Q(k,v,w):return (model.connectivity(cr.p0)*np.exp(-model.Sigma*k*k/2))@(nf.f2*v*w/2)/model.tau
    def C(k,u,v,w):return (model.connectivity(cr.p0)*np.exp(-model.Sigma*k*k/2))@(nf.f3*u*v*w/6)/model.tau
    def R(o,k,v):return np.linalg.solve(1j*o*np.eye(model.K)-linear_symbol(model,k,cr.p0),v)
    cp=np.vdot(psi,2*Q(q,phi,nf.h11_zero)
       +4*Q(q,phi.conj(),R(2*omega,np.sqrt(2)*q,Q(np.sqrt(2)*q,phi,phi)))
       +4*Q(q,phi,R(0,np.sqrt(2)*q,Q(np.sqrt(2)*q,phi,phi.conj())))
       +6*C(q,phi,phi,phi.conj()))
    drive=np.asarray(model.metadata['drive_weights'] if drive is None else drive)/2
    force=compute_forcing_coefficients(model,nf,temporal_harmonic=drive,spatial_harmonic=drive)
    ew=2*np.vdot(psi,Q(q,phi.conj(),R(2*omega,2*q,drive/model.tau)))
    vals,left,right=eig(linear_symbol(model,q,cr.p0),left=True,right=True)
    j=np.argmin(abs(vals-1j*omega))
    Mq=-q*(model.Sigma*model.connectivity(cr.p0)*np.exp(-model.Sigma*q*q/2)*nf.f1[None,:])/model.tau[:,None]
    slope=np.vdot(left[:,j],Mq@right[:,j])/np.vdot(left[:,j],right[:,j])
    return cr,nf,dict(c_perp=cp,eta_t_unit=force.eta_t,eta_x_unit=force.eta_x,eta_w_unit=ew,nu_q=slope)


def spatial_sign(e):
    """Standing branch A=sigma*B destabilized by real spatial input: sigma=sign Re eta_x.

    On A=sigma*B the reduced equation is z'=z(a+sigma*eta_x+(b+c)|z|^2)+sigma*eta_t*conj(z).
    sigma=+1 (four populations) is the report's A=B case; sigma=-1 is the same
    standing pattern shifted by a quarter wavelength relative to the input.
    """
    return 1. if e['eta_x_unit'].real>0 else -1.


def audit(model,cr,nf,e,*,temporal=3.,spatial=8.,combined=(3.,2.),traveling=2.,lattice='square',require_spatial=True):
    """Spectral and cubic-order checks for the selected forcing peaks (units of eps^2).

    lattice='square' checks every square-torus shell |k|^2=m^2+n^2 (2D notebooks);
    lattice='line' checks only the 1D torus modes k^2. require_spatial=False records,
    instead of asserting, whether the spatial peak exceeds its leading threshold.
    """
    radii=(np.unique(np.add.outer(np.arange(80)**2,np.arange(80)**2)) if lattice=='square'
           else np.arange(80)**2);qs=2*np.pi*np.sqrt(radii)/model.length;keep=qs<8
    def spectrum(kappa):
        mats=(-np.eye(model.K)+model.connectivity(kappa)[None]*np.exp(-model.Sigma[None]*qs[keep,None,None]**2/2)*nf.f1[None,None,:])/model.tau[None,:,None]
        return np.linalg.eigvals(mats).real.max(axis=1)
    gap=-spectrum(cr.p0)[radii[keep]!=cr.mode**2].max()
    tail=np.max(np.sum(abs(model.connectivity(cr.p0))*np.exp(-model.Sigma*8**2/2)*nf.f1[None,:],axis=1))
    assert gap>0 and tail<1
    assert nf.b.real<0 and nf.c.real<0 and e['c_perp'].real<0 and e['nu_q'].imag>0
    for kappa in np.linspace(0,cr.p0,61)[:-1]:assert spectrum(kappa).max()<0
    def root(a,g,eta):
        rr=np.roots([abs(g)**2,2*(a*np.conj(g)).real,abs(a)**2-abs(eta)**2])
        assert np.max(abs(rr.imag))<1e-10
        return float(max(rr.real))
    sigma=spatial_sign(e);ex=sigma*e['eta_x_unit']
    temporal_rho=root(-nf.alpha,nf.b+nf.c,temporal*e['eta_t_unit'])
    dt,dx=combined
    ap=-nf.alpha+dx*ex;rho=root(ap,nf.b+nf.c,dt*e['eta_t_unit'])
    combined_conditions=[(ap+2*(nf.b+nf.c)*rho).real,
        (-nf.alpha-dx*ex+2*nf.b*rho).real,
        ((-nf.alpha+2*nf.b*rho)*np.conj(dx*ex)).real]
    rw=root(-nf.alpha,nf.b,traveling*e['eta_w_unit'])
    travel=[(-nf.alpha+2*nf.b*rw).real,(-nf.alpha+nf.c*rw).real,(-nf.alpha+e['c_perp']*rw).real]
    assert temporal*abs(e['eta_t_unit'])>abs(nf.alpha)
    assert max(combined_conditions)<0 and max(travel)<0
    spatial_above=bool((spatial*ex-nf.alpha).real>0)
    if require_spatial:assert spatial_above
    return dict(square_spectral_gap=gap,tail_row_sum_bound=tail,
        temporal_threshold=float(abs(nf.alpha)/abs(e['eta_t_unit'])),
        traveling_threshold=float(abs(nf.alpha)/abs(e['eta_w_unit'])),
        spatial_threshold=float(nf.alpha.real/abs(e['eta_x_unit'].real)),
        spatial_branch_sign=sigma,spatial_above_threshold=spatial_above,lattice=lattice,
        temporal_rho=temporal_rho,combined_rho=rho,travel_rho=rw,
        combined_stability_conditions=combined_conditions,travel_stability_conditions=travel)

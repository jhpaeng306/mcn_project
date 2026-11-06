"""Reproducible V1 coefficient audit and 1D/square-torus experiments.

New models are explicitly phenomenological calibrations, not fitted physiology.
The original notebooks and presets are left untouched. Time is ms, space is
0.1 mm, and all matrices use target rows / source columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq, minimize
from scipy.special import expit

from .simulation import ModelParameters, activation_derivatives
from .fft import (linear_symbol, compute_bifurcation, compute_normal_form,
                  compute_forcing_coefficients)
from .amplitude import zero_state_growth_rate

LEGACY_G = np.array([[6.22009, -1.45455, -6.48944, 0.],
                     [5.09798, -.90615, -8.04066, 0.],
                     [2.67174, 0., 0., -16.],
                     [1.68844, 0., -3.13584, 0.]])
S_UM = np.array([[114., 95., 85., 85.], [92., 120., 120., 120.],
                 [103., 120., 120., 120.], [103., 120., 120., 120.]])
TAU = np.array([10., 12., 35., 20.])
THETA = np.array([.28483799, .12068148, .40137356, .7])
GAIN = np.array([3.47682761, 7.34413013, 3.74650527, 6.])
LENGTH = 47.6023
P0 = .25
Q0 = 2*np.pi*3/LENGTH


def make_model(K=4, G=None):
    if K not in (3, 4):
        raise ValueError("Only E/PV/SST and E/PV/SST/VIP are supported")
    G = LEGACY_G[:K, :K].copy() if G is None else np.array(G)
    return ModelParameters(K=K, A=.5*G, A_p=G,
        Sigma=(S_UM[:K, :K]/100)**2/2, tau=TAU[:K],
        theta=THETA[:K], gain=GAIN[:K], length=LENGTH,
        names=("E", "PV", "SST", "VIP")[:K], time_unit="ms",
        space_unit="0.1 mm", metadata={"status": "phenomenological calibration"})


def dominant(M, tol=1e-10):
    """Rightmost eigenvalue, positive imaginary representative at a tie."""
    v = np.linalg.eigvals(M)
    tied = v[np.abs(v.real-v.real.max()) <= tol]
    return tied[np.argmax(tied.imag)]


def eigen_derivative(model, q, p):
    """Simple-branch derivative l* M'(q) r / (l* r)."""
    M = linear_symbol(model, q, p)
    values, left, right = eig(M, left=True, right=True)
    target = dominant(M)
    j = np.argmin(abs(values-target))
    slope = activation_derivatives(model)[0]
    Mp = (-q*model.Sigma*model.connectivity(p)*
          np.exp(-model.Sigma*q*q/2)*slope[None, :])/model.tau[:, None]
    return np.vdot(left[:, j], Mp@right[:, j])/np.vdot(left[:, j], right[:, j])


def calibrate(K):
    """Minimum squared log-change from legacy gains subject to four targets.

    Re lambda(q0)=0; d_q Re lambda(q0)=0; Im lambda(q0)=.08;
    max Re lambda(0)=-.001, at p=.25. Bounds exp(-2)..exp(2).
    Starting point is z=0. This is a local constrained optimum, not a claim
    of uniqueness or a statistical fit. Zeros and Dale signs are exact.
    """
    base = LEGACY_G[:K, :K]
    mask = base != 0
    def model(z):
        G = base.copy()
        G[mask] *= np.exp(z)
        return make_model(K, G)
    def residual(z):
        m = model(z)
        v = dominant(linear_symbol(m, Q0, P0))
        # Central difference is recorded as part of the calibration algorithm.
        h = 1e-4
        dq = (dominant(linear_symbol(m, Q0+h, P0)).real-
              dominant(linear_symbol(m, Q0-h, P0)).real)/(2*h)
        return 100*np.array([v.real, dq, v.imag-.08,
                             dominant(linear_symbol(m, 0, P0)).real+.001])
    opt = minimize(lambda z: np.dot(z, z), np.zeros(mask.sum()),
        method="SLSQP", bounds=[(-2, 2)]*mask.sum(),
        constraints=[{"type": "eq", "fun": residual}],
        options={"maxiter": 500, "ftol": 1e-12})
    if not opt.success or max(abs(residual(opt.x))) > 1e-6:
        raise RuntimeError(f"Calibration failed: {opt.message}")
    m = model(opt.x)
    assert np.array_equal(np.sign(m.A_p), np.sign(base))
    return m, {"objective": float(opt.fun), "log_multipliers": opt.x.tolist(),
               "scaled_constraint_residual": residual(opt.x).tolist(),
               "G": m.A_p.tolist(), "tau_ms": m.tau.tolist(),
               "Sigma": m.Sigma.tolist(), "gain": m.gain.tolist(),
               "theta": m.theta.tolist()}


def analyze_model(model):
    critical = compute_bifurcation(model, p_bounds=(0, .5), max_mode=64,
                                  scan_points=101)
    normal = compute_normal_form(model, critical)
    q = np.linspace(0, 8, 1601)
    ev = np.array([dominant(linear_symbol(model, x, critical.p0)) for x in q])
    # Gershgorin: beyond q=8 the absolute recurrent row sums only decrease.
    slope = activation_derivatives(model)[0]
    row_sum = np.sum(abs(model.connectivity(.5))*slope[None, :]*
                     np.exp(-model.Sigma*8**2/2), axis=1)
    assert row_sum.max() < 1
    assert critical.mode == 3 and abs(critical.omega0-.08) < 1e-7
    assert normal.supercritical
    return critical, normal, q, ev, row_sum


def forcing_cases(model, normal, margin=.0005, target_growth=.0001):
    """Equal E+PV input; independently calibrate each case's critical growth.

    A real cosine of peak d has complex coefficient d/2. Both-case d is
    each component's peak. All physical input amplitudes are reported.
    """
    drive = np.zeros(model.K); drive[:2] = 1
    p = normal.critical.p0-margin
    out = {"none": {"temporal": 0., "spatial": 0.}}
    for name in ("temporal", "spatial", "both"):
        def coeff(d):
            return compute_forcing_coefficients(model, normal,
                temporal_harmonic=drive*d/2 if name in ("temporal", "both") else None,
                spatial_harmonic=drive*d/2 if name in ("spatial", "both") else None)
        def fun(d):
            return zero_state_growth_rate(normal, p=p, forcing=coeff(d))-target_growth
        hi = .01
        while fun(hi) < 0 and hi < 1:
            hi *= 2
        d = brentq(fun, 0, hi)
        out[name] = {"temporal": d if name in ("temporal", "both") else 0.,
                     "spatial": d if name in ("spatial", "both") else 0.}
    for name, c in out.items():
        fc = compute_forcing_coefficients(model, normal,
             temporal_harmonic=drive*c['temporal']/2,
             spatial_harmonic=drive*c['spatial']/2)
        c['predicted_growth_per_ms'] = zero_state_growth_rate(normal, p=p, forcing=fc)
    return p, out


def integrate_cases(model, critical, cases, *, p, dim=1, n=64,
                    duration=80000., dt=1., save_ms=10., seed=2026,
                    exact_axis=(), initial_scale=3e-4, initial_state=None):
    """Batched nonlinear periodic RK4 solver; no amplitude equations in RHS.

    Axes: case, population, x[, y]. 2D cases start from the same x-dependent
    field. exact_axis has no y perturbation; other 2D cases receive the same
    small seeded transverse Fourier wave. x-direction forcing is always
    independent of y. Saved snapshots are decimated, modal diagnostics are
    retained every save_ms. Seed controls only the initial field.
    """
    if dim not in (1, 2) or n <= 4*critical.mode:
        raise ValueError("Use dim 1/2 and resolve the forced harmonic")
    labels = list(cases); C = len(labels); K = model.K
    shape = (n,)*dim; axes = tuple(range(2, 2+dim))
    q = 2*np.pi*np.fft.fftfreq(n, d=model.length/n)
    q2 = q*q if dim == 1 else q[:, None]**2+q[None, :]**2
    ext = (None,)*dim
    coupling = model.connectivity(p)
    symbol = coupling[(slice(None), slice(None))+ext]*np.exp(
        -.5*model.Sigma[(slice(None), slice(None))+ext]*q2)
    flat_symbol = symbol.reshape(K, K, -1)
    tau = model.tau.reshape((1, K)+(1,)*dim)
    gain = model.gain.reshape((1, K)+(1,)*dim)
    theta = model.theta.reshape((1, K)+(1,)*dim)
    offset = expit(-gain*theta)
    rng = np.random.default_rng(seed)
    base = initial_scale*rng.standard_normal((K, n))
    if dim == 2:
        base = np.broadcast_to(base[:, :, None], (K, n, n)).copy()
    state = np.repeat(base[None], C, axis=0)
    if dim == 2:
        phase_y = 2*np.pi*critical.mode*np.arange(n)/n
        transverse = initial_scale*np.real(critical.phi[:, None, None]*
                                           np.exp(1j*phase_y)[None, None, :])
        for c, label in enumerate(labels):
            if label not in exact_axis:
                state[c] += transverse
    if initial_state is not None:
        if np.shape(initial_state) != state.shape:
            raise ValueError('initial_state must match (case,population,x[,y])')
        state=np.array(initial_state,dtype=float,copy=True)
    initial=state.copy()
    drive = np.zeros(K); drive[:2] = 1
    amps_t = np.array([cases[k].get('temporal', 0.) for k in labels])
    amps_s = np.array([cases[k].get('spatial', 0.) for k in labels])
    amps_w = np.array([cases[k].get('traveling', 0.) for k in labels])
    tdrive = (amps_t[:, None]*drive[None, :]).reshape((C,K)+(1,)*dim)
    sdrive = (amps_s[:, None]*drive[None, :]).reshape((C,K)+(1,)*dim)
    wdrive = (amps_w[:, None]*drive[None, :]).reshape((C,K)+(1,)*dim)
    phase_x = 2*np.pi*(2*critical.mode)*np.arange(n)/n
    if dim == 2:
        phase_x = phase_x[:, None]
    spatial = np.cos(phase_x)
    omega = 2*critical.omega0
    def rhs(u, t):
        rate = expit(gain*(u-theta))-offset
        rh = np.fft.fftn(rate, axes=axes).reshape(C,K,-1)
        recurrent = np.fft.ifftn(np.einsum('ijz,cjz->ciz', flat_symbol, rh)
                               .reshape((C,K)+shape), axes=axes).real
        force = tdrive*np.cos(omega*t)+sdrive*spatial+wdrive*np.cos(phase_x-omega*t)
        return (-u+recurrent+force)/tau
    steps = int(round(duration/dt)); every = max(1, int(round(save_ms/dt)))
    times=[]; modes=[]; transverse_rms=[]; snapshots=[]; snapshot_t=[]
    k=critical.mode
    for step in range(steps+1):
        t=step*dt
        if step % every == 0:
            uh = np.fft.fftn(state, axes=axes)/n**dim
            if dim == 1:
                coeffs = np.stack([uh[:,:,0], uh[:,:,k], uh[:,:,2*k]], axis=-1)
                perp=np.zeros(C)
            else:
                coeffs=np.stack([uh[:,:,0,0], uh[:,:,k,0], uh[:,:,0,k],
                                 uh[:,:,k,k], uh[:,:,2*k,0]], axis=-1)
                perp=np.sqrt(np.mean((state-state.mean(axis=-1,keepdims=True))**2,
                                     axis=(1,2,3)))
            times.append(t);modes.append(coeffs);transverse_rms.append(perp)
            if dim == 1 or step % max(every, steps//100//every*every) == 0:
                snapshots.append(state[:,0].copy());snapshot_t.append(t)
        if step == steps:break
        a=rhs(state,t);b=rhs(state+dt*a/2,t+dt/2)
        c=rhs(state+dt*b/2,t+dt/2);d=rhs(state+dt*c,t+dt)
        state += dt*(a+2*b+2*c+d)/6
    if not np.isfinite(state).all():raise FloatingPointError('Nonfinite field')
    return dict(t=np.array(times), modes=np.array(modes),
        transverse_rms=np.array(transverse_rms), snapshots=np.array(snapshots),
        snapshot_t=np.array(snapshot_t), initial=initial, final=state, labels=labels,
        dt=dt, n=n, duration=duration, p=p, dim=dim,
        G=model.A_p.copy(), tau=model.tau.copy(), Sigma=model.Sigma.copy(),
        gain=model.gain.copy(), theta=model.theta.copy())


def measure_1d(result, critical):
    # Projection separates oscillatory critical activity from copied k=6 input.
    coeff = result['modes'][:,:,:,1]
    projected = np.einsum('j,tcj->tc', np.conj(critical.psi), coeff)
    t=result['t'];tail=t>t[-1]*.8
    metrics={}
    for i,name in enumerate(result['labels']):
        amp=np.abs(projected[:,i])
        # Fit a finite early-time window, after rapid stable-mode transients.
        fit=(t>=500)&(t<=min(4500,t[-1]*.4))&(amp>1e-12)
        rate=np.polyfit(t[fit], np.log(amp[fit]),1)[0]
        e_signal=result['modes'][tail,i,0,1]
        spectrum=abs(np.fft.fft(e_signal-e_signal.mean()))
        frequencies=np.fft.fftfreq(len(e_signal),d=(t[1]-t[0])/1000)
        frequency=abs(frequencies[np.argmax(spectrum)])
        metrics[name]={'early_growth_per_ms':float(rate),
             'tail_critical_amplitude':float(np.mean(amp[tail])),
             'tail_frequency_hz':float(frequency)}
    return metrics


def plot_dispersion(models, analyses, out):
    fig, axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,K in enumerate((3,4)):
        model=models[K];c,n,q,ev,_=analyses[K]
        for p,style in [(c.p0-.02,'--'),(c.p0,'-'),(c.p0+.02,':')]:
            values=np.array([dominant(linear_symbol(model,x,p)) for x in q])
            axes[row,0].plot(q*10,values.real,label=f'p={p:.2f}',ls=style)
            axes[row,1].plot(q*10,values.imag,label=f'p={p:.2f}',ls=style)
        axes[row,0].axhline(0,color='black',lw=.7)
        axes[row,0].set(xlim=(0,20),ylabel='growth (1/ms)',title=f'{K} populations')
        axes[row,1].set(xlim=(0,80),ylabel='dominant Im eigenvalue (rad/ms)',
                       title='Positive-imaginary representative')
        modes=np.arange(0,61);qd=2*np.pi*modes/model.length
        ed=np.array([dominant(linear_symbol(model,x,c.p0)) for x in qd])
        axes[row,1].scatter(qd*10,ed.imag,s=9,color='black',zorder=4)
        for ax in axes[row]:
            ax.axvline(c.q*10,color='grey',lw=.8);ax.grid(alpha=.2)
            ax.set_xlabel('angular wavenumber (rad/mm)');ax.legend(fontsize=8)
    path=out/'dispersion.png';fig.savefig(path,dpi=170);plt.close(fig);return path


def plot_1d(result,critical,K,out):
    labels=result['labels'];fig,axes=plt.subplots(3,len(labels),figsize=(14,8),
                                              constrained_layout=True)
    t=result['t'];late=t>=t[-1]-1500
    for j,label in enumerate(labels):
        field=result['snapshots'][:,j,:]
        lim=max(np.max(abs(field[late])),1e-5)
        axes[0,j].imshow(field[late].T,aspect='auto',origin='lower',cmap='RdBu_r',
            extent=[t[late][0]/1000,t[-1]/1000,0,LENGTH*.1],vmin=-lim,vmax=lim)
        axes[0,j].set(title=label,ylabel='x (mm)',xlabel='late time (s)')
        for mode,idx in [(3,1),(6,2)]:
            axes[1,j].semilogy(t/1000,np.maximum(abs(result['modes'][:,j,0,idx]),1e-12),label=f'E mode {mode}')
        axes[1,j].set(xlabel='time (s)',ylabel='Fourier magnitude');axes[1,j].legend(fontsize=8)
        axes[2,j].plot(t[late]/1000,field[late,0],lw=.8)
        axes[2,j].set(xlabel='late time (s)',ylabel='E deviation at x=0')
    fig.suptitle(f'{K}-population V1: direct nonlinear 1D field, below unforced onset')
    path=out/f'forcing_1d_{K}pop.png';fig.savefig(path,dpi=150);plt.close(fig);return path


def plot_2d(result,K,out):
    labels=result['labels'];fig,axes=plt.subplots(2,len(labels),figsize=(15,6),
                                             constrained_layout=True)
    for j,label in enumerate(labels):
        u=result['final'][j,0];lim=max(abs(u).max(),1e-5)
        axes[0,j].imshow(u.T,origin='lower',cmap='RdBu_r',extent=[0,LENGTH*.1]*2,
                        vmin=-lim,vmax=lim)
        axes[0,j].set(title=label,xlabel='x (mm)',ylabel='y (mm)')
        for idx,name in [(1,'(3,0)'),(2,'(0,3)'),(3,'(3,3)')]:
            axes[1,j].semilogy(result['t']/1000,
                 np.maximum(abs(result['modes'][:,j,0,idx]),1e-16),label=name)
        axes[1,j].set(xlabel='time (s)',ylabel='E mode magnitude');axes[1,j].legend(fontsize=7)
    fig.suptitle(f'{K}-population square torus: axis symmetry and transverse seeds')
    path=out/f'torus_2d_{K}pop.png';fig.savefig(path,dpi=150);plt.close(fig);return path


def dump_json(path, data):
    def convert(obj):
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,complex):return {'real':obj.real,'imag':obj.imag}
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(data,default=convert,indent=2)+'\n')

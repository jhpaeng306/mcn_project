"""Current analysis notebooks: restored literature-based widths and fixed gains and direct field runs.

Run with ``python -m simulations.main_analysis --dim 1`` (or --dim 2).
Only parameter-matched caches made by this module are loaded. Generic older
pipelines are used only for the boundary spectrum, never for their presets.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.fft import fftn, ifftn, dctn, idctn
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .v1_literature import make_model, coefficients, audit, spatial_sign
from .v1_simple import dump
from .fft import linear_symbol
from .pipeline_2d import compute_bifurcation_2d

OUT = Path(__file__).parent / 'output' / 'main_literature'
EPS = .02
VERSION = 2
# Cosine peaks in units of epsilon^2 (identical for K=3 and K=4, so the cases compare
# directly). K=4 thresholds: temporal 1.741, spatial 4.659 (A=B branch). K=3 (E->E=6.72):
# temporal 1.510; Re eta_x<0 so real spatial input acts on the A=-B branch with
# threshold 26.62. At eps=0.02 such a peak is outside the weakly nonlinear regime (the
# forced field is then more stable, see analysis.ipynb Sec. 7), so spatial 8 is kept
# and is below threshold for K=3.
PEAKS = {4: dict(temporal=3, spatial=8, combined=(3, 2), weak=1, traveling=2),
         3: dict(temporal=3, spatial=8, combined=(3, 2), weak=1, traveling=2)}
# Standing seed phase (quarter wave for K=3, aligned with its A=-B spatial branch).
SEED_SHIFT = {4: 0., 3: np.pi/2}
LATTICE = {4: 'square', 3: 'line'}
# K=3 combined forcing grows at ~1.5e-5/ms from the common seed; 300 s lets it saturate.
DURATION = {4: 150000., 3: 300000.}


def forcing_cases(K=4):
    P = PEAKS[K]; t, x = P['combined']
    base = {'none': {}, 'temporal': {'temporal': P['temporal']}, 'spatial': {'spatial': P['spatial']},
            'spatial + temporal': {'temporal': t, 'spatial': x}}
    controls = {'weak temporal': {'temporal': P['weak']},
                'detuned temporal': {'temporal': P['temporal'], 'omega_factor': 1.5},
                'detuned spatial': {'spatial': P['spatial'], 'space_factor': 3},
                'detuned combined': {'temporal': t, 'spatial': x, 'omega_factor': 1.5, 'space_factor': 3}}
    return base, controls


def model_audit(K, model, cr, nf, extra):
    P = PEAKS[K]
    return audit(model, cr, nf, extra, temporal=P['temporal'], spatial=P['spatial'],
                 combined=P['combined'], traveling=P['traveling'], lattice=LATTICE[K],
                 require_spatial=K==4)


BASE, CONTROLS = forcing_cases(4)
UNFORCED = {'below onset': {}, 'standing seed above onset': {'offset': 1, 'seed_kind': 'standing'},
            'traveling seed above onset': {'offset': 1, 'seed_kind': 'traveling'}}


def specification(model, critical, cases, dim, boundary, wavevector, n, dt, duration, save_ms, seed_shift=0.):
    extra = {} if seed_shift == 0 else {'seed_shift': seed_shift}
    return dict(**extra, version=VERSION, K=model.K, G=model.A_p.tolist(), A=model.A.tolist(),
        Sigma=model.Sigma.tolist(), tau=model.tau.tolist(), gain=model.gain.tolist(),
        theta=model.theta.tolist(), L=model.length, kappa_c=critical.p0, omega=critical.omega0,
        epsilon=EPS, dim=dim, boundary=boundary, wavevector=list(wavevector), n=n,
        dt=dt, duration=duration, save_ms=save_ms, seed=2026, cases=cases,
        drive_weights=model.metadata.get('drive_weights',[1.,1.]+[0.]*(model.K-2)))


def integrate(model, critical, cases, *, dim=1, boundary='periodic', wavevector=None,
              n=64, dt=1., duration=150000., save_ms=5., seed_shift=0.):
    """Batched RK4 with FFT or reflecting DCT-II convolution.

    Common small initial perturbation, with transverse critical modes in 2D.
    Dense mode histories, 1D E histories, and sparse/late all-population fields.
    Offsets and input peaks in cases are coefficients of epsilon squared.
    """
    wavevector = tuple(wavevector or ((critical.mode,) if dim == 1 else (critical.mode, 0)))
    labels = list(cases); count = len(labels); K = model.K
    shape = (n,) * dim; axes = tuple(range(2, 2 + dim)); ext = (None,) * dim
    x = (np.arange(n) + .5) * model.length / n
    grid = np.meshgrid(*([x] * dim), indexing='ij')
    factor = 2*np.pi if boundary == 'periodic' else np.pi
    q = 2*np.pi*np.fft.fftfreq(n, d=model.length/n) if boundary == 'periodic' else np.pi*np.arange(n)/model.length
    q2 = q*q if dim == 1 else q[:, None]**2 + q[None, :]**2
    offsets = np.array([cases[k].get('offset', -1.) for k in labels])
    kappas = critical.p0 + EPS**2 * offsets
    symbol = np.array([model.connectivity(p)[(slice(None), slice(None))+ext] *
                       np.exp(-model.Sigma[(slice(None), slice(None))+ext]*q2/2) for p in kappas]).reshape(count, K, K, -1)
    tau = model.tau.reshape((1,K)+(1,)*dim)
    gain = model.gain.reshape((1,K)+(1,)*dim); theta = model.theta.reshape((1,K)+(1,)*dim)
    offset = expit(-gain*theta)
    def forward(u):
        return fftn(u, axes=axes) if boundary == 'periodic' else dctn(u, type=2, norm='ortho', axes=axes)
    def inverse(u):
        return ifftn(u, axes=axes).real if boundary == 'periodic' else idctn(u, type=2, norm='ortho', axes=axes)
    def mode_shape(vector, shift=0.):
        phases = [factor*k*z/model.length for k,z in zip(vector,grid)]
        phases[0] = phases[0] + shift
        return np.exp(1j*sum(phases)) if boundary == 'periodic' else np.prod([np.cos(z) for z in phases],axis=0)
    carrier = mode_shape(wavevector, seed_shift)
    base = .02*EPS*critical.phi.real.reshape((K,)+(1,)*dim)*carrier.real
    rng = np.random.default_rng(2026)
    if dim == 2:
        base += .01*EPS*critical.phi.real[:,None,None]*mode_shape(wavevector[::-1]).real
    base += EPS*1e-4*rng.standard_normal(base.shape)
    state = np.repeat(base[None], count, axis=0)
    for i,k in enumerate(labels):
        kind = cases[k].get('seed_kind')
        if kind == 'standing':
            state[i] = .02*EPS*critical.phi.real.reshape((K,)+(1,)*dim)*carrier.real
        elif kind == 'traveling':
            state[i] = .02*EPS*np.real(critical.phi.reshape((K,)+(1,)*dim)*carrier)
    initial = state.copy()
    amplitudes = np.array([[cases[k].get(a,0.) for a in ('temporal','spatial','traveling')] for k in labels])*EPS**2
    omega = critical.omega0*np.array([cases[k].get('omega_factor',2.) for k in labels])
    phase = np.array([cases[k].get('space_factor',2.)*sum(factor*v*z/model.length for v,z in zip(wavevector,grid)) for k in labels])
    if boundary == 'reflect':
        spatial = np.array([np.prod([np.cos(cases[k].get('space_factor',2.)*factor*v*z/model.length) for v,z in zip(wavevector,grid)],axis=0) for k in labels])
    else:
        spatial = np.cos(phase)
    drive = np.array([cases[k].get('drive',model.metadata.get('drive_weights',[1.,1.]+[0.]*(K-2))) for k in labels])
    force_shape = (count,)+(1,)*dim
    def forcing(t):
        return amplitudes[:,0].reshape(force_shape)*np.cos(omega*t).reshape(force_shape) + amplitudes[:,1].reshape(force_shape)*spatial + amplitudes[:,2].reshape(force_shape)*np.cos(phase- (omega*t).reshape(force_shape))
    def rhs(u,t):
        fu = forward(expit(gain*(u-theta))-offset).reshape(count,K,-1)
        recur = inverse(np.einsum('cijz,cjz->ciz',symbol,fu).reshape((count,K)+shape))
        return (-u+recur+forcing(t)[:,None]*drive.reshape((count,K)+(1,)*dim))/tau
    def sample_modes(u):
        uh = forward(u)
        vectors = [(0,)*dim, wavevector]
        if dim == 2: vectors += [wavevector[::-1], tuple(sum(wavevector) for _ in range(dim))]
        vectors += [tuple(2*k for k in wavevector)]
        vals = []
        for v in vectors:
            c = uh[(slice(None),slice(None))+v]
            scale = n**dim if boundary == 'periodic' else np.sqrt(n**dim/2**sum(k!=0 for k in v))
            vals.append(c/scale)
        return np.stack(vals,axis=-1)
    steps = round(duration/dt); every = round(save_ms/dt)
    if every < 1 or abs(every*dt-save_ms)>1e-9: raise ValueError('save_ms must be a multiple of dt')
    ts=[]; modes=[]; rms=[]; transverse=[]; efield=[]; snaps=[]; st=[]
    snap_every = max(every, (steps//80//every)*every)
    for step in range(steps+1):
        t=step*dt
        if step%every==0:
            ts.append(t); modes.append(sample_modes(state))
            rms.append(np.std(state[:,0].reshape(count,-1),axis=1))
            transverse.append(np.sqrt(np.mean((state-state.mean(axis=-1,keepdims=True))**2,axis=(1,2,3))) if dim==2 else np.zeros(count))
            if dim==1: efield.append(state[:,0].astype(np.float32))
            if step%snap_every==0 or t>=duration-600:
                snaps.append(state.astype(np.float32));st.append(t)
        if step==steps: break
        a=rhs(state,t);b=rhs(state+dt*a/2,t+dt/2);c=rhs(state+dt*b/2,t+dt/2);d=rhs(state+dt*c,t+dt)
        state += dt*(a+2*b+2*c+d)/6
    if not np.isfinite(state).all(): raise FloatingPointError('Nonfinite field')
    return dict(t=np.array(ts),modes=np.array(modes),rms=np.array(rms),transverse=np.array(transverse),
        efield=np.array(efield),snapshot_t=np.array(st),snapshots=np.array(snaps),initial=initial,final=state,
        labels=np.array(labels),x=x,kappas=kappas,omega=omega,amplitudes=amplitudes,spatial=spatial,
        phase=phase,boundary=boundary,wavevector=np.array(wavevector),dim=dim,dt=dt,length=model.length,drive=drive)


def same_spec(a,b,rtol=1e-12):
    """Equal specifications up to floating-point round-off from different BLAS/LAPACK builds."""
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(same_spec(a[k],b[k],rtol) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(same_spec(x,y,rtol) for x,y in zip(a,b))
    if isinstance(a,float) or isinstance(b,float):
        return isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b)<=rtol*max(abs(a),abs(b),1e-300)
    return a==b


def cached_run(model,critical,cases,*,out,dim=1,boundary='periodic',wavevector=None,n=64,dt=1.,duration=150000.,save_ms=5.,recompute=False,seed_shift=0.):
    wavevector=tuple(wavevector or ((critical.mode,) if dim==1 else (critical.mode,0)))
    spec=specification(model,critical,cases,dim,boundary,wavevector,n,dt,duration,save_ms,seed_shift)
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    path=out/f'field_{boundary}.npz';out.mkdir(parents=True,exist_ok=True)
    if path.exists() and not recompute:
        with np.load(path) as saved:
            settings=out/f'settings_{boundary}.json'
            same=str(saved['fingerprint'])==fingerprint or (settings.exists() and
                same_spec(json.loads(settings.read_text()),json.loads(json.dumps(spec))))
            if same:
                print('Loaded matching run:',path,flush=True)
                return dict(saved)
    print('Integrating:',dim,boundary,len(cases),'cases',flush=True)
    r=integrate(model,critical,cases,dim=dim,boundary=boundary,wavevector=wavevector,n=n,dt=dt,duration=duration,save_ms=save_ms,seed_shift=seed_shift)
    r['fingerprint']=fingerprint
    np.savez_compressed(path,**r);dump(out/f'settings_{boundary}.json',spec)
    print('Saved:',path,flush=True)
    return r


def save(fig,path):
    fig.savefig(path,dpi=140);plt.close(fig)


def rms_blocks(t,values,window=100.):
    width=max(1,round(window/(t[1]-t[0])));count=len(t)//width
    return t[:count*width].reshape(count,width).mean(axis=1)/1000, np.sqrt(np.mean(abs(values[:count*width].reshape((count,width)+values.shape[1:]))**2,axis=1))


def connectivity(model,critical,out):
    J=model.connectivity(critical.p0);x=np.linspace(-400,400,401)
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i in range(model.K):
        for j in range(model.K):
            if J[i,j]==0:continue
            sigma=100*np.sqrt(model.Sigma[i,j]);kernel=J[i,j]*np.exp(-x*x/(2*sigma*sigma))/(np.sqrt(2*np.pi)*sigma)
            label=f'{model.names[j]} → {model.names[i]}'
            axs[0,0].plot(x,kernel,label=label);axs[0,1].plot(x,abs(kernel))
    axs[0,0].set(title='Signed Gaussian kernels at onset',xlabel='distance (μm)',ylabel='gain / μm')
    axs[0,0].legend(fontsize=6,ncol=2);axs[0,1].set(title='Kernel magnitudes',xlabel='distance (μm)')
    for ax,matrix,title in zip([axs[0,2],*axs[1]], [model.A_p,J,model.Sigma,100*np.sqrt(2*model.Sigma)], ['Effective gain G','Integrated gain κc G','Variance Σ (0.1 mm)²','Paper distance scale s (μm)']):
        width_panel=title.startswith(('Variance','Paper distance'))
        shown=np.ma.masked_where(J==0,matrix) if width_panel else matrix
        ax.imshow(shown,cmap='RdBu_r');ax.set(title=title,xticks=range(model.K),xticklabels=model.names,yticks=range(model.K),yticklabels=model.names,xlabel='source',ylabel='target')
        for i in range(model.K):
            for j in range(model.K):
                label='–' if width_panel and J[i,j]==0 else f'{matrix[i,j]:.5g}'
                ax.text(j,i,label,ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
    save(fig,out/'connectivity.png')


def tracked_spectrum(model,kappa,qs):
    rows=[]
    for q in qs:
        values=np.linalg.eigvals(linear_symbol(model,q,kappa))
        if rows:
            _,indices=linear_sum_assignment(abs(rows[-1][:,None]-values[None,:]))
            values=values[indices]
        else:values=values[np.lexsort((values.imag,values.real))]
        rows.append(values)
    rows=np.array(rows)
    selected=[]
    for row in rows:
        ids=np.flatnonzero(np.isclose(row.real,row.real.max(),atol=1e-11,rtol=0))
        selected.append(ids[np.argmax(row[ids].imag)])
    return rows,np.array(selected)


def dispersion(model,cr,nf,extra,out):
    qs=np.linspace(0,4,1001);rows,ids=tracked_spectrum(model,cr.p0,qs);lead=rows[np.arange(len(qs)),ids]
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j in range(model.K):
        axs[0].plot(qs*10,rows[:,j].real,color='.75',lw=.7)
        axs[1].plot(qs*10,rows[:,j].imag*1000/(2*np.pi),color='.8',lw=.7)
    axs[0].plot(qs*10,lead.real,color='C0',label='rightmost');axs[0].axhline(0,color='k',lw=.5)
    switches=np.flatnonzero(ids[1:]!=ids[:-1])+1
    # Break the selected curve at identity changes; do not draw a fictitious continuous branch.
    display=lead.imag*1000/(2*np.pi);display[switches]=np.nan
    axs[1].plot(qs*10,display,color='C0',label='rightmost, Im ≥ 0')
    for s in switches:
        for ax in axs:ax.axvline(qs[s]*10,color='C3',ls=':',alpha=.5)
    for ax in axs:ax.axvline(cr.q*10,color='k',ls='--',label='critical q');ax.set(xlabel='wavenumber (rad/mm)');ax.legend(fontsize=8)
    axs[0].set(ylabel='growth (ms⁻¹)',title='All branches and the rightmost eigenvalue')
    axs[1].set(ylabel='signed frequency (Hz)',title='Dotted lines: selected branch changes')
    save(fig,out/'dispersion.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for p in [-2,-1,0,1,2]:
        r,i=tracked_spectrum(model,cr.p0+p*EPS**2,qs);v=r[np.arange(len(qs)),i]
        axs[0].plot(qs*10,v.imag*1000/(2*np.pi),label=f'p = {p}ε²')
        ms=np.arange(13);vals=[]
        for m in ms:
            ev=np.linalg.eigvals(linear_symbol(model,2*np.pi*m/model.length,cr.p0+p*EPS**2))
            vals.append(max(ev,key=lambda z:(round(z.real,11),z.imag)))
        axs[1].plot(ms,np.array(vals).imag*1000/(2*np.pi),'o-',ms=3,label=f'p = {p}ε²')
    axs[0].set(xlim=(2,7),xlabel='wavenumber (rad/mm)',ylabel='frequency (Hz)',title='Frequency near the critical wave number')
    axs[1].set(xlabel='integer spatial mode k',ylabel='frequency (Hz)',title='Discrete torus modes');axs[1].legend(fontsize=8)
    save(fig,out/'frequency_near_onset.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    ps=np.linspace(-3*EPS**2,3*EPS**2,101)
    for k in range(6):axs[0].plot(ps,[np.linalg.eigvals(linear_symbol(model,2*np.pi*k/model.length,cr.p0+p)).real.max() for p in ps],label=f'k={k}')
    axs[0].axhline(0,color='k',lw=.5);axs[0].legend(ncol=2,fontsize=8);axs[0].set(xlabel='p = κ − κc',ylabel='growth (ms⁻¹)',title=f'First torus mode: k={cr.mode}')
    mode=np.arange(13);growth=[np.linalg.eigvals(linear_symbol(model,2*np.pi*k/model.length,cr.p0)).real.max() for k in mode]
    axs[1].plot(mode,growth,'o-');axs[1].axhline(0,color='k',lw=.5);axs[1].set(xlabel='mode k',ylabel='growth (ms⁻¹)',title='Discrete spectrum at onset')
    save(fig,out/'bifurcation.png')
    return dict(nu_q=extra['nu_q'],nu_q_physical=extra['nu_q']*.1,switch_brackets_rad_per_mm=[[float(qs[s-1]*10),float(qs[s]*10)] for s in switches])


def forcing_figure(r,labels,out,name,late=False):
    indices=[list(r['labels']).index(k) for k in labels]
    times=r['t'];keep=times>=times[-1]-500 if late else np.ones(len(times),bool)
    t=times[keep];fields=r['efield'][keep][:,indices]
    fig,axs=plt.subplots(len(labels),3,figsize=(13,2.5*len(labels)),layout='constrained',squeeze=False)
    limit=max(abs(fields).max(),1e-8)
    for row,(j,label) in enumerate(zip(indices,labels)):
        amp=r['amplitudes'][j];w=r['omega'][j]
        force=amp[0]*np.cos(w*t[:,None])+amp[1]*r['spatial'][j]+amp[2]*np.cos(r['phase'][j]-w*t[:,None])
        field=r['efield'][keep,j]
        if late:
            shown=[force,field];tt=t/1000;titles=['Input to E only' if r['drive'][j,1]==0 else 'Input to E/PV','E activity']
        else:
            tt,fr=rms_blocks(t,force);_,uu=rms_blocks(t,field)
            shown=[fr,uu];titles=['Input, 100-ms RMS','E activity, 100-ms RMS']
        for col,values in enumerate(shown):
            lim=max(abs(values).max(),EPS**2) if col==0 else limit
            im=axs[row,col].imshow(values.T,origin='lower',aspect='auto',extent=[tt[0],tt[-1],0,float(r['length'])*.1],cmap='RdBu_r' if late else 'magma',vmin=-lim if late else 0,vmax=lim)
            axs[row,col].set(xlabel='time (s)',ylabel='x (mm)',title=f'{label}: {titles[col]}');fig.colorbar(im,ax=axs[row,col],shrink=.8)
        for idx,labelmode in [(1,f'critical k={int(r["wavevector"][0])}'),(-1,f'quadratic harmonic k={2*int(r["wavevector"][0])}')] :
            tt,mag=rms_blocks(times,r['modes'][:,j,0,idx]);axs[row,2].semilogy(tt,np.maximum(mag,1e-12),label=labelmode)
        axs[row,2].set(xlabel='time (s)',ylabel='E Fourier magnitude (100-ms RMS)',title=f'p = {r["kappas"][j]-r["kappas"][0]-EPS**2:+.4g}' if 'below onset' in labels else 'Critical-mode growth or decay')
        if late:axs[row,2].set_xlim(t[0]/1000,t[-1]/1000)
        axs[row,2].legend(fontsize=8)
    save(fig,out/name)


def frequency_figure(r,cr,out):
    labels=['temporal','spatial + temporal'];fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,label in enumerate(labels):
        j=list(r['labels']).index(label);tail=r['t']>=r['t'][-1]-10000;t=r['t'][tail]
        response=r['modes'][tail,j,0,1].real
        drive=r['amplitudes'][j,0]*np.cos(r['omega'][j]*t)
        freq=np.fft.rfftfreq(len(t),d=(t[1]-t[0])/1000)
        for ax,signal,title in [(axs[row,0],drive,'temporal input'),(axs[row,1],response,'critical E mode')]:
            power=abs(np.fft.rfft((signal-signal.mean())*np.hanning(len(signal))))**2
            ax.plot(freq,power/max(power.max(),1e-30));ax.set(xlim=(0,65),xlabel='frequency (Hz)',ylabel='normalized power',title=f'{label}: {title}')
            ax.axvline(cr.frequency_hz,color='C1',ls='--',label='f₀');ax.axvline(2*cr.frequency_hz,color='C2',ls=':',label='2f₀');ax.legend(fontsize=8)
    save(fig,out/'frequency_doubling_verification.png')


def branch_figures(model,cr,nf,out):
    ps=np.linspace(-3,3,301)*EPS**2
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    a=nf.alpha.real*np.maximum(ps,0)
    travel_stable=nf.b.real<0 and (nf.c-nf.b).real<0
    standing_stable=(nf.b+nf.c).real<0 and (nf.b-nf.c).real<0
    ax.plot(ps,np.sqrt(a/-nf.b.real),'-' if travel_stable else '--',label='traveling ('+('stable' if travel_stable else 'unstable to counter-propagating wave')+' at cubic order)')
    ax.plot(ps,np.sqrt(a/-(nf.b+nf.c).real),'-' if standing_stable else '--',label='standing ('+('stable' if standing_stable else 'unstable to imbalance')+' at cubic order)')
    ax.plot(ps[ps<=0],np.zeros(sum(ps<=0)),color='k',label='homogeneous (stable)')
    ax.set(xlabel='p = κ − κc',ylabel='normal-form amplitude',title='Unforced branches');ax.legend(fontsize=8);save(fig,out/'normal_form_branches.png')
    xx=np.linspace(0,model.length*.1,256);tt=np.linspace(0,.3,301);phase=cr.q*xx[None,:]*10;time=cr.omega0*tt[:,None]*1000
    fields=[np.zeros_like(phase+time),np.cos(phase)*np.cos(time),np.cos(phase-time),np.cos(phase+time)]
    fig,axs=plt.subplots(1,4,figsize=(13,3),layout='constrained')
    for ax,u,label in zip(axs,fields,['homogeneous','standing','traveling +x','traveling −x']):
        ax.imshow(u.T,origin='lower',aspect='auto',extent=[0,.3,0,model.length*.1],vmin=-1,vmax=1,cmap='RdBu_r');ax.set(title=label,xlabel='time (s)',ylabel='x (mm)')
    fig.suptitle('Analytic leading-order illustrations (not simulated fields)');save(fig,out/'ideal_wave_patterns.png')


def summarize(r,cr):
    tail=r['t']>.9*r['t'][-1];answer={}
    for j,label in enumerate(r['labels']):
        coefficient=r['modes'][:,j,:,1]
        if str(r['boundary'])=='periodic':
            aa=coefficient@cr.psi.conj();bb=coefficient.conj()@cr.psi.conj();amp=np.sqrt(abs(aa)**2+abs(bb)**2)
        else:amp=np.linalg.norm(coefficient,axis=1)
        signal=coefficient[tail,0];freq=np.fft.fftfreq(len(signal),d=(r['t'][1]-r['t'][0])/1000)
        power=abs(np.fft.fft((signal-signal.mean())*np.hanning(len(signal))))
        answer[str(label)]=dict(kappa=float(r['kappas'][j]),p=float(r['kappas'][j]-cr.p0),
            critical_initial=float(amp[0]),critical_late=float(amp[tail].mean()),
            E_mode_late_rms=float(np.sqrt(np.mean(abs(signal)**2))),
            mode_frequency_hz=float(abs(freq[power.argmax()])),
            transverse_initial=float(r['transverse'][0,j]),transverse_final=float(r['transverse'][-1,j]))
    return answer


def run_1d(K=4,*,recompute=False):
    out=OUT/f'{K}pop_1d';out.mkdir(parents=True,exist_ok=True)
    model=make_model(K);cr,nf,extra=coefficients(model);checks=model_audit(K,model,cr,nf,extra)
    base,controls=forcing_cases(K)
    e_only=[1.]+[0.]*(K-1)
    cases={**base,**controls,**UNFORCED,'temporal E only':{'temporal':PEAKS[K]['temporal'],'drive':e_only}}
    r=cached_run(model,cr,cases,out=out,recompute=recompute,seed_shift=SEED_SHIFT[K],duration=DURATION[K])
    connectivity(model,cr,out);disp=dispersion(model,cr,nf,extra,out)
    forcing_figure(r,list(base),out,'forcing_comparison.png')
    forcing_figure(r,list(base),out,'forcing_comparison_late_zoom.png',True)
    forcing_figure(r,list(controls),out,'off_resonance_and_subthreshold.png')
    forcing_figure(r,list(UNFORCED),out,'unforced_parameter_sweep.png')
    forcing_figure(r,list(UNFORCED),out,'unforced_parameter_sweep_late_zoom.png',True)
    forcing_figure(r,['temporal','temporal E only'],out,'input_population_comparison.png',True)
    frequency_figure(r,cr,out);branch_figures(model,cr,nf,out)
    _,_,e_only_extra=coefficients(model,drive=e_only)
    summary=dict(kappa_c=cr.p0,critical_mode=cr.mode,frequency_hz=cr.frequency_hz,
        alpha=nf.alpha,b=nf.b,c=nf.c,checks=checks,dispersion=disp,cases=summarize(r,cr))
    if K!=4:summary.update(G=model.A_p,peaks=PEAKS[K],seed_shift=SEED_SHIFT[K],duration_ms=DURATION[K],
        e_only_temporal_threshold=float(abs(nf.alpha)/abs(e_only_extra['eta_t_unit'])))
    dump(out/'summary.json',summary)
    return model,cr,nf,extra,r,summary,out


def plot_2d(results,geometries,out):
    labels=list(BASE);boundaries=list(results)  # same four labels for every K
    fig,axs=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    limit=max(abs(results[b]['final'][:4,0]-results[b]['final'][:4,0].mean(axis=(1,2),keepdims=True)).max() for b in boundaries)
    for row,b in enumerate(boundaries):
        r=results[b]
        for j,label in enumerate(labels):
            u=r['final'][j,0];u=u-u.mean()
            im=axs[row,j].imshow(u.T,origin='lower',extent=[0,float(r['length'])*.1,0,float(r['length'])*.1],cmap='RdBu_r',vmin=-limit,vmax=limit)
            axs[row,j].set(title=f'{b}: {label}',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='E activity − spatial mean',shrink=.7);save(fig,out/'final_state_comparison.png')
    for name,key,title in [('pattern_contrast.png','rms','Spatial contrast'),('critical_mode_amplitude.png','modes','Selected critical mode')]:
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,b in zip(axs,boundaries):
            r=results[b]
            for j,label in enumerate(r['labels']):
                v=r[key][:,j] if key=='rms' else r[key][:,j,0,1]
                tt,mag=rms_blocks(r['t'],v)
                # Periodic complex coefficients reconstruct real cosine with factor two.
                if key=='modes' and b=='periodic':mag=2*mag
                ax.semilogy(tt,np.maximum(mag,1e-12),label=label)
            ax.set(xlabel='time (s)',ylabel='100-ms RMS amplitude',title=f'{b}: {title}');ax.legend(fontsize=8)
        save(fig,out/name)
    fig,axs=plt.subplots(2,4,figsize=(13,7),layout='constrained')
    for row,b in enumerate(boundaries):
        r=results[b];j=list(r['labels']).index('spatial + temporal')
        # Snapshots closest to the stated fractions; no phase optimization.
        for col,fraction in enumerate([0,.5,.8,1.]):
            idx=np.argmin(abs(r['snapshot_t']-fraction*r['t'][-1]));u=r['snapshots'][idx,j,0];u=u-u.mean()
            im=axs[row,col].imshow(u.T,origin='lower',extent=[0,float(r['length'])*.1,0,float(r['length'])*.1],cmap='RdBu_r',vmin=-limit,vmax=limit)
            axs[row,col].set(title=f'{b}, {r["snapshot_t"][idx]/1000:.1f} s',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='centered E activity');save(fig,out/'combined_evolution.png')
    r=results['periodic'];j=list(r['labels']).index('traveling');fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    u=r['final'][j,0];im=axs[0].imshow(u.T,origin='lower',extent=[0,float(r['length'])*.1,0,float(r['length'])*.1],cmap='RdBu_r');fig.colorbar(im,ax=axs[0],shrink=.7)
    axs[0].set(title='Traveling x input: stripes parallel to y',xlabel='x (mm)',ylabel='y (mm)')
    for idx,label in [(1,str(tuple(r['wavevector'].tolist()))),(2,str(tuple(r['wavevector'][::-1].tolist()))),(3,str((int(r['wavevector'].sum()),)*2))]:
        tt,mag=rms_blocks(r['t'],r['modes'][:,j,0,idx]);axs[1].semilogy(tt,np.maximum(mag,1e-14),label=label)
    axs[1].set(xlabel='time (s)',ylabel='E Fourier magnitude (100-ms RMS)');axs[1].legend()
    tt,mag=rms_blocks(r['t'],r['transverse'][:,j]);axs[2].semilogy(tt,np.maximum(mag,1e-14));axs[2].set(xlabel='time (s)',ylabel='RMS variation along y',title='Transverse perturbation')
    save(fig,out/'traveling_stripes.png')


def run_2d(K=4,*,recompute=False,boundary=None):
    out=OUT/f'{K}pop_2d';out.mkdir(parents=True,exist_ok=True)
    model=make_model(K);results={};geometries={};summary={}
    for b in ([boundary] if boundary else ['periodic','reflect']):
        geometry=compute_bifurcation_2d(model,boundary=b,p_bounds=(0,1),scan_points=101)
        geometries[b]=geometry;cases=dict(forcing_cases(K)[0])
        if b=='periodic':cases['traveling']={'traveling':2}
        r=cached_run(model,geometry.point,cases,out=out,dim=2,boundary=b,wavevector=geometry.wavevector,n=32,recompute=recompute)
        results[b]=r
        summary[b]=dict(kappa_c=geometry.point.p0,wavevector=geometry.wavevector,frequency_hz=geometry.point.frequency_hz,cases=summarize(r,geometry.point))
        dump(out/f'summary_{b}.json',summary[b])
    if boundary is None:plot_2d(results,geometries,out);dump(out/'summary.json',summary)
    return model,geometries,results,summary,out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dim',type=int,choices=[1,2],required=True);parser.add_argument('--K',type=int,choices=[3,4],default=4);parser.add_argument('--recompute',action='store_true');parser.add_argument('--boundary',choices=['periodic','reflect']);args=parser.parse_args()
    if args.dim==1:run_1d(args.K,recompute=args.recompute)
    else:run_2d(args.K,recompute=args.recompute,boundary=args.boundary)

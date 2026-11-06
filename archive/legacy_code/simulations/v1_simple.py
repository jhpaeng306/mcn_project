"""Fresh integer-gain examples. No previous gains, fits, or saved runs are read.

Run: MPLCONFIGDIR=/private/tmp/mcn-mpl .venv/bin/python -m simulations.v1_simple
Units: ms, 0.1 mm, dimensionless activity. Geometry L=20 (2 mm).
"""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.special import expit
from scipy.linalg import eig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .simulation import ModelParameters
from .fft import (linear_symbol, compute_bifurcation, compute_normal_form,
                  compute_forcing_coefficients)

OUT = Path(__file__).parent / 'output' / 'v1_simple'
LENGTH = 20.0
EPSILON = .02
G3 = np.array([[7., -7., -8.], [4., -7., -8.], [8., 0., 0.]])


def make_model(K):
    """Integer motif; VIP adds unit-strength E->VIP, SST->VIP, VIP->SST."""
    G = np.zeros((K, K)); G[:3, :3] = G3
    if K == 4:
        G[2, 3] = -1; G[3, 0] = 1; G[3, 2] = -1
    elif K != 3:
        raise ValueError('K must be 3 or 4')
    # Source-dependent widths: 75 um excitation, 100 um inhibition.
    return ModelParameters(K=K, A=np.zeros_like(G), A_p=G,
        Sigma=np.tile([.75**2]+[1.]*(K-1), (K, 1)),
        tau=[10.,15.,30.,20.][:K], gain=[4.]*K, theta=[.25]*K,
        length=LENGTH, names=('E','PV','SST','VIP')[:K],
        time_unit='ms', space_unit='0.1 mm')


def coefficients(model):
    critical = compute_bifurcation(model, p_bounds=(0,1), max_mode=40, scan_points=101)
    normal = compute_normal_form(model, critical)
    q=critical.q; omega=critical.omega0; phi=critical.phi; psi=critical.psi
    def Q(k,v,w):
        return (model.connectivity(critical.p0)*np.exp(-model.Sigma*k*k/2)) @ (normal.f2*v*w/2)/model.tau
    def C(k,u,v,w):
        return (model.connectivity(critical.p0)*np.exp(-model.Sigma*k*k/2)) @ (normal.f3*u*v*w/6)/model.tau
    def R(o,k,v):
        return np.linalg.solve(1j*o*np.eye(model.K)-linear_symbol(model,k,critical.p0),v)
    cp=np.vdot(psi,2*Q(q,phi,normal.h11_zero)
       +4*Q(q,phi.conj(),R(2*omega,np.sqrt(2)*q,Q(np.sqrt(2)*q,phi,phi)))
       +4*Q(q,phi,R(0,np.sqrt(2)*q,Q(np.sqrt(2)*q,phi,phi.conj())))
       +6*C(q,phi,phi,phi.conj()))
    drive=np.zeros(model.K);drive[:2]=.5
    force=compute_forcing_coefficients(model,normal,temporal_harmonic=drive,spatial_harmonic=drive)
    ew=2*np.vdot(psi,Q(q,phi.conj(),R(2*omega,2*q,drive/model.tau)))
    M=linear_symbol(model,q,critical.p0)
    vals,left,right=eig(M,left=True,right=True);j=np.argmin(abs(vals-1j*omega))
    Mq=-q*(model.Sigma*model.connectivity(critical.p0)*np.exp(-model.Sigma*q*q/2)*normal.f1[None,:])/model.tau[:,None]
    slope=np.vdot(left[:,j],Mq@right[:,j])/np.vdot(left[:,j],right[:,j])
    return critical,normal,dict(c_perp=cp,eta_t_unit=force.eta_t,eta_x_unit=force.eta_x,eta_w_unit=ew,nu_q=slope)


def audit(model,critical,normal,extra):
    # Check all distinct square-torus radii below q=8, then a decreasing tail bound.
    radii=np.unique(np.add.outer(np.arange(40)**2,np.arange(40)**2))
    qs=2*np.pi*np.sqrt(radii)/model.length;keep=qs<8
    growth=np.array([np.linalg.eigvals(linear_symbol(model,q,critical.p0)).real.max() for q in qs[keep]])
    gap=-growth[radii[keep]!=critical.mode**2].max()
    tail=np.max(np.sum(abs(model.connectivity(critical.p0))*np.exp(-model.Sigma*8**2/2)*normal.f1[None,:],axis=1))
    assert critical.mode==2 and gap>0 and tail<1
    assert normal.b.real<0 and normal.c.real<0 and extra['c_perp'].real<0
    # Full gain sweep: detect re-entrant/earlier instabilities on all these shells.
    for kappa in np.linspace(0,critical.p0,61)[:-1]:
        mats=(-np.eye(model.K)+model.connectivity(kappa)[None]*np.exp(-model.Sigma[None]*qs[keep,None,None]**2/2)*normal.f1[None,None,:])/model.tau[None,:,None]
        assert np.linalg.eigvals(mats).real.max()<0
    alpha=normal.alpha;b=normal.b;c=normal.c
    def positive_root(a,g,e):
        roots=np.roots([abs(g)**2,2*(a*np.conj(g)).real,abs(a)**2-abs(e)**2])
        return float(max(roots.real))
    temporal_rho=positive_root(-alpha,b+c,8*extra['eta_t_unit'])
    combined_a=-alpha+4*extra['eta_x_unit']
    combined_rho=positive_root(combined_a,b+c,8*extra['eta_t_unit'])
    combined_conditions=[(combined_a+2*(b+c)*combined_rho).real,
        (-alpha-4*extra['eta_x_unit']+2*b*combined_rho).real,
        ((-alpha+2*b*combined_rho)*np.conj(4*extra['eta_x_unit'])).real]
    assert max(combined_conditions)<0
    travel_rho=positive_root(-alpha,b,8*extra['eta_w_unit'])
    travel_conditions=[(-alpha+2*b*travel_rho).real,(-alpha+c*travel_rho).real,
                       (-alpha+extra['c_perp']*travel_rho).real]
    assert max(travel_conditions)<0
    return dict(square_spectral_gap=gap,tail_row_sum_bound=tail,
        temporal_rho=temporal_rho,combined_rho=combined_rho,travel_rho=travel_rho,
        combined_stability_conditions=combined_conditions,travel_stability_conditions=travel_conditions)


def integrate(model,critical,cases,*,epsilon=EPSILON,dim=1,n=64,slow_duration=60.,dt=1.,save_ms=5.,seed=2026,transverse=True,initial=None):
    """Direct FFT/RK4 integration of the nonlinear field, not of amplitudes.

    Cases contain scaled cosine peaks; physical peaks are epsilon**2 times these.
    Same initial state for controls; include generic transverse perturbations in 2D.
    """
    labels=list(cases);count=len(labels);K=model.K
    axes=tuple(range(2,2+dim));shape=(n,)*dim
    q=2*np.pi*np.fft.fftfreq(n,d=model.length/n)
    q2=q*q if dim==1 else q[:,None]**2+q[None,:]**2
    ext=(None,)*dim
    p=critical.p0-epsilon**2
    symbol=(model.connectivity(p)[(slice(None),slice(None))+ext]*np.exp(-model.Sigma[(slice(None),slice(None))+ext]*q2/2)).reshape(K,K,-1)
    tau=model.tau.reshape((1,K)+(1,)*dim);gain=model.gain.reshape((1,K)+(1,)*dim);theta=model.theta.reshape((1,K)+(1,)*dim)
    offset=expit(-gain*theta)
    phase=2*np.pi*critical.mode*np.arange(n)/n
    base=.02*epsilon*np.real(critical.phi[:,None])*np.cos(phase)[None,:]
    rng=np.random.default_rng(seed)
    if dim==2:
        base=np.broadcast_to(base[:,:,None],(K,n,n)).copy()
        if transverse:
            base+=.01*epsilon*np.real(critical.phi[:,None,None])*np.cos(phase)[None,None,:]
            base+=epsilon*1e-4*rng.standard_normal(base.shape)
    else:
        base+=epsilon*1e-4*rng.standard_normal(base.shape)
    state=np.repeat(base[None],count,axis=0) if initial is None else np.array(initial,copy=True)
    initial=state.copy()
    drive=np.zeros(K);drive[:2]=1
    forces=[]
    for name in ['temporal','spatial','traveling']:
        forces.append((epsilon**2*np.array([cases[k].get(name,0.) for k in labels])[:,None]*drive[None,:]).reshape((count,K)+(1,)*dim))
    phase_x=2*phase if dim==1 else (2*phase)[:,None];spatial=np.cos(phase_x);omega=2*critical.omega0
    def rhs(u,t):
        fu=np.fft.fftn(expit(gain*(u-theta))-offset,axes=axes).reshape(count,K,-1)
        recur=np.fft.ifftn(np.einsum('ijz,cjz->ciz',symbol,fu).reshape((count,K)+shape),axes=axes).real
        force=forces[0]*np.cos(omega*t)+forces[1]*spatial+forces[2]*np.cos(phase_x-omega*t)
        return (-u+recur+force)/tau
    duration=slow_duration/epsilon**2;steps=int(round(duration/dt));every=max(1,round(save_ms/dt))
    times=[];modes=[];perp=[];snapshots=[];snap_times=[];k=critical.mode
    snap_every=max(every,steps//80//every*every)
    for step in range(steps+1):
        t=step*dt
        if step%every==0:
            uh=np.fft.fftn(state,axes=axes)/n**dim
            if dim==1:
                coeff=np.stack([uh[:,:,0],uh[:,:,k],uh[:,:,2*k]],axis=-1);pr=np.zeros(count)
            else:
                coeff=np.stack([uh[:,:,0,0],uh[:,:,k,0],uh[:,:,0,k],uh[:,:,k,k],uh[:,:,2*k,0]],axis=-1)
                pr=np.sqrt(np.mean((state-state.mean(axis=-1,keepdims=True))**2,axis=(1,2,3)))
            times.append(t);modes.append(coeff);perp.append(pr)
            if (dim==1 and t>=duration-2000) or step%snap_every==0:
                snapshots.append(state[:,0].copy());snap_times.append(t)
        if step==steps:break
        ka=rhs(state,t);kb=rhs(state+dt*ka/2,t+dt/2);kc=rhs(state+dt*kb/2,t+dt/2);kd=rhs(state+dt*kc,t+dt)
        state+=dt*(ka+2*kb+2*kc+kd)/6
    if not np.isfinite(state).all():raise FloatingPointError('Nonfinite state')
    return dict(t=np.array(times),modes=np.array(modes),transverse_rms=np.array(perp),snapshots=np.array(snapshots),snapshot_t=np.array(snap_times),
        initial=initial,final=state,labels=np.array(labels),epsilon=epsilon,dt=dt,n=n,dim=dim,duration=duration,p=p,
        G=model.A_p,Sigma=model.Sigma,tau=model.tau,gain=model.gain,theta=model.theta)


def metrics(result,critical):
    times=result['t'];tail=times>.9*times[-1];coeff=result['modes'][:,:,:,1]
    # psi projection removes the negative temporal-frequency eigenvector.
    a=np.einsum('j,tcj->tc',critical.psi.conj(),coeff)
    # Both +/- spatial directions contribute to standing modes. Report total
    # positive-frequency energy, averaged over late oscillation cycles.
    b=np.einsum('j,tcj->tc',critical.psi.conj(),coeff.conj())
    out={}
    for j,label in enumerate(result['labels']):
        sig=coeff[tail,j,0];freq=np.fft.fftfreq(len(sig),d=(times[1]-times[0])/1000)
        power=abs(np.fft.fft(sig-sig.mean()));peak=abs(freq[np.argmax(power)])
        out[str(label)]=dict(critical_amplitude=float(np.mean(np.sqrt(abs(a[tail,j])**2+abs(b[tail,j])**2))),
            frequency_hz=float(peak),transverse_initial=float(result['transverse_rms'][0,j]),transverse_final=float(result['transverse_rms'][-1,j]),
            critical_initial=float(np.sqrt(abs(a[0,j])**2+abs(b[0,j])**2)))
    return out


def plot_mode(ax,result,case,idx,label):
    width=max(1,round(100/(result['t'][1]-result['t'][0])))
    count=len(result['t'])//width
    time=result['t'][:count*width].reshape(count,width).mean(axis=1)/1000
    magnitude=np.sqrt(np.mean(abs(result['modes'][:count*width,case,0,idx].reshape(count,width))**2,axis=1))
    ax.semilogy(time,np.maximum(magnitude,1e-14),label=label)


def plots(models,analyses,records,results,out):
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,K in enumerate([3,4]):
        model=models[K];cr,nf,extra=analyses[K];qs=np.linspace(0,4,801)
        branches=np.array([np.linalg.eigvals(linear_symbol(model,q,cr.p0)) for q in qs])
        selected=np.array([max(v,key=lambda z:(round(z.real,12),z.imag)) for v in branches])
        axs[row,0].plot(qs*10,selected.real);axs[row,0].axhline(0,color='k',lw=.5)
        axs[row,1].plot(qs*10,np.maximum(0,selected.imag)*1000/(2*np.pi))
        modes=np.arange(13);qm=2*np.pi*modes/model.length;ev=np.array([max(np.linalg.eigvals(linear_symbol(model,q,cr.p0)),key=lambda z:(round(z.real,12),z.imag)) for q in qm])
        axs[row,0].scatter(qm*10,ev.real,s=14);axs[row,1].scatter(qm*10,ev.imag*1000/(2*np.pi),s=14)
        axs[row,0].set(title=f'{K} populations: onset',ylabel='rightmost growth (1/ms)',ylim=(-.04,.003))
        axs[row,1].set(title=f'{K} populations: dominant frequency',ylabel='frequency (Hz)')
        for ax in axs[row]:ax.set(xlabel='wavenumber (rad/mm)');ax.grid(alpha=.2)
    fig.savefig(out/'dispersion.png',dpi=160);plt.close(fig)
    for K in [3,4]:
        r=results[(K,1)];labs=r['labels'];ts=r['snapshot_t'];late=ts>=ts[-1]-600
        fig,axs=plt.subplots(2,len(labs),figsize=(14,6),layout='constrained')
        limit=max(abs(r['snapshots'][late]).max(),1e-8)
        for j,label in enumerate(labs):
            axs[0,j].imshow(r['snapshots'][late,j].T,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[ts[late][0]/1000,ts[-1]/1000,0,2])
            axs[0,j].set(title=str(label),xlabel='time (s)',ylabel='x (mm)')
            for idx,name in [(1,'critical k=2'),(2,'harmonic k=4')]:plot_mode(axs[1,j],r,j,idx,name)
            axs[1,j].set(xlabel='time (s)',ylabel='E mode magnitude (100 ms RMS)');axs[1,j].legend(fontsize=8)
        fig.colorbar(axs[0,0].images[0],ax=list(axs[0]),label='E activity (dimensionless)',shrink=.8)
        fig.suptitle(f'{K} populations: direct field, p = -epsilon², epsilon = 0.02; common color scale')
        fig.savefig(out/f'forcing_1d_{K}pop.png',dpi=160);plt.close(fig)
        r=results[(K,2)];fig,axs=plt.subplots(2,len(r['labels']),figsize=(11,6),layout='constrained');limit=abs(r['final'][:,0]).max()
        for j,label in enumerate(r['labels']):
            axs[0,j].imshow(r['final'][j,0].T,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[0,2,0,2]);axs[0,j].set(title=str(label),xlabel='x (mm)',ylabel='y (mm)')
            for idx,name in [(1,'(2,0)'),(2,'(0,2)'),(3,'(2,2)')]:plot_mode(axs[1,j],r,j,idx,name)
            axs[1,j].set(xlabel='time (s)',ylabel='E mode magnitude (100 ms RMS)');axs[1,j].legend(fontsize=8)
        fig.colorbar(axs[0,0].images[0],ax=list(axs[0]),label='E activity (dimensionless)',shrink=.8)
        fig.suptitle(f'{K} populations: square torus, transverse perturbations included')
        fig.savefig(out/f'torus_2d_{K}pop.png',dpi=160);plt.close(fig)


def dump(path,data):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,complex):return dict(real=v.real,imag=v.imag)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(data,default=convert,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-only',action='store_true');parser.add_argument('--dim',type=int,choices=[1,2]);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);models={};analyses={};records={};results={}
    for K in [3,4]:
        model=make_model(K);cr,nf,extra=coefficients(model);checks=audit(model,cr,nf,extra)
        models[K]=model;analyses[K]=(cr,nf,extra)
        records[K]=dict(G=model.A_p,Sigma=model.Sigma,tau_ms=model.tau,gain=model.gain,theta=model.theta,L_model=LENGTH,L_mm=2.,
            epsilon=EPSILON,scaled_peaks=dict(temporal=8,spatial=8,combined=[8,4],traveling=8),
            slow_duration=60.,dt_ms=1.,grid_1d=64,grid_2d=32,initial_seed=2026,kappa_c=cr.p0,k0=cr.mode,omega0=cr.omega0,alpha=nf.alpha,b=nf.b,c=nf.c,**extra,**checks)
        print('NEW PARAMETERS',K,cr.p0,cr.mode,cr.omega0,flush=True)
        if args.analysis_only:continue
        for dim in ([args.dim] if args.dim else [1,2]):
            cases=({'none':{},'temporal':{'temporal':8},'spatial':{'spatial':8},
                    'combined':{'temporal':8,'spatial':4}} if dim==1 else
                   {'none':{},'traveling':{'traveling':8}})
            print('RUN',K,dim,flush=True)
            r=integrate(model,cr,cases,dim=dim,n=64 if dim==1 else 32)
            np.savez_compressed(OUT/f'field_{dim}d_{K}pop.npz',**r);results[K,dim]=r
            records[K][f'metrics_{dim}d']=metrics(r,cr);print('DONE',K,dim,records[K][f'metrics_{dim}d'],flush=True)
        dump(OUT/f'parameters_{K}pop.json',records[K])
    dump(OUT/('analysis.json' if args.analysis_only else 'parameters.json'),records)
    if not args.analysis_only and not args.dim:plots(models,analyses,records,results,OUT)

if __name__=='__main__':main()

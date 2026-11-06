"""Resolve frequency extrema versus dominance switches for the LaTeX report."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import brentq
from .v1_extension import make_model,dominant,linear_symbol,eigen_derivative,compute_bifurcation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'simulations/output/v1_extension'

def branches(model,q,p):
    values=np.linalg.eigvals(linear_symbol(model,q,p))
    complex_values=values[values.imag>1e-8]
    real_values=values[abs(values.imag)<1e-8]
    pair=complex_values[np.argmax(complex_values.real)] if len(complex_values) else complex(np.nan,np.nan)
    real=real_values.real.max() if len(real_values) else np.nan
    assert len(complex_values)<=1, 'These diagnostics assume at most one complex pair'
    return pair,real

def scan(model,p):
    qs=np.linspace(0,2.2,4401)
    pairs,reals=map(np.array,zip(*(branches(model,q,p) for q in qs)))
    gap=pairs.real-reals
    switches=[brentq(lambda q:branches(model,q,p)[0].real-branches(model,q,p)[1],qs[i],qs[i+1],xtol=1e-13)
              for i in np.flatnonzero(gap[:-1]*gap[1:]<0)]
    boundaries=[0]+switches+[2.2];intervals=[];stationary=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        pair,real=branches(model,(lo+hi)/2,p)
        if not pair.real>real:continue
        intervals.append([lo,hi]);grid=np.linspace(lo+1e-7,hi-1e-7,1001)
        derivatives=np.array([eigen_derivative(model,q,p).imag for q in grid])
        stationary.extend(brentq(lambda q:eigen_derivative(model,q,p).imag,grid[i],grid[i+1],xtol=1e-13)
                          for i in np.flatnonzero(derivatives[:-1]*derivatives[1:]<0))
    return {'p':p,'dominant_complex_intervals_q_model':intervals,
            'switches_q_per_mm':[10*q for q in switches],
            'pair_frequencies_at_switch':[branches(model,q,p)[0].imag for q in switches],
            'stationary_frequency_q_per_mm':[10*q for q in stationary],
            'stationary_omega':[branches(model,q,p)[0].imag for q in stationary],
            'modes_0_to_14_omega':[dominant(linear_symbol(model,2*np.pi*k/model.length,p)).imag for k in range(15)]}

def edge_sensitivity(model, q, p):
    M = linear_symbol(model, q, p)
    values, left, right = eig(M, left=True, right=True)
    index = np.argmin(abs(values - dominant(M)))
    w, v = left[:, index], right[:, index]
    recurrent = M + np.diag(1 / model.tau)
    contributions = (-q * model.Sigma * recurrent *
                     (np.conj(w)[:, None] * v[None, :] / np.vdot(w, v)).imag)
    assert abs(contributions.sum() - eigen_derivative(model, q, p).imag) < 1e-12
    return dict(contributions=contributions.tolist(),
                positive_sum=float(contributions[contributions > 0].sum()),
                negative_sum=float(contributions[contributions < 0].sum()))

def main(verbose=True):
    data=json.loads((OUT/'calibration.json').read_text())
    models=[make_model(K,data[str(K)]['G']) for K in (3,4)]+[make_model(4)]
    ps=[.25,.25,compute_bifurcation(models[2],p_bounds=(0,.5),max_mode=64,scan_points=101).p0]
    names=['Calibrated E/PV/SST','Calibrated E/PV/SST/VIP','Original E/PV/SST/VIP']
    fig,axes=plt.subplots(3,2,figsize=(10.5,8.5),layout='constrained')
    result={}
    for row,(m,p,name) in enumerate(zip(models,ps,names)):
        info=scan(m,p);result[name]=info
        info['edge_sensitivity_at_mode3'] = edge_sensitivity(m, 6*np.pi/m.length, p)
        q=np.linspace(0,2.2,4401)
        pair,real=map(np.array,zip(*(branches(m,x,p) for x in q)))
        ev=np.array([dominant(linear_symbol(m,x,p)) for x in q])
        axes[row,0].plot(q*10,pair.real,label='oscillatory pair',color='C0')
        axes[row,0].plot(q*10,real,label='rightmost real root',color='C1')
        axes[row,0].axhline(0,color='black',lw=.5)
        axes[row,0].set(title=name,ylabel='real part (1/ms)',ylim=(-.07,.003))
        axes[row,1].plot(q*10,pair.imag,ls='--',color='0.6',label='pair, even when subdominant')
        # Break the curve at dominance switches; draw vertical jump markers separately.
        freq=ev.imag.copy();jumps=np.flatnonzero(abs(np.diff(freq))>.01)
        for i in jumps:freq[i+1]=np.nan
        axes[row,1].plot(q*10,freq,color='C0',lw=2,label='dominant frequency')
        modes=np.arange(15);qm=2*np.pi*modes/m.length
        axes[row,1].scatter(qm*10,info['modes_0_to_14_omega'],s=16,color='black',zorder=5,label='integer modes')
        axes[row,1].set(ylabel='imaginary part (rad/ms)',ylim=(-.005,.115))
        for ax in axes[row]:
            for switch in info['switches_q_per_mm']:ax.axvline(switch,color='0.5',ls=':',lw=.8)
            ax.set(xlabel='angular wavenumber (rad/mm)',xlim=(0,22));ax.grid(alpha=.2)
        if row==0:
            for ax in axes[row]:ax.legend(fontsize=7,loc='lower left')
    fig.savefig(OUT/'dispersion_branches.pdf');fig.savefig(OUT/'dispersion_branches.png',dpi=180);plt.close(fig)
    result['parameter_sweep']={str(K):[scan(models[K-3],p) for p in [.23,.2495,.25,.27]] for K in (3,4)}
    (OUT/'dispersion_detail.json').write_text(json.dumps(result,indent=2)+'\n')
    if verbose: print(json.dumps(result,indent=2))

if __name__=='__main__':main()

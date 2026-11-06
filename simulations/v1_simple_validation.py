"""Independent numerical checks for the new simple examples."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .v1_simple import OUT,make_model,coefficients,integrate,metrics,dump

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-scaling',action='store_true');args=parser.parse_args()
    report={};fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for row,K in enumerate([3,4]):
        model=make_model(K);cr,nf,extra=coefficients(model);rec={};scaling=[]
        for eps in [.01,.03]:
            print('SCALING',K,eps,flush=True)
            path=OUT/f'scaling_{K}pop_eps{eps:.2f}.npz'
            if args.reuse_scaling:
                r=np.load(path)
                for name,value in [('G',model.A_p),('Sigma',model.Sigma),('tau',model.tau),('gain',model.gain),('theta',model.theta)]:
                    np.testing.assert_array_equal(r[name],value)
            else:
                r=integrate(model,cr,{'temporal':{'temporal':8},'subthreshold':{'temporal':4}},epsilon=eps,n=32)
                np.savez_compressed(path,**r)
            met=metrics(r,cr);scaling.append(dict(epsilon=eps,**met));print(met,flush=True)
        main=np.load(OUT/f'field_1d_{K}pop.npz')
        mid=metrics(main,cr)['temporal'];data=sorted([(x['epsilon'],x['temporal']['critical_amplitude']) for x in scaling]+[(.02,mid['critical_amplitude'])])
        rec['scaling']=scaling;rec['amplitude_over_epsilon']=[amp/eps for eps,amp in data]
        # Temporal forcing is a standing wave: combined norm = sqrt(2)*eps*sqrt(rho).
        gamma=nf.b+nf.c;a=-nf.alpha;eta=8*extra['eta_t_unit']
        rho=max(np.roots([abs(gamma)**2,2*(a*gamma.conjugate()).real,abs(a)**2-abs(eta)**2]).real)
        rec['leading_amplitude_over_epsilon']=float(np.sqrt(2*rho))
        axs[row].plot([x for x,y in data],[y for x,y in data],'o-',label='direct field')
        ee=np.array([0,.03]);axs[row].plot(ee,ee*np.sqrt(2*rho),'--',label='leading amplitude equation')
        axs[row].set(title=f'{K} populations',xlabel='epsilon',ylabel='critical-mode amplitude');axs[row].legend();axs[row].grid(alpha=.2)
        initial=main['final'][1:2]
        coarse=integrate(model,cr,{'temporal':{'temporal':8}},n=64,initial=initial,slow_duration=2,dt=1)
        fine=integrate(model,cr,{'temporal':{'temporal':8}},n=64,initial=initial,slow_duration=2,dt=.5)
        rec['dt_halving_relative_final_field']=float(np.linalg.norm(coarse['final']-fine['final'])/np.linalg.norm(fine['final']))
        rec['dt_halving_relative_critical_amplitude']=abs(metrics(coarse,cr)['temporal']['critical_amplitude']/metrics(fine,cr)['temporal']['critical_amplitude']-1)
        # Upsample the same final 2D field spectrally for a short grid-refinement check.
        r2=np.load(OUT/f'field_2d_{K}pop.npz');u=r2['final'][1:2]
        n=u.shape[-1];uh=np.fft.fftshift(np.fft.fft2(u),axes=(-2,-1));pad=np.zeros((*u.shape[:-2],2*n,2*n),complex);pad[...,n//2:3*n//2,n//2:3*n//2]=uh
        up=np.fft.ifft2(np.fft.ifftshift(pad,axes=(-2,-1))).real*4
        c2=integrate(model,cr,{'traveling':{'traveling':8}},dim=2,n=n,initial=u,slow_duration=.2,dt=1)
        f2=integrate(model,cr,{'traveling':{'traveling':8}},dim=2,n=2*n,initial=up,slow_duration=.2,dt=.5)
        rec['grid_time_refinement_relative_final_field']=float(np.linalg.norm(c2['final']-f2['final'][...,::2,::2])/np.linalg.norm(f2['final'][...,::2,::2]))
        assert rec['dt_halving_relative_final_field']<.01 and rec['grid_time_refinement_relative_final_field']<.01
        report[K]=rec;dump(OUT/'validation.json',report);print('VALIDATED',K,rec,flush=True)
    fig.savefig(OUT/'epsilon_scaling.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()

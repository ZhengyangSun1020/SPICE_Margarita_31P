#!/usr/bin/env python3
"""Fit Pi & PCr PER VOXEL (no spatial summing) across all reps — MID00644.

Prior: PK_3T_Skeletal_2Pi.m (AMARES): Pi [4.3,5.1]ppm IV4.85, PCR [-1,1], LW<=50Hz.
For each ticked voxel: per-rep Lorentzian fit (faint) + 5-rep temporal-binned fit
(bold) of that same voxel. Reports Pi amplitude, Pi ppm, and pH per voxel.
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT="/home/zsun/SPICE_Margarita"
DATA=os.path.join(ROOT,"output_dynamic_3T_MID00644_10x10_5rings_dft_adj/data")
LARMOR=49.895; SPEC_DWELL=3.6e-4
MVOX=[(4,3),(4,4),(4,5),(4,6),(5,3),(5,4),(5,5)]
LW_MAX_PPM=50.0/LARMOR
PI_CS=(4.3,5.1,4.85); PCR_CS=(-1.0,1.0,0.0)
PI_W=(3.6,6.4); PCR_W=(-1.8,1.8); BIN=5

def lor(x,A,x0,w,b0,b1): return A*(w/2)**2/((x-x0)**2+(w/2)**2)+b0+b1*x
def fit_peak(ppm,z,win,cs):
    m=(ppm>=win[0])&(ppm<=win[1]); x=ppm[m]; y=z[m]
    b=np.median(y); A0=max(y.max()-b,1e-9); lo,hi,x0=cs
    try:
        p,_=curve_fit(lor,x,y,p0=[A0,x0,0.3,b,0],
                      bounds=([0,lo,0.08,-np.inf,-np.inf],[A0*5,hi,LW_MAX_PPM,np.inf,np.inf]),maxfev=8000)
        return p[0],p[1]
    except Exception:
        i=np.argmax(y); return y[i]-b, x[i]
def pH(s): d=np.clip(s,3.29,5.67); return 6.75+np.log10((d-3.27)/(5.69-d))

def main():
    reps=sorted(int(f.split("rep")[1].split(".")[0])
                for f in os.listdir(DATA) if f.startswith("iter3d_recon_spec_rep"))
    a0=np.load(os.path.join(DATA,f"iter3d_recon_spec_rep{reps[0]:02d}.npy"))
    ns=a0.shape[-1]; PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=SPEC_DWELL))/LARMOR
    reps=np.array(reps)
    # per-voxel complex spectra over reps: zc[v] = (n_rep, ns)
    zc={v:[] for v in MVOX}
    for r in reps:
        a=np.load(os.path.join(DATA,f"iter3d_recon_spec_rep{r:02d}.npy"))[:, :, 0, :]
        for v in MVOX: zc[v].append(a[v[0],v[1]])
    zc={v:np.array(zc[v]) for v in MVOX}

    cols=plt.cm.tab10(np.linspace(0,1,len(MVOX)))
    fig,axs=plt.subplots(3,1,figsize=(13,11),sharex=True)
    print(f"{'vox':>7} {'rep~':>5} {'Pi_amp':>10} {'Pi_ppm':>7} {'pH':>6}")
    for vi,v in enumerate(MVOX):
        # per-rep (faint)
        pa=[];pp=[];ph=[]
        for k in range(len(reps)):
            z=np.abs(zc[v][k]); A,X=fit_peak(PPM,z,PI_W,PI_CS); _,cc=fit_peak(PPM,z,PCR_W,PCR_CS)
            pa.append(A);pp.append(X);ph.append(pH(X-cc))
        # binned per voxel (bold)
        bc=[];ba=[];bp=[];bph=[]
        for s in range(0,len(reps),BIN):
            e=min(s+BIN,len(reps)); zb=np.abs(zc[v][s:e].sum(0))
            A,X=fit_peak(PPM,zb,PI_W,PI_CS); _,cc=fit_peak(PPM,zb,PCR_W,PCR_CS)
            bc.append(reps[s:e].mean());ba.append(A/(e-s));bp.append(X);bph.append(pH(X-cc))
        bc=np.array(bc)
        lab=f"[{v[0]},{v[1]}]"
        axs[0].plot(reps,pa,'.',color=cols[vi],alpha=0.2); axs[0].plot(bc,ba,'-o',color=cols[vi],lw=1.6,label=lab)
        axs[1].plot(bc,bp,'-o',color=cols[vi],lw=1.6,label=lab)
        axs[2].plot(bc,bph,'-o',color=cols[vi],lw=1.6,label=lab)
        for i in range(len(bc)):
            print(f"{lab:>7} {bc[i]:5.0f} {ba[i]:10.2e} {bp[i]:7.2f} {bph[i]:6.2f}")
    axs[0].set_ylabel("Pi amplitude"); axs[0].set_title("Pi amplitude per voxel (dots=per-rep, line=5-rep binned)"); axs[0].legend(fontsize=7,ncol=4)
    axs[1].set_ylabel("Pi ppm"); axs[1].axhline(4.85,color='k',ls='--',lw=0.5,alpha=0.5); axs[1].set_ylim(5.2,4.2); axs[1].set_title("Pi peak ppm per voxel (AMARES bounds 4.3-5.1)")
    axs[2].set_ylabel("pH"); axs[2].set_xlabel("rep"); axs[2].set_title("intracellular pH per voxel")
    fig.suptitle("MID00644 — per-voxel Pi fit (no summing), AMARES prior",fontsize=13)
    fig.tight_layout()
    outp=os.path.join(ROOT,"_pi_fit_pervox.png"); fig.savefig(outp,dpi=140); plt.close(fig)
    print("saved ->",outp)

if __name__=="__main__": main()

"""Compare Python DCF-adjoint (MATLAB direct-FT equivalent) vs iterative CG, rep30."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="/home/zsun/SPICE_Margarita/output_dynamic_3T_0724_5rings_06c/data"
LARMOR=49.895; SPEC_DWELL=3.6e-4
VOX=[(4,5),(4,4),(3,5)]
def FIDToSpec(x): return np.fft.fftshift(np.fft.fft(x,axis=-1),axes=-1)
rec=np.load(os.path.join(D,"iter3d_recon_spec_rep30.npy"))[:,:,0,:]           # iterative (already spec)
adf=np.load(os.path.join(D,"iter3d_adj_dcf_fid_rep30.npy"))                    # adjoint fid
adf=adf[:,:,0,:] if adf.ndim==4 else adf
adj=FIDToSpec(adf.conj())                                                      # adjoint spec
ns=rec.shape[-1]
PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=SPEC_DWELL))/LARMOR
peaks=[(0,'PCr'),(4.85,'Pi'),(-2.53,'γ'),(-7.58,'α'),(-15.86,'β')]
noise=(PPM>=12)&(PPM<=25); pcr=(PPM>=-1)&(PPM<=1)
def snr(z): return z[pcr].max()/(z[noise].std()+1e-12)
fig,axs=plt.subplots(len(VOX),1,figsize=(12,3.6*len(VOX)),squeeze=False)
for i,(xi,yi) in enumerate(VOX):
    ax=axs[i][0]
    zi=np.abs(rec[xi,yi]); za=np.abs(adj[xi,yi])
    zi/=zi.max(); za/=za.max()
    ax.plot(PPM,za,'g-',lw=1.1,label=f"DCF-adjoint (SNR {snr(np.abs(adj[xi,yi])):.1f})")
    ax.plot(PPM,zi,'m-',lw=0.9,alpha=0.8,label=f"iterative CG (SNR {snr(np.abs(rec[xi,yi])):.1f})")
    for pos,name in peaks:
        ax.axvline(pos,color='k',ls=':',lw=0.5,alpha=0.4)
        ax.text(pos,1.0,name,fontsize=8,ha='center',va='top',rotation=90)
    ax.set_xlim(PPM.max(),PPM.min()); ax.set_ylim(0,1.05)
    ax.set_xlabel("ppm"); ax.set_ylabel("norm |spec|")
    ax.set_title(f"voxel [{xi},{yi}] rep30 — adjoint(green) vs iterative(magenta)")
    ax.legend(fontsize=8)
fig.tight_layout()
outp="/home/zsun/SPICE_Margarita/_cmp_adj_iter.png"; fig.savefig(outp,dpi=140); plt.close(fig)
print("saved ->",outp)

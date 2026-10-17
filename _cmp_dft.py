"""Compare NUFFT vs DFT 06c recon_spec (rep30), correct -freq/larmor axis."""
import os, importlib.util as ilu
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/home/zsun/SPICE_Margarita"
NUFFT=os.path.join(ROOT,"output_dynamic_3T_0724_5rings_06c/data/iter3d_recon_spec_rep30.npy")
DFT  =os.path.join(ROOT,"output_dynamic_3T_0724_5rings_06c_DFT/data/iter3d_recon_spec_rep30.npy")
LARMOR=49.895; SPEC_DWELL=3.6e-4
VOX=[(4,5),(4,4),(3,5)]

a=np.load(NUFFT); b=np.load(DFT)
a=a[:,:,0,:]; b=b[:,:,0,:]
ns=a.shape[-1]
PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=SPEC_DWELL))/LARMOR
peaks=[(0,'PCr'),(4.85,'Pi'),(-2.53,'γ'),(-7.58,'α'),(-15.86,'β')]

# global correlation over all voxels (magnitude)
ma=np.abs(a).ravel(); mb=np.abs(b).ravel()
corr=np.abs(np.vdot(ma-ma.mean(),mb-mb.mean()))/(np.linalg.norm(ma-ma.mean())*np.linalg.norm(mb-mb.mean())+1e-12)
scale=np.linalg.norm(ma)/(np.linalg.norm(mb)+1e-12)
print(f"global |spec| corr(NUFFT,DFT)={corr:.4f}  amp ratio NUFFT/DFT={scale:.3f}")

fig,axs=plt.subplots(len(VOX),1,figsize=(12,3.6*len(VOX)),squeeze=False)
for i,(xi,yi) in enumerate(VOX):
    ax=axs[i][0]
    za=np.abs(a[xi,yi]); zb=np.abs(b[xi,yi])*scale   # rescale DFT to NUFFT amplitude
    ax.plot(PPM,za,'b-',lw=1.0,label="NUFFT")
    ax.plot(PPM,zb,'r--',lw=1.0,label="DFT (scaled)")
    top=max(za.max(),zb.max())*1.05
    for pos,name in peaks:
        ax.axvline(pos,color='k',ls=':',lw=0.5,alpha=0.4)
        ax.text(pos,top*0.96,name,fontsize=8,ha='center',va='top',rotation=90)
    ax.set_xlim(PPM.max(),PPM.min()); ax.set_ylim(0,top)
    ax.set_xlabel("ppm"); ax.set_ylabel("|spec|")
    c=np.corrcoef(za,zb)[0,1]
    ax.set_title(f"voxel [{xi},{yi}]  rep30  |  NUFFT vs DFT  (corr={c:.4f})")
    ax.legend(fontsize=8)
fig.tight_layout()
outp=os.path.join(ROOT,"_cmp_dft_nufft.png"); fig.savefig(outp,dpi=140); plt.close(fig)
print("saved ->",outp)

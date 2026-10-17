"""Per-voxel data+fit+residual spectra from fsl_mrsi output (MID00276 80-ring)."""
import os, numpy as np, nibabel as nib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/home/zsun/SPICE_Margarita/output_80rings_MID00276_fit31p"
LARMOR=49.895; BW=2778.0
def load_c(p):
    a=nib.load(p).get_fdata(dtype=np.complex64) if nib.load(p).get_data_dtype().kind=='c' \
      else np.asarray(nib.load(p).dataobj)
    return np.asarray(a)
data=np.asarray(nib.load(os.path.join(OUT,"spice31p.nii.gz")).dataobj)
fit =np.asarray(nib.load(os.path.join(OUT,"fit","fit","fit.nii.gz")).dataobj)
base=np.asarray(nib.load(os.path.join(OUT,"fit","fit","baseline.nii.gz")).dataobj)
mask=np.asarray(nib.load(os.path.join(OUT,"mask.nii.gz")).dataobj)
print("data",data.shape,data.dtype,"fit",fit.shape)
data=data[:,:,0,:]; fit=fit[:,:,0,:]; base=base[:,:,0,:]; mask=np.squeeze(mask).astype(bool)
ns=data.shape[-1]
PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=1.0/BW))/LARMOR
def spec(fid): return np.fft.fftshift(np.fft.fft(np.conj(fid),axis=-1),axes=-1)
D=spec(data); F=spec(fit); B=spec(base)
win=(PPM>=-20)&(PPM<=10); xw=PPM[win]
vox=np.argwhere(mask)
peaks=[(0,'PCr'),(4.85,'Pi'),(-2.53,'γ'),(-7.58,'α'),(-15.86,'β')]
nc=min(3,len(vox)); nr=int(np.ceil(len(vox)/nc)) or 1
fig,axs=plt.subplots(nr,nc,figsize=(4.5*nc,3.0*nr),squeeze=False)
for k,(xi,yi) in enumerate(vox):
    ax=axs[k//nc][k%nc]
    d=D[xi,yi,win].real; f=F[xi,yi,win].real
    ax.plot(xw,d,'k-',lw=0.7,label="data")
    ax.plot(xw,f,'r-',lw=1.1,label="fit")
    off=-1.15*np.abs(d).max()
    ax.plot(xw,(d-f)+off,'g-',lw=0.5,label="resid")
    for pos,nm in peaks: ax.axvline(pos,color='b',ls=':',lw=0.4,alpha=0.4)
    ax.set_xlim(10,-20); ax.set_title(f"voxel [{xi},{yi}]",fontsize=9)
    ax.set_xlabel("ppm"); ax.legend(fontsize=6)
for k in range(len(vox),nr*nc): axs[k//nc][k%nc].axis('off')
fig.suptitle("MID00276 80-ring: data(black) / fit(red) / residual(green)",fontsize=12)
fig.tight_layout()
outp="/home/zsun/SPICE_Margarita/_fitgrid.png"; fig.savefig(outp,dpi=140); plt.close(fig)
print("saved ->",outp)

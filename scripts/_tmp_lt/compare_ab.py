#!/usr/bin/env python3
"""A/B compare: 31P recon with vs without 1H-navigator 0th-order drift correction."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A=np.load("/tmp/lt_nodrift/data/recon_spec.npy")   # (Nx,Ny,Nz,n_spec) uncorrected
B=np.load("/tmp/lt_drift/data/recon_spec.npy")     # corrected
OUT=os.path.dirname(os.path.abspath(__file__))
Nx,Ny,Nz,ns=A.shape
BW=2778.0; larmor=120.315
PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=1.0/BW))/larmor
def win(lo,hi): return (PPM>=lo)&(PPM<=hi)
pcr=win(-1.2,1.2); noisew=win(12,20)
def metrics(V):
    mag=np.abs(V)
    peak=mag[...,pcr].max(-1)                      # PCr peak per voxel
    noise=mag[...,noisew].std(-1)+1e-30
    return peak, peak/noise
pkA,snrA=metrics(A); pkB,snrB=metrics(B)
# best voxel from uncorrected PCr SNR
vx,vy,vz=np.unravel_index(np.argmax(snrA),snrA.shape)
print(f"best voxel (uncorr) [{vx},{vy},{vz}]  SNR uncorr={snrA[vx,vy,vz]:.1f} corr={snrB[vx,vy,vz]:.1f}")

def fwhm(mag1d):
    m=mag1d.copy(); i0=np.argmax(m*pcr); pk=m[i0]
    half=pk/2; l=i0
    while l>0 and m[l]>half: l-=1
    r=i0
    while r<ns-1 and m[r]>half: r+=1
    return abs(PPM[r]-PPM[l])*larmor   # Hz
sA=np.abs(A[vx,vy,vz]); sB=np.abs(B[vx,vy,vz])
print(f"PCr peak: uncorr={sA[pcr].max():.3e} corr={sB[pcr].max():.3e}  ratio={sB[pcr].max()/sA[pcr].max():.3f}")
print(f"PCr FWHM Hz: uncorr={fwhm(sA):.1f} corr={fwhm(sB):.1f}")

# global metrics over voxels with decent signal
mask=snrA>np.percentile(snrA,80)
print(f"\nGlobal over top-20% voxels (n={mask.sum()}):")
print(f"  median PCr SNR: uncorr={np.median(snrA[mask]):.2f} corr={np.median(snrB[mask]):.2f}  "
      f"(+{100*(np.median(snrB[mask])/np.median(snrA[mask])-1):.1f}%)")
print(f"  sum PCr peak:   uncorr={pkA[mask].sum():.3e} corr={pkB[mask].sum():.3e}  "
      f"(+{100*(pkB[mask].sum()/pkA[mask].sum()-1):.1f}%)")
print(f"  total |PCr| all voxels: uncorr={pkA.sum():.3e} corr={pkB.sum():.3e} (+{100*(pkB.sum()/pkA.sum()-1):.1f}%)")

# figure
fig,ax=plt.subplots(1,3,figsize=(16,4.5))
ax[0].plot(PPM,sA,label=f"uncorrected (SNR {snrA[vx,vy,vz]:.0f})",color='gray')
ax[0].plot(PPM,sB,label=f"drift-corrected (SNR {snrB[vx,vy,vz]:.0f})",color='crimson')
ax[0].set_xlim(20,-20); ax[0].set_xlabel("ppm"); ax[0].set_ylabel("|spectrum|")
ax[0].set_title(f"best voxel [{vx},{vy},{vz}]"); ax[0].legend(fontsize=8)
for p,n in [(0,'PCr'),(5,'Pi'),(-2.5,'g'),(-7.5,'a'),(-16,'b')]:
    ax[0].axvline(p,color='b',ls=':',lw=0.5,alpha=0.5)
m0=pkA[:,:,vz]; m1=pkB[:,:,vz]; vmax=max(m0.max(),m1.max())
im=ax[1].imshow(np.flipud(m0.T),vmin=0,vmax=vmax,cmap='viridis'); ax[1].set_title(f"PCr map uncorr (z={vz})"); ax[1].axis('off')
ax[2].imshow(np.flipud(m1.T),vmin=0,vmax=vmax,cmap='viridis'); ax[2].set_title("PCr map drift-corr"); ax[2].axis('off')
fig.colorbar(im,ax=ax[1:].tolist(),fraction=0.02)
fig.savefig(os.path.join(OUT,"g_ab_compare.png"),dpi=150,bbox_inches='tight'); print("saved g_ab_compare.png")

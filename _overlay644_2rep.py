#!/usr/bin/env python3
"""rep0 (rest) vs rep26 (nadir) spectra on the MID00644 localizer, shared scale."""
import os, glob, importlib.util as ilu
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pydicom

ROOT="/home/zsun/SPICE_Margarita"
DAT="/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_crt_10_10/meas_MID00644_FID214692_LV_crt_dynamic_10x10_5rings.dat"
LOCDIR="/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_crt_10_10"
DATADIR=os.path.join(ROOT,"output_dynamic_3T_MID00644_10x10_5rings_dft_adj/data")
REPA, REPB = 0, 26
COLA, COLB = "#39ff14", "#ff4444"     # rep0 green, rep26 red
LARMOR=49.895; SPEC_DWELL=3.6e-4
PPM_LO,PPM_HI=-8.0,8.0

rs_spec=ilu.spec_from_file_location("rs",os.path.join(ROOT,"scripts","roi_selector_31p.py"))
rs=ilu.module_from_spec(rs_spec); rs_spec.loader.exec_module(rs)
lspec=ilu.spec_from_file_location("_06c",os.path.join(ROOT,"scripts","06c_iterative_recon_31p_3d.py"))
lm=ilu.module_from_spec(lspec); lspec.loader.exec_module(lm)

info=lm.load_concept_twix_3d(DAT)
N=int(info["matrix_size"]); fov=float(info["fov_mm"]); vox=fov/N
ps=float(info["pos_sag"]); pc=float(info["pos_cor"]); S_mrsi=float(info["pos_tra"])

planes=[]
for f in glob.glob(os.path.join(LOCDIR,"*.dcm")):
    try:
        d=pydicom.dcmread(f,stop_before_pixels=True); iop=np.array(d.ImageOrientationPatient,float)
        if abs(abs(iop[0])-1)<0.2 and abs(abs(iop[4])-1)<0.2:
            planes.append((float(d.ImagePositionPatient[2]),f))
    except Exception: pass
chosen_s,chosen=min(planes,key=lambda t:abs(t[0]-S_mrsi))
print(f"localizer slice S={chosen_s:.1f} (MRSI S={S_mrsi:.1f})")
img,extent,ds=rs.load_localizer_with_geometry(chosen)

def load(rep):
    a=np.load(os.path.join(DATADIR,f"iter3d_recon_spec_rep{rep:02d}.npy"))
    if a.ndim==4: a=a[:,:,a.shape[2]//2,:]
    return np.abs(a).transpose(1,0,2)                 # roi_selector frame [xi(L),yi(P),ns]
A=load(REPA); B=load(REPB); ns=A.shape[-1]
PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=SPEC_DWELL))/LARMOR
win=(PPM>=PPM_LO)&(PPM<=PPM_HI); xw=PPM[win]; span=max(win.sum()-1,1)
g=max(A[:,:,win].max(), B[:,:,win].max())+1e-12      # SHARED scale -> drop is visible

fig,ax=plt.subplots(figsize=(11,11))
ax.imshow(img,cmap='gray',origin='lower',extent=extent,aspect='equal')
ax.set_xlim(ps+fov/2, ps-fov/2) if extent[0]>extent[1] else ax.set_xlim(ps-fov/2, ps+fov/2)
ax.set_ylim(pc+fov/2, pc-fov/2) if extent[2]>extent[3] else ax.set_ylim(pc-fov/2, pc+fov/2)
x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
sgnx=1 if x1>x0 else -1; sgny=1 if y1>y0 else -1
for xi in range(N):
    for yi in range(N):
        L,P=rs.voxel_to_lps(xi,yi,N,ps,pc,vox)
        xr=L + sgnx*(-vox/2 + np.arange(len(xw))/span*vox*0.92 + vox*0.04)
        yb=P - sgny*vox/2
        ax.plot(xr, yb + sgny*(A[xi,yi,win]/g)*0.9*vox, color=COLA, lw=0.6)
        ax.plot(xr, yb + sgny*(B[xi,yi,win]/g)*0.9*vox, color=COLB, lw=0.6)
        ax.add_patch(plt.Rectangle((L-vox/2,P-vox/2),vox,vox,fill=False,ec='cyan',lw=0.25,alpha=0.3))
ax.plot([],[],color=COLA,label=f"rep{REPA} (rest)"); ax.plot([],[],color=COLB,label=f"rep{REPB} (nadir)")
ax.legend(loc='upper right',fontsize=9,facecolor='k',labelcolor='w')
ax.set_title(f"MID00644 rep{REPA}(green) vs rep{REPB}(red) on localizer | shared scale | PCr-centered +/-8ppm",
             fontsize=9,color='w')
ax.set_facecolor('k'); ax.tick_params(colors='w')
outp=os.path.join(ROOT,"_overlay644_2rep.png")
fig.savefig(outp,dpi=150,facecolor='k',bbox_inches='tight'); plt.close(fig)
print("saved ->",outp)

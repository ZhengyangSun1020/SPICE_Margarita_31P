#!/usr/bin/env python3
"""rep0 spectra map of MID00644 overlaid on its localizer (roi_selector geometry)."""
import os, glob, importlib.util as ilu
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pydicom

ROOT="/home/zsun/SPICE_Margarita"
DAT="/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_crt_10_10/meas_MID00644_FID214692_LV_crt_dynamic_10x10_5rings.dat"
LOCDIR="/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_crt_10_10"
SPEC=os.path.join(ROOT,"output_dynamic_3T_MID00644_10x10_5rings_dft_adj/data/iter3d_recon_spec_rep00.npy")
LARMOR=49.895; SPEC_DWELL=3.6e-4
PPM_LO,PPM_HI=-15.0,15.0   # PCr (0 ppm) centered, +/-15 ppm

rs_spec=ilu.spec_from_file_location("rs",os.path.join(ROOT,"scripts","roi_selector_31p.py"))
rs=ilu.module_from_spec(rs_spec); rs_spec.loader.exec_module(rs)

# full MRSI geometry incl. through-plane S (pos_tra) via the 06c loader
lspec=ilu.spec_from_file_location("_06c",os.path.join(ROOT,"scripts","06c_iterative_recon_31p_3d.py"))
lm=ilu.module_from_spec(lspec); lspec.loader.exec_module(lm)
info=lm.load_concept_twix_3d(DAT)
N=int(info["matrix_size"]); fov=float(info["fov_mm"]); vox=fov/N
ps=float(info["pos_sag"]); pc=float(info["pos_cor"]); S_mrsi=float(info["pos_tra"])
print(f"MRSI:  N={N} FOV={fov}mm vox={vox:.1f}  center L={ps:.1f} P={pc:.1f} S(tra)={S_mrsi:.1f}")
print(f"MRSI grid box:  L[{ps-fov/2:.1f}, {ps+fov/2:.1f}]  P[{pc-fov/2:.1f}, {pc+fov/2:.1f}]")

# pick transverse localizer slice whose S is nearest the MRSI slab center
planes=[]
for f in glob.glob(os.path.join(LOCDIR,"*.dcm")):
    try:
        d=pydicom.dcmread(f,stop_before_pixels=True)
        iop=np.array(d.ImageOrientationPatient,float)
        if abs(abs(iop[0])-1)<0.2 and abs(abs(iop[4])-1)<0.2:   # transverse
            planes.append((float(d.ImagePositionPatient[2]),f))
    except Exception: pass
planes.sort()
S_list=[s for s,_ in planes]
chosen_s,chosen=min(planes,key=lambda t:abs(t[0]-S_mrsi))
print(f"Loc:   {len(planes)} transverse slices, S range [{S_list[0]:.1f}, {S_list[-1]:.1f}]")
print(f"       MRSI S={S_mrsi:.1f} -> nearest localizer slice S={chosen_s:.1f} ({os.path.basename(chosen)})  |dS|={abs(chosen_s-S_mrsi):.1f}mm")
img,extent,ds=rs.load_localizer_with_geometry(chosen)
locL=(extent[0]+extent[1])/2; locP=(extent[2]+extent[3])/2
print(f"Loc extent:  L[{min(extent[0],extent[1]):.1f}, {max(extent[0],extent[1]):.1f}]  P[{min(extent[2],extent[3]):.1f}, {max(extent[2],extent[3]):.1f}]  (FOV can exceed MRSI)")
print(f"CENTER CHECK: MRSI center (L,P)=({ps:.1f},{pc:.1f})  localizer-image center=({locL:.1f},{locP:.1f})  -> registration uses shared LPS world frame")

# recon spec rep0 -> roi_selector frame (native is transposed vs voxel_to_lps xi->L,yi->P)
a=np.load(SPEC)
if a.ndim==4: a=a[:,:,a.shape[2]//2,:]
spec=np.abs(a).transpose(1,0,2)                       # -> [xi(col/L), yi(row/P), ns]
ns=spec.shape[-1]
PPM=-np.fft.fftshift(np.fft.fftfreq(ns,d=SPEC_DWELL))/LARMOR
win=(PPM>=PPM_LO)&(PPM<=PPM_HI); xw=PPM[win]; span=max(win.sum()-1,1)
g=spec[:,:,win].max()+1e-12

fig,ax=plt.subplots(figsize=(11,11))
ax.imshow(img,cmap='gray',origin='lower',extent=extent,aspect='equal')
# zoom to MRSI grid box centered on MRSI center
ax.set_xlim(ps+fov/2, ps-fov/2) if extent[0]>extent[1] else ax.set_xlim(ps-fov/2, ps+fov/2)
ax.set_ylim(pc+fov/2, pc-fov/2) if extent[2]>extent[3] else ax.set_ylim(pc-fov/2, pc+fov/2)
x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
sgnx=1 if x1>x0 else -1; sgny=1 if y1>y0 else -1
for xi in range(N):
    for yi in range(N):
        L,P=rs.voxel_to_lps(xi,yi,N,ps,pc,vox)
        s=spec[xi,yi,win]/g
        xr=L + sgnx*(-vox/2 + np.arange(len(xw))/span*vox*0.92 + vox*0.04)
        yb=P - sgny*vox/2
        ax.plot(xr, yb + sgny*s*0.9*vox, color="#39ff14", lw=0.5)
        ax.add_patch(plt.Rectangle((L-vox/2,P-vox/2),vox,vox,fill=False,ec='cyan',lw=0.25,alpha=0.3))
ax.set_title(f"MID00644 rep0 spectra on localizer (roi_selector geom)  ppm[{PPM_HI},{PPM_LO}]",
             fontsize=10,color='w')
ax.set_facecolor('k'); ax.tick_params(colors='w')
outp=os.path.join(ROOT,"_overlay644_rep0.png")
fig.savefig(outp,dpi=150,facecolor='k',bbox_inches='tight'); plt.close(fig)
print("saved ->",outp)

import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pydicom
from scipy.ndimage import map_coordinates

OUT=sys.argv[1] if len(sys.argv)>1 else "/home/zsun/SPICE_Margarita/output_7T_LiverYang_MID00050_70R_spice_roi"
FOV_ARG=float(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2] else None
D=OUT+"/data"; NFINE=120

def to_native(a): return np.rot90(np.flip(a, axis=1), 1, axes=(0,1))
adj=to_native(np.abs(np.load(f"{D}/adj_spec.npy")))
spc=to_native(np.abs(np.load(f"{D}/spice3d_spec_rank12.npy")))
blocky=np.load(f"{D}/localizer_spatial_prior_native_3d.npy").astype(float)
Nx,Ny,Nz,ns=adj.shape

LOCDIR=sys.argv[3] if len(sys.argv)>3 else None
mapping=[]
for line in open(OUT+"/localizer_slice_mapping.txt"):
    if line.startswith("MRSI z") or "|" not in line: continue
    p=[s.strip() for s in line.split("|")]
    mapping.append((int(p[0]), [float(x) for x in p[3].split(",")], p[5]))
mapping.sort()

if LOCDIR:   # re-select nearest plane per MRSI slice from a given localizer folder
    planes=[]
    for f in os.listdir(LOCDIR):
        try:
            ds=pydicom.dcmread(os.path.join(LOCDIR,f),stop_before_pixels=True)
            iop=np.array(ds.ImageOrientationPatient,float); n=np.cross(iop[:3],iop[3:]); n/=np.linalg.norm(n)
            ipp=np.array(ds.ImagePositionPatient,float)
            planes.append((os.path.join(LOCDIR,f), ipp, n))
        except: pass
    rows=[]
    for z,c,_ in mapping:
        cc=np.array(c)
        best=min(planes, key=lambda pl: abs(float(np.dot(cc-pl[1], pl[2]))))
        dist=abs(float(np.dot(cc-best[1], best[2])))
        rows.append((z,c,best[0])); print(f"z{z}: nearest plane dist={dist:.1f}mm")
    print(f"[using localizer dir: {os.path.basename(LOCDIR)}]")
else:
    rows=mapping
dsc={p:pydicom.dcmread(p) for _,_,p in rows}   # cache DICOMs

def sample_fine(path, center, FOV):
    ds=dsc[path]; img=ds.pixel_array.astype(float)
    iop=np.array(ds.ImageOrientationPatient,float)
    rd=iop[:3]/np.linalg.norm(iop[:3]); cd=iop[3:]/np.linalg.norm(iop[3:])
    ipp=np.array(ds.ImagePositionPatient,float)
    spr=float(ds.PixelSpacing[0]); spc_=float(ds.PixelSpacing[1])
    cL,cP,cS=center
    Ls=np.linspace(cL-FOV/2,cL+FOV/2,NFINE); Ps=np.linspace(cP-FOV/2,cP+FOV/2,NFINE)
    LL,PP=np.meshgrid(Ls,Ps,indexing='ij'); world=np.stack([LL,PP,np.full_like(LL,cS)],-1)
    delta=world-ipp; rr=(delta@rd)/spr; cc=(delta@cd)/spc_
    return map_coordinates(img,[rr.ravel(),cc.ravel()],order=1,mode='constant',cval=0).reshape(NFINE,NFINE)

def dih(m,k): return np.rot90(m,k) if k<4 else np.rot90(np.fliplr(m),k-4)
def down(m):
    b=m.shape[0]//Nx; return m[:Nx*b,:Ny*b].reshape(Nx,b,Ny,b).mean((1,3))
def corr(a,b):
    a=a.ravel()-a.mean(); b=b.ravel()-b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def score(FOV):
    fine={z:sample_fine(p,c,FOV) for z,c,p in rows}
    best=np.full(8,0.0)
    for z,c,p in rows:
        bl=blocky[:,:,z]
        if bl.std()<1e-6: continue
        d=down(fine[z])
        for k in range(8): best[k]+=corr(dih(d,k),bl)
    return best.max(), int(best.argmax()), fine

if FOV_ARG is not None:
    bestFOV=FOV_ARG; sc,bestk,fine=score(bestFOV)
else:
    cand=list(range(180,381,10)); res=[(score(f)[0],f) for f in cand]
    bestFOV=max(res)[1]; sc,bestk,fine=score(bestFOV)
    print("FOV scan:", [(f,round(s,1)) for s,f in res])
print(f"chosen FOV={bestFOV}mm  dihedral k={bestk}  corr={sc:.2f}")

BW=2778.0; larmor=120.6
ppm=-np.fft.fftshift(np.fft.fftfreq(ns,d=1/BW))/larmor
env=np.maximum(adj.max(axis=(0,1,2))/adj.max(), spc.max(axis=(0,1,2))/spc.max())
idx=np.where(env>0.03*env.max())[0]; lo,hi=max(0,idx.min()-15),min(ns,idx.max()+15)
xw=ppm[lo:hi]; xlim=(xw.max(),xw.min())
g=max(adj[...,lo:hi].max(), spc[...,lo:hi].max())+1e-12
outdir=OUT+"/adjspice_on_realimage"; os.makedirs(outdir,exist_ok=True)
for z,c,p in rows:
    fa=dih(fine[z],bestk)[::-1,::-1]; fa=fa/(np.percentile(fa,99.0)+1e-9)
    fig=plt.figure(figsize=(Ny*1.6,Nx*1.25),facecolor='k')
    axb=fig.add_axes([0,0,1,1]); axb.imshow(np.clip(fa,0,1),cmap='gray',origin='lower',extent=[0,Ny,0,Nx])
    axb.set_xlim(0,Ny); axb.set_ylim(0,Nx); axb.axis('off')
    pad=0.025; vw,vh=1.0/Ny,1.0/Nx
    for yi in range(Ny):
        for xi in range(Nx):
            ax=fig.add_axes([yi*vw+pad*vw, xi*vh+pad*vh, vw*(1-2*pad), vh*(1-2*pad)])
            ax.plot(xw,adj[xi,yi,z,lo:hi]/g,color="#ff3b3b",lw=0.5)
            ax.plot(xw,spc[xi,yi,z,lo:hi]/g,color="#2ea1ff",lw=0.6)
            ax.set_xlim(xlim); ax.set_ylim(0,1.02); ax.patch.set_alpha(0); ax.axis('off')
    fig.suptitle(f"{os.path.basename(OUT)} | adj(red) vs SPICE(blue) on REAL localizer | shared y | slice {z+1}/{Nz}",color='w',fontsize=9)
    fig.savefig(f"{outdir}/real_z{z:02d}.png",dpi=150,facecolor='k'); plt.close(fig); print("z",z)
print("OUTDIR",outdir)

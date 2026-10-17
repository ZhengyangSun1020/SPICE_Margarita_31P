import os, sys, importlib.util as ilu, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT=sys.argv[1]; DAT=sys.argv[2]
sc=ilu.spec_from_file_location("rs","/home/zsun/SPICE_Margarita/scripts/roi_selector_31p.py")
rs=ilu.module_from_spec(sc); sc.loader.exec_module(rs)
N,ps,pc,fov=rs.load_mrsi_geometry(DAT); vox=fov/N
print(f"N={N} center L={ps:.1f} P={pc:.1f} FOV={fov} vox={vox}")

D=OUT+"/data"
def to_native(a): return np.rot90(np.flip(a,axis=1),1,axes=(0,1))
adj=to_native(np.abs(np.load(f"{D}/adj_spec.npy"))).transpose(1,0,2,3)  # transpose to roi_selector (xi->L, yi->P) frame
spc=to_native(np.abs(np.load(f"{D}/spice3d_spec_rank12.npy"))).transpose(1,0,2,3)
Nx,Ny,Nz,ns=adj.shape

rows=[]
for line in open(OUT+"/localizer_slice_mapping.txt"):
    if line.startswith("MRSI z") or "|" not in line: continue
    p=[s.strip() for s in line.split("|")]; rows.append((int(p[0]), p[5]))
rows.sort()

BW=2778.0; larmor=120.6
ppm=-np.fft.fftshift(np.fft.fftfreq(ns,d=1/BW))/larmor
env=np.maximum(adj.max(axis=(0,1,2))/adj.max(), spc.max(axis=(0,1,2))/spc.max())
idx=np.where(env>0.03*env.max())[0]; lo,hi=max(0,idx.min()-15),min(ns,idx.max()+15)
xw=ppm[lo:hi]; span=max(hi-lo-1,1); g=max(adj[...,lo:hi].max(),spc[...,lo:hi].max())+1e-12
outdir=OUT+"/adjspice_on_realimage_v2"; os.makedirs(outdir,exist_ok=True)
zonly=[int(x) for x in sys.argv[3:]] if len(sys.argv)>3 else [z for z,_ in rows]

for z,path in rows:
    if z not in zonly: continue
    img,extent,_=rs.load_localizer_with_geometry(path)
    fig,ax=plt.subplots(figsize=(11,11)); ax.imshow(img,cmap='gray',origin='lower',extent=extent,aspect='equal')
    # zoom to MRSI grid box (centered on MRSI center)
    ax.set_xlim(ps+fov/2, ps-fov/2) if extent[0]>extent[1] else ax.set_xlim(ps-fov/2, ps+fov/2)
    ax.set_ylim(pc+fov/2, pc-fov/2) if extent[2]>extent[3] else ax.set_ylim(pc-fov/2, pc+fov/2)
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    sgnx=1 if x1>x0 else -1; sgny=1 if y1>y0 else -1
    for xi in range(Nx):
        for yi in range(Ny):
            L,P=rs.voxel_to_lps(xi,yi,N,ps,pc,vox)
            a=adj[xi,yi,z,lo:hi]/g; b=spc[xi,yi,z,lo:hi]/g
            xr=L + sgnx*(-vox/2 + (np.arange(len(xw)))/span*vox*0.92 + vox*0.04)
            yb=P - sgny*vox/2                                  # display-bottom of cell
            for s,col,lw in ((a,"#ff3b3b",0.5),(b,"#2ea1ff",0.6)):
                ax.plot(xr, yb + sgny*s*0.9*vox, color=col, lw=lw)
            ax.add_patch(plt.Rectangle((L-vox/2,P-vox/2),vox,vox,fill=False,ec='cyan',lw=0.25,alpha=0.35))
    ax.set_title(f"{os.path.basename(OUT)} | adj(red) vs SPICE(blue) on localizer (roi_selector geom) | z={z+1}/{Nz}",fontsize=9,color='w')
    ax.set_facecolor('k'); ax.tick_params(colors='w')
    fig.savefig(f"{outdir}/rv2_z{z:02d}.png",dpi=150,facecolor='k'); plt.close(fig); print("z",z)
print("OUTDIR",outdir)

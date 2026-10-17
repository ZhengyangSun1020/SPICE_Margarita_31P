#!/usr/bin/env python3
"""Associate each 80/100/120-sample launch-track ADC with its ring (via the
following 7560-sample phasecor block) and characterize the ramp vs ring radius."""
import os, struct, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"
OUT = os.path.dirname(os.path.abspath(__file__))
MASK_ACQEND, MASK_SYNC = 1 << 0, 1 << 5
def sgn(v): return v-65536 if v>32768 else v

f = open(dat, "rb")
_, numMeas = struct.unpack("<II", f.read(8))
ent=[]
for i in range(numMeas):
    f.seek(8+i*152); b=f.read(152)
    off,length = struct.unpack("<QQ", b[8:24]); ent.append((off,length))
moff,mlen = ent[-1]
f.seek(moff); hdrLen = struct.unpack("<I", f.read(4))[0]
pos, end = moff+hdrLen, moff+mlen

rec=[]
while pos < end:
    f.seek(pos); h=f.read(192)
    if len(h)<192: break
    dma = struct.unpack("<I",h[0:4])[0] & 0x01FFFFFF
    e0,e1 = struct.unpack("<II",h[40:48])
    samp,cha = struct.unpack("<HH",h[48:52])
    sLC = struct.unpack("<14H",h[52:80]); Line=sLC[0]; Seg=sLC[8]; Ida=sLC[9]; Idb=sLC[10]
    ice = struct.unpack("<24H",h[128:176])
    if e0 & MASK_ACQEND: break
    if dma<=0: break
    rec.append((pos,dma,samp,cha,Line,Seg,Ida,Idb,bool(e0&MASK_SYNC),ice))
    pos += dma
print(f"{len(rec)} blocks")

def read_adc(i):
    pos,dma,samp,cha = rec[i][0],rec[i][1],rec[i][2],rec[i][3]
    f.seek(pos+192)
    out=np.zeros((samp,cha),np.complex64)
    for c in range(cha):
        f.read(32); raw=np.frombuffer(f.read(samp*8),np.float32)
        out[:,c]=raw[0::2]+1j*raw[1::2]
    return out

launch=[i for i,r in enumerate(rec) if (not r[8]) and r[2] in (80,100,120) and r[3]>=8]
# associate each launch with the NEXT 7560 phasecor block -> ring radius from ITS ice
assoc=[]
for i in launch:
    j=i+1
    while j<len(rec) and not (rec[j][2]==7560 and not rec[j][8]):
        j+=1
    if j<len(rec):
        ic=rec[j][9]
        r=np.hypot(sgn(ic[0])+sgn(ic[1])/1e4, sgn(ic[2])+sgn(ic[3])/1e4)
        assoc.append((i, rec[i][2], rec[i][4], rec[j][4], r))
A=np.array([(a[1],a[2],a[3],a[4]) for a in assoc], float)  # samp, launchLin, pcLin, radius
samp_len=A[:,0]; radius=A[:,3]; sameLin=np.mean(A[:,1]==A[:,2])
print(f"associated {len(assoc)} launches; launchLin==phasecorLin fraction={sameLin:.2f}")
print(f"radius range {radius.min():.1f}..{radius.max():.1f}")
for L in (80,100,120):
    m=samp_len==L
    if m.sum(): print(f"  {int(L)}-sample launches: n={m.sum():3d}  radius {radius[m].min():.0f}..{radius[m].max():.0f}  mean={radius[m].mean():.0f}")

# ---- Fig 1: launch length vs radius, and active-length vs radius ----
def active_len(i, thr_frac=0.25):
    m=np.sqrt(np.mean(np.abs(read_adc(i))**2,axis=1))
    base=m[:min(10,len(m))].mean()
    thr=thr_frac*base
    # last sample above threshold from the start run
    above=np.where(m>thr)[0]
    return (above.max()+1) if len(above) else 0
act=np.array([active_len(a[0]) for a in assoc])

fig,ax=plt.subplots(1,2,figsize=(14,5))
sc=ax[0].scatter(radius, samp_len, c=radius, cmap='viridis', s=18)
ax[0].set_xlabel("ring radius (grad-moment, from phasecor ice)"); ax[0].set_ylabel("launch ADC allocated samples")
ax[0].set_title("Launch ADC length (80/100/120) vs ring radius")
ax[1].scatter(radius, act, c=radius, cmap='viridis', s=18)
ax[1].set_xlabel("ring radius"); ax[1].set_ylabel("active samples (|S|>0.25*start)")
ax[1].set_title("Active launch-ramp length vs ring radius")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"d_len_vs_radius.png"),dpi=150); print("saved d_len_vs_radius.png")

# ---- Fig 2: |S(t)| for launches spanning radius, colored by radius ----
order=np.argsort(radius)
picks=order[np.linspace(0,len(order)-1,9).astype(int)]
fig,ax=plt.subplots(1,2,figsize=(14,5))
norm=plt.Normalize(radius.min(),radius.max())
for k in picks:
    i=assoc[k][0]; m=np.sqrt(np.mean(np.abs(read_adc(i))**2,axis=1))
    col=cm.viridis(norm(radius[k]))
    ax[0].plot(m,color=col,lw=1.0,marker='.',ms=2,label=f"r={radius[k]:.0f} ({int(assoc[k][1])}pt)")
ax[0].set_title("Launch-track |S(t)| (color = ring radius)"); ax[0].set_xlabel("sample"); ax[0].set_ylabel("|S| RMS"); ax[0].legend(fontsize=6)
# peak (k-centre) sample vs radius
peak=np.array([np.argmax(np.sqrt(np.mean(np.abs(read_adc(a[0]))**2,axis=1))) for a in assoc])
ax[1].scatter(radius,peak,c=radius,cmap='viridis',s=16)
ax[1].set_xlabel("ring radius"); ax[1].set_ylabel("sample index of |S| max")
ax[1].set_title("Where launch |S| peaks vs radius")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"d_profiles.png"),dpi=150); print("saved d_profiles.png")

# ---- save proper extract with radius ----
Lmax=int(samp_len.max())
LT=np.zeros((Lmax,16,len(assoc)),np.complex64)
for k,a in enumerate(assoc):
    adc=read_adc(a[0]); LT[:adc.shape[0],:adc.shape[1],k]=adc
np.savez_compressed(os.path.join(OUT,"d_launchtrack_with_radius.npz"),
                    lt=LT, samp_len=samp_len, launch_lin=A[:,1], pc_lin=A[:,2],
                    radius=radius, active_len=act, peak_sample=peak)
print("saved d_launchtrack_with_radius.npz", LT.shape)
print("\nsample-length vs radius correlation:", np.corrcoef(radius,samp_len)[0,1].round(3))
print("active-length vs radius correlation:", np.corrcoef(radius,act)[0,1].round(3))
f.close()

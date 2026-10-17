#!/usr/bin/env python3
"""Stage 1: estimate per-TR 1H k0 phase/frequency drift from the launch navigator,
key it to each ring (Lin,Seg,Idb), and quantify the 31P-equivalent drift in Hz."""
import os, struct, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"/home/zsun/SPICE_Margarita/scripts")

dat="/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"
OUT=os.path.dirname(os.path.abspath(__file__))
GAMMA_RATIO = 17.235/42.577   # gamma_31P/gamma_1H (MHz/T) = 0.4048 (== 7T Larmor ratio)
print(f"gamma_31P/gamma_1H = {GAMMA_RATIO:.4f}")

# --- header timing via mapvbvd (fast) ---
import mapvbvd
from twix_fast import accelerate_mdh_read; accelerate_mdh_read()
tw=mapvbvd.mapVBVD(dat, quiet=True)
if isinstance(tw,list): tw=tw[-1]
hdr=tw['hdr']
def hf(sec,*k,default=None):
    d=hdr.get(sec,{})
    for kk in k:
        v=d.get(kk if isinstance(kk,tuple) else (kk,),None)
        if v is not None: return float(v)
    return default
dwell=[hf('MeasYaps',('sRXSPEC','alDwellTime',str(i))) for i in range(4)]
f1=hf('MeasYaps',('sTXSPEC','asNucleusInfo','0','lFrequency'))
f2=hf('MeasYaps',('sTXSPEC','asNucleusInfo','1','lFrequency'))
print("alDwellTime[0..3] ns:", dwell)
print("nucleus freqs Hz:", f1, f2)
larmor_1h_mhz = max(x for x in [f1,f2] if x)/1e6
larmor_31p_mhz = min(x for x in [f1,f2] if x)/1e6
print(f"1H Larmor {larmor_1h_mhz:.3f} MHz, 31P {larmor_31p_mhz:.3f} MHz, ratio {larmor_31p_mhz/larmor_1h_mhz:.4f}")

# --- raw MDH scan ---
MASK_ACQEND, MASK_SYNC = 1<<0, 1<<5
def sgn(v): return v-65536 if v>32768 else v
f=open(dat,"rb"); _,nm=struct.unpack("<II",f.read(8))
ent=[]
for i in range(nm):
    f.seek(8+i*152); b=f.read(152); off,length=struct.unpack("<QQ",b[8:24]); ent.append((off,length))
moff,mlen=ent[-1]; f.seek(moff); hL=struct.unpack("<I",f.read(4))[0]
pos,e=moff+hL,moff+mlen
rec=[]
while pos<e:
    f.seek(pos); h=f.read(192)
    if len(h)<192: break
    dma=struct.unpack("<I",h[0:4])[0]&0x01FFFFFF
    ts=struct.unpack("<I",h[12:16])[0]
    tslr=struct.unpack("<I",h[92:96])[0]            # ulTimeSinceLastRF
    e0,_=struct.unpack("<II",h[40:48]); samp,cha=struct.unpack("<HH",h[48:52])
    sLC=struct.unpack("<14H",h[52:80]); Lin=sLC[0]; Seg=sLC[8]; Ida=sLC[9]; Idb=sLC[10]
    if e0&1: break
    if dma<=0: break
    rec.append((pos,dma,samp,cha,Lin,sgn(Seg),Ida,Idb,ts,tslr,bool(e0&MASK_SYNC)))
    pos+=dma
def read_adc(i):
    p,dma,samp,cha=rec[i][0],rec[i][1],rec[i][2],rec[i][3]
    f.seek(p+192); out=np.zeros((samp,cha),np.complex128)
    for c in range(cha):
        f.read(32); raw=np.frombuffer(f.read(samp*8),np.float32); out[:,c]=raw[0::2]+1j*raw[1::2]
    return out

launch=[i for i,r in enumerate(rec) if (not r[10]) and r[2] in (80,100,120) and r[3]==8]
print(f"\n{len(launch)} launch navigators (1H, 8ch)")
print("TimeSinceLastRF (raw) sample:", [rec[i][9] for i in launch[:6]])

# per-TR k0 (near-centre) and noise; phase-aligned noise-weighted coil combine
K=len(launch)
k0=np.zeros((K,8),np.complex128); noise=np.zeros((K,8))
lin=np.array([rec[i][4] for i in launch]); seg=np.array([rec[i][5] for i in launch])
idb=np.array([rec[i][7] for i in launch]); ts=np.array([rec[i][8] for i in launch],float)
for j,i in enumerate(launch):
    adc=read_adc(i); k0[j]=adc[0:3].mean(0)
    mag=np.sqrt(np.mean(np.abs(adc)**2,1)); base=mag[:5].mean(); ab=np.where(mag>0.25*base)[0]
    a=(ab.max()+1) if len(ab) else adc.shape[0]//2
    tail=adc[min(a+3,adc.shape[0]-8):]; noise[j]=tail.std(0) if tail.shape[0]>=4 else adc[-8:].std(0)
wph=np.exp(-1j*np.angle(k0.mean(0)))
k0c=np.sum(k0*wph/(noise+1e-30)**2,1)
t=(ts-ts.min())*2.5e-3                     # seconds
o=np.argsort(t); t=t[o]; k0c=k0c[o]; lin=lin[o]; seg=seg[o]; idb=idb[o]
phi_1h=np.unwrap(np.angle(k0c))            # 1H 0th-order phase drift [rad]
phi_1h-=np.median(phi_1h)

# smooth (drift) vs residual: moving-median then lowpass poly
from numpy.polynomial import polynomial as P
# robust smooth: convolution with ~15 s window
def smooth(x,tt,win=15.0):
    y=np.empty_like(x)
    for k in range(len(x)):
        m=np.abs(tt-tt[k])<=win/2; y[k]=np.median(x[m])
    return y
phi_drift=smooth(phi_1h,t,20.0)
resid=phi_1h-phi_drift

# frequency: local slope of the SMOOTH drift -> d(phase)/dt = 2*pi*df_1H
dphi_dt=np.gradient(phi_drift,t)           # rad/s
df_1h = dphi_dt/(2*np.pi)                  # Hz (1H off-res drift rate proxy)
df_31p = df_1h*GAMMA_RATIO
# 31P 0th-order phase drift = 1H phase * gamma ratio (same B0 pathway, assumes shared reference time)
phi_31p = phi_drift*GAMMA_RATIO

print(f"\n1H k0 phase drift (0th order): peak-to-peak {np.ptp(phi_drift):.1f} rad  "
      f"({np.ptp(phi_drift)/(2*np.pi):.2f} cycles)")
print(f"31P-equiv 0th-order phase drift: p2p {np.ptp(phi_31p):.2f} rad ({np.degrees(np.ptp(phi_31p)):.0f} deg)")
print(f"[phase-wander RATE, not off-resonance] d(phi)/dt 1H: [{df_1h.min():.2f},{df_1h.max():.2f}] Hz")
print(f"NOTE: TimeSinceLastRF=0 -> cannot convert k0 phase to absolute off-resonance;")
print(f"      the robust deliverable is the per-TR 0th-order phase; freq scale needs tau/self-cal.")
print(f"31P ref: PCr FWHM ~10-30 Hz at 7T; 31P scale {larmor_31p_mhz:.1f} Hz/ppm; 640 deg = big 0th-order spread")
# cumulative frequency excursion is not meaningful; report the 0th-order phase as the primary correction
print(f"residual (non-drift) 1H phase jitter: std {np.std(resid):.2f} rad")

# --- correction table keyed by (Lin,Seg,Idb) ---
corr={}
for k in range(K):
    corr[(int(lin[k]),int(seg[k]),int(idb[k]))]=dict(phi0_31p=float(phi_31p[k]),
                                                     df_31p=float(df_31p[k]),
                                                     phi_1h=float(phi_drift[k]))
np.savez_compressed(os.path.join(OUT,"f_drift_correction.npz"),
                    lin=lin,seg=seg,idb=idb,t=t,phi_1h=phi_1h,phi_drift=phi_drift,
                    phi_31p=phi_31p,df_1h=df_1h,df_31p=df_31p,gamma_ratio=GAMMA_RATIO)
print("saved f_drift_correction.npz  (correction table for", K, "TRs)")

fig,ax=plt.subplots(2,2,figsize=(14,9))
ax[0,0].plot(t,phi_1h,'.',ms=3,alpha=0.5,label='raw 1H k0 phase'); ax[0,0].plot(t,phi_drift,'-',lw=1.5,color='r',label='smoothed drift')
ax[0,0].set_title("1H k0 phase drift"); ax[0,0].set_xlabel("time [s]"); ax[0,0].set_ylabel("phase [rad]"); ax[0,0].legend(fontsize=8)
ax[0,1].plot(t,np.degrees(phi_31p),'-',lw=1.5,color='purple')
ax[0,1].set_title(f"31P-equivalent 0th-order phase drift (x{GAMMA_RATIO:.3f})"); ax[0,1].set_xlabel("time [s]"); ax[0,1].set_ylabel("31P phase [deg]")
ax[1,0].plot(t,df_1h,label='df 1H'); ax[1,0].plot(t,df_31p,label='df 31P')
ax[1,0].set_title("drift rate (d phase/dt)"); ax[1,0].set_xlabel("time [s]"); ax[1,0].set_ylabel("Hz"); ax[1,0].legend(fontsize=8)
ax[1,1].hist(resid,bins=40,color='gray'); ax[1,1].set_title(f"residual 1H phase jitter (std {np.std(resid):.2f} rad)")
ax[1,1].set_xlabel("rad")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"f_drift.png"),dpi=150); print("saved f_drift.png")
f.close()

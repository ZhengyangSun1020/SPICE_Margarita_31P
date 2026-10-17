#!/usr/bin/env python3
"""Feasibility of the launch-track k0 (k-space centre) navigator for 31P:
SNR per TR, drift-phase trajectory, and respiratory resolvability."""
import os, struct, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"
OUT = os.path.dirname(os.path.abspath(__file__))
MASK_ACQEND, MASK_SYNC = 1 << 0, 1 << 5
def sgn(v): return v-65536 if v>32768 else v

f = open(dat, "rb")
_, numMeas = struct.unpack("<II", f.read(8))
ent=[]
for i in range(numMeas):
    f.seek(8+i*152); b=f.read(152); off,length=struct.unpack("<QQ",b[8:24]); ent.append((off,length))
moff,mlen=ent[-1]
f.seek(moff); hdrLen=struct.unpack("<I",f.read(4))[0]
pos,end=moff+hdrLen, moff+mlen
rec=[]
while pos<end:
    f.seek(pos); h=f.read(192)
    if len(h)<192: break
    dma=struct.unpack("<I",h[0:4])[0] & 0x01FFFFFF
    ts=struct.unpack("<I",h[12:16])[0]           # TimeStamp, 2.5 ms units
    e0,_=struct.unpack("<II",h[40:48])
    samp,cha=struct.unpack("<HH",h[48:52])
    sLC=struct.unpack("<14H",h[52:80]); Line=sLC[0]
    if e0 & MASK_ACQEND: break
    if dma<=0: break
    rec.append((pos,dma,samp,cha,Line,ts,bool(e0&MASK_SYNC)))
    pos+=dma

def read_adc(i):
    pos,dma,samp,cha=rec[i][0],rec[i][1],rec[i][2],rec[i][3]
    f.seek(pos+192); out=np.zeros((samp,cha),np.complex64)
    for c in range(cha):
        f.read(32); raw=np.frombuffer(f.read(samp*8),np.float32); out[:,c]=raw[0::2]+1j*raw[1::2]
    return out

launch=[i for i,r in enumerate(rec) if (not r[6]) and r[2] in (80,100,120) and r[3]>=8]
ts0=rec[launch[0]][5]
t_s=np.array([(rec[i][5]-ts0)*2.5e-3 for i in launch])   # seconds since first launch
print(f"{len(launch)} launches; scan span {t_s.max():.1f}s; median dt={np.median(np.diff(np.sort(t_s)))*1e3:.1f} ms")

ncha_all=np.array([rec[i][3] for i in launch])
print("launch channel counts:", dict(zip(*[x.tolist() for x in np.unique(ncha_all,return_counts=True)])))
NC=int(ncha_all.max())
nL=len(launch)
k0=np.zeros((nL,NC),np.complex128)          # near-centre (mean samp 0:3) per coil
noise=np.full((nL,NC), np.inf)               # per-coil noise from parked tail (inf => weight 0)
for j,i in enumerate(launch):
    adc=read_adc(i).astype(np.complex128)     # (samp,cha)
    cha=adc.shape[1]
    k0[j,:cha]=adc[0:3].mean(axis=0)
    mag=np.sqrt(np.mean(np.abs(adc)**2,axis=1))
    base=mag[:5].mean(); above=np.where(mag>0.25*base)[0]
    a=(above.max()+1) if len(above) else adc.shape[0]//2
    tail=adc[min(a+3,adc.shape[0]-8):]        # parked region ~ noise
    noise[j,:cha]=tail.std(axis=0) if tail.shape[0]>=4 else adc[-8:].std(axis=0)

# matched-filter (coil) SNR of the k0 sample per TR
snr_tr=np.sqrt(np.sum(np.abs(k0)**2,axis=1))/ (np.sqrt(np.sum(noise**2,axis=1))+1e-30)
# coherent coil combine for phase: align each coil by its own mean phase across TRs
wphase=np.exp(-1j*np.angle(k0.mean(axis=0)))
k0c=np.sum(k0*wphase*(1.0/(noise+1e-30)**2),axis=1)   # noise-weighted, phase-aligned
best=np.argmax(np.abs(k0).mean(axis=0))

print(f"\nper-TR k0 matched-filter SNR: median={np.median(snr_tr):.1f}  "
      f"p10={np.percentile(snr_tr,10):.1f}  p90={np.percentile(snr_tr,90):.1f}  min={snr_tr.min():.1f}")
print(f"best coil (mean |k0|)= {best};  |k0| best-coil median SNR="
      f"{np.median(np.abs(k0[:,best])/(noise[:,best]+1e-30)):.1f}")
print(f"coil-combined |k0| dynamic range across TRs: "
      f"{np.abs(k0c).min():.2e}..{np.abs(k0c).max():.2e} (cv={np.std(np.abs(k0c))/np.mean(np.abs(k0c)):.3f})")

# order by time
o=np.argsort(t_s); t=t_s[o]; ph=np.unwrap(np.angle(k0c[o])); amp=np.abs(k0c[o]); snro=snr_tr[o]

fig,ax=plt.subplots(2,2,figsize=(14,9))
ax[0,0].hist(snr_tr,bins=40,color='steelblue'); ax[0,0].axvline(np.median(snr_tr),color='r',ls='--')
ax[0,0].set_title(f"per-TR k0 SNR (median {np.median(snr_tr):.0f})"); ax[0,0].set_xlabel("matched-filter SNR"); ax[0,0].set_ylabel("#TR")

ax[0,1].plot(t, ph, '.-', ms=3, lw=0.5)
ax[0,1].set_title("k0 phase vs time (drift surrogate)"); ax[0,1].set_xlabel("time [s]"); ax[0,1].set_ylabel("unwrapped phase [rad]")

ax[1,0].plot(t, amp/np.median(amp), '.-', ms=3, lw=0.5)
ax[1,0].set_title("k0 magnitude vs time (motion surrogate)"); ax[1,0].set_xlabel("time [s]"); ax[1,0].set_ylabel("|k0| / median")

# resample magnitude to uniform grid -> spectrum (respiratory/cardiac?)
tu=np.linspace(t.min(),t.max(),len(t))
au=np.interp(tu, t, amp-amp.mean()); pu=np.interp(tu, t, np.unwrap(ph)); pu=pu-np.polyval(np.polyfit(tu,pu,2),tu)
fs=1.0/np.median(np.diff(tu))
FA=np.abs(np.fft.rfft(au*np.hanning(len(au)))); FP=np.abs(np.fft.rfft(pu*np.hanning(len(pu))))
fr=np.fft.rfftfreq(len(au), d=1.0/fs)
ax[1,1].plot(fr, FA/ (FA.max()+1e-30), label="|k0| mag"); ax[1,1].plot(fr, FP/(FP.max()+1e-30), label="k0 phase", alpha=0.7)
ax[1,1].axvspan(0.1,0.5,color='g',alpha=0.1,label="resp 0.1-0.5Hz"); ax[1,1].axvspan(0.7,1.6,color='m',alpha=0.1,label="cardiac 0.7-1.6Hz")
ax[1,1].set_xlim(0,min(2.5,fs/2)); ax[1,1].set_xlabel("Hz"); ax[1,1].set_ylabel("norm |FFT|")
ax[1,1].set_title(f"navigator temporal spectrum (fs~{fs:.2f} Hz)"); ax[1,1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"e_k0_feasibility.png"),dpi=150); print("saved e_k0_feasibility.png")

print(f"\neffective navigator sampling rate fs = {fs:.2f} Hz (Nyquist {fs/2:.2f} Hz)")
print(f"  -> respiratory (~0.2-0.35 Hz): {'RESOLVABLE' if fs/2>0.5 else 'NOT resolvable'}")
print(f"  -> cardiac (~1 Hz): {'resolvable' if fs/2>1.2 else 'NOT resolvable (aliased)'}")
f.close()

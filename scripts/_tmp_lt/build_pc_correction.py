#!/usr/bin/env python3
"""Build a per-phasecor-ADC 0th-order phase correction (31P), assigning each 31P
readout ADC the most-recent launch navigator's smoothed drift phase. Verify it
aligns 1:1 with mapvbvd's phasecor ordering."""
import os, struct, sys, numpy as np
sys.path.insert(0,"/home/zsun/SPICE_Margarita/scripts")
dat="/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"
OUT=os.path.dirname(os.path.abspath(__file__))
GAMMA=17.235/42.577    # 0.4048
MASK_SYNC=1<<5

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
    e0,_=struct.unpack("<II",h[40:48]); samp,cha=struct.unpack("<HH",h[48:52])
    sLC=struct.unpack("<14H",h[52:80]); Lin=sLC[0]
    if e0&1: break
    if dma<=0: break
    rec.append((pos,dma,samp,cha,Lin,ts,bool(e0&MASK_SYNC)))
    pos+=dma
def read_adc(i):
    p,dma,samp,cha=rec[i][0],rec[i][1],rec[i][2],rec[i][3]
    f.seek(p+192); out=np.zeros((samp,cha),np.complex128)
    for c in range(cha):
        f.read(32); raw=np.frombuffer(f.read(samp*8),np.float32); out[:,c]=raw[0::2]+1j*raw[1::2]
    return out

# launches in acquisition order
launch=[i for i,r in enumerate(rec) if (not r[6]) and r[2] in (80,100,120) and r[3]==8]
k0=np.zeros((len(launch),8),np.complex128); noise=np.zeros((len(launch),8)); ts=np.array([rec[i][5] for i in launch],float)
for j,i in enumerate(launch):
    adc=read_adc(i); k0[j]=adc[0:3].mean(0)
    mag=np.sqrt(np.mean(np.abs(adc)**2,1)); base=mag[:5].mean(); ab=np.where(mag>0.25*base)[0]
    a=(ab.max()+1) if len(ab) else adc.shape[0]//2; tail=adc[min(a+3,adc.shape[0]-8):]
    noise[j]=tail.std(0) if tail.shape[0]>=4 else adc[-8:].std(0)
wph=np.exp(-1j*np.angle(k0.mean(0))); k0c=np.sum(k0*wph/(noise+1e-30)**2,1)
t=(ts-ts.min())*2.5e-3
phi=np.unwrap(np.angle(k0c)); phi-=np.median(phi)
def smooth(x,tt,win=20.0):
    y=np.empty_like(x)
    for k in range(len(x)): y[k]=np.median(x[np.abs(tt-tt[k])<=win/2])
    return y
phi_d=smooth(phi,t,20.0)
phi_31p=GAMMA*phi_d                       # per-launch 31P 0th-order phase (acq order)
print(f"{len(launch)} launches; 31P 0th-order phase p2p={np.ptp(phi_31p):.2f} rad")

# walk records; assign most-recent launch phase to each 7560 phasecor ADC
launch_set={i:phi_31p[j] for j,i in enumerate(launch)}
cur=0.0; phi_pc=[]; lin_pc=[]
for i,r in enumerate(rec):
    if i in launch_set: cur=launch_set[i]
    if (not r[6]) and r[2]==7560 and r[3]==16:
        phi_pc.append(cur); lin_pc.append(r[4])
phi_pc=np.array(phi_pc); lin_pc=np.array(lin_pc)
print(f"phasecor ADCs: {len(phi_pc)}  (expect 3114)")

# verify alignment with mapvbvd phasecor order
from twix_fast import accelerate_mdh_read; accelerate_mdh_read()
import mapvbvd
tw=mapvbvd.mapVBVD(dat,quiet=True); tw=tw[-1] if isinstance(tw,list) else tw
pc=tw['phasecor']; lin_mv=pc.Lin.astype(int)
print(f"mapvbvd phasecor NAcq={len(lin_mv)}")
if len(lin_mv)==len(lin_pc):
    match=np.mean(lin_mv==lin_pc)
    print(f"Lin-order match raw-vs-mapvbvd: {match*100:.1f}%")
else:
    print("LENGTH MISMATCH -> need key-based alignment")

np.save(os.path.join(OUT,"phi_pc_31p.npy"), phi_pc.astype(np.float64))
print("saved phi_pc_31p.npy", phi_pc.shape)
f.close()

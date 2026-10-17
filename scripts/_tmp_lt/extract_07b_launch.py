#!/usr/bin/env python3
"""Extract & analyze the pre-launch-track ADC (first ADC of each TR) from phasecor (31P CRT)."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/zsun/SPICE_Margarita/scripts")
from twix_fast import accelerate_mdh_read
accelerate_mdh_read()
import mapvbvd

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"
OUT = os.path.dirname(os.path.abspath(__file__))

tw = mapvbvd.mapVBVD(dat, quiet=True)
if isinstance(tw, list): tw = tw[-1]
pc = tw['phasecor']
pc.squeeze = True; pc.flagRemoveOS = False

Lin = pc.Lin.astype(int); Ida = pc.Ida.astype(int); Idb = pc.Idb.astype(int)
Seg = pc.Seg.astype(int); Idc = pc.Idc.astype(int)
ice = pc.iceParam
def sgn(v):
    v=int(v); return v-65536 if v>32768 else v
kx_gm = np.array([sgn(ice[s,0])+sgn(ice[s,1])/1e4 for s in range(len(Lin))])
ky_gm = np.array([sgn(ice[s,2])+sgn(ice[s,3])/1e4 for s in range(len(Lin))])
r_gm  = np.sqrt(kx_gm**2+ky_gm**2)
ppc = 2*int(Idc[0])   # pts per circle = 180

print("Reading phasecor (unsorted) ...")
u = pc.unsorted()                     # (7560, 16, 3114)
ncol, ncha, nacq = u.shape
print("  shape", u.shape, u.dtype)
magrms = np.sqrt(np.mean(np.abs(u)**2, axis=1)).astype(np.float32)  # (7560, 3114)

# select representative rings by radius (use Ida==0 events)
first = np.where(Ida==0)[0]
rr = r_gm[first]; order = np.argsort(rr)
pick = [order[0], order[len(order)//4], order[len(order)//2], order[3*len(order)//4], order[-1]]
sel_first = [first[i] for i in pick]
# keep complex for these few (for phase)
sel_cplx = {int(s): u[:, :, s].copy() for s in sel_first}
del u   # free 3GB

# ---------- Plot 1: ensemble mean |S(t)| per Ida ----------
fig, ax = plt.subplots(1,2, figsize=(14,5))
for ida in range(18):
    idx = np.where(Ida==ida)[0]
    if len(idx)==0: continue
    m = magrms[:, idx].mean(axis=1)
    ax[0].plot(m, lw=0.6, label=f"{ida}" if ida in (0,1,2,9,17) else None,
               color='crimson' if ida==0 else None, zorder=5 if ida==0 else 1)
ax[0].set_title("Ensemble mean |S(t)| per Ida (red=Ida0)"); ax[0].set_xlabel("sample"); ax[0].set_ylabel("|S| RMS")
ax[0].legend(fontsize=7, title="Ida")
for ida in [0,1,2,9,17]:
    idx=np.where(Ida==ida)[0]; m=magrms[:,idx].mean(axis=1)
    ax[1].plot(m[:400], lw=1.0, label=f"Ida={ida}", color='crimson' if ida==0 else None)
ax[1].set_title("first 400 samples (launch ramp / k-centre?)"); ax[1].set_xlabel("sample"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"b_ensemble_per_ida.png"), dpi=150); print("saved b_ensemble_per_ida.png")

# ---------- Plot 2: Ida==0 |S(t)| per ring radius ----------
fig, ax = plt.subplots(1,2, figsize=(14,5))
for k,s in enumerate(sel_first):
    m = magrms[:, s]
    ax[0].plot(m, lw=0.6, label=f"Lin={Lin[s]} r={r_gm[s]:.0f}")
    ax[1].plot(m[:600], lw=0.9, label=f"Lin={Lin[s]} r={r_gm[s]:.0f}")
ax[0].set_title("Ida=0 |S(t)| full, per ring"); ax[0].set_xlabel("sample"); ax[0].set_ylabel("|S| RMS"); ax[0].legend(fontsize=7)
ax[1].set_title("Ida=0 first 600 samples"); ax[1].set_xlabel("sample"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"b_ida0_perring.png"), dpi=150); print("saved b_ida0_perring.png")

# ---------- Plot 3: circling periodicity (FFT of |S(t)|) Ida0 vs Ida9 ----------
def fftmag(ida):
    idx=np.where(Ida==ida)[0]; m=magrms[:,idx].mean(axis=1); m=m-m.mean()
    return np.abs(np.fft.rfft(m*np.hanning(len(m))))
f0=fftmag(0); f9=fftmag(9); fr=np.fft.rfftfreq(ncol)
plt.figure(figsize=(12,4.5))
plt.plot(fr,f0,label="Ida=0"); plt.plot(fr,f9,label="Ida=9",alpha=0.7)
plt.axvline(1/ppc,color='r',ls='--',lw=0.8,label=f"1/{ppc} circle rate")
plt.xlim(0,0.05); plt.legend(); plt.xlabel("cycles/sample"); plt.ylabel("|FFT|")
plt.title("Circling-rate content of |S(t)|")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"b_circling_fft.png"), dpi=150); print("saved b_circling_fft.png")

# ---------- Plot 4: phase of coil0 first samples, Ida0 ----------
plt.figure(figsize=(12,4.5))
for s in sel_first[:3]:
    ph=np.unwrap(np.angle(sel_cplx[int(s)][:,0]))
    plt.plot(ph[:600], lw=0.7, label=f"Lin={Lin[s]} r={r_gm[s]:.0f}")
plt.title("Ida=0 phase (coil0), first 600 samples"); plt.xlabel("sample"); plt.ylabel("unwrapped phase")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"b_ida0_phase.png"), dpi=150); print("saved b_ida0_phase.png")

# ---------- numeric summary ----------
print("\n=== ensemble mean |S| at first samples, per Ida ===")
print(" Ida    s0       s1       s2       s5      s90     s400    overall   n")
for ida in range(18):
    idx=np.where(Ida==ida)[0]
    if len(idx)==0: continue
    m=magrms[:,idx].mean(axis=1)
    print(f"  {ida:2d}  {m[0]:.3e} {m[1]:.3e} {m[2]:.3e} {m[5]:.3e} {m[90]:.3e} {m[400]:.3e} {m.mean():.3e} {len(idx)}")

# save Ida0 block (coil-RMS) + counters for later
np.savez_compressed(os.path.join(OUT,"b_launch_ida0.npz"),
                    mag_ida0=magrms[:, first].astype(np.float32),
                    ring=Lin[first], radius=r_gm[first], seg=Seg[first], idb=Idb[first],
                    ppc=ppc)
print("\nsaved b_launch_ida0.npz  (mag for", len(first), "Ida0 events)")
print("pts_per_circle =", ppc, " NCol/ppc =", ncol/ppc)

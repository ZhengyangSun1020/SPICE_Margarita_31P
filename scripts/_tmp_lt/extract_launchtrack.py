#!/usr/bin/env python3
"""Extract & analyze the pre-launch-track ADC (first ADC / Ida==0 of each TR)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mapvbvd

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/260911_yife_liver_launchtrack_BS/meas_MID00057_FID98135_ZS_crt_dualband_sat_BS_Launchtrack_1000.dat"
OUT = os.path.dirname(os.path.abspath(__file__))

tw = mapvbvd.mapVBVD(dat, quiet=True)
if isinstance(tw, list): tw = tw[-1]
img = tw['image']
img.squeeze = True
img.flagRemoveOS = False

Lin = img.Lin.astype(int); Ida = img.Ida.astype(int); Idb = img.Idb.astype(int)
Seg = img.Seg.astype(int); Idd = img.Idd.astype(int)
ice = img.iceParam
seg_signed = Seg.copy(); seg_signed[seg_signed > 1000] -= 65536

def sgn(v):
    v = int(v); return v-65536 if v > 32768 else v
kx_gm = np.array([sgn(ice[s,0]) + sgn(ice[s,1])/1e4 for s in range(len(Lin))])
ky_gm = np.array([sgn(ice[s,2]) + sgn(ice[s,3])/1e4 for s in range(len(Lin))])
r_gm  = np.sqrt(kx_gm**2 + ky_gm**2)

print("Reading raw data (unsorted)...")
u = img.unsorted()            # (NCol, NCha, NAcq)
print("  unsorted shape:", u.shape, u.dtype)
ncol, ncha, nacq = u.shape

first = np.where(Ida == 0)[0]      # first ADC of each TR
print(f"  #first-ADC (Ida==0) events = {len(first)}")
# RMS combine over coils
def rms(a):  # a: (ncol, ncha)
    return np.sqrt(np.mean(np.abs(a)**2, axis=1))

# Build per-ring info for the first-ADC events
rings = Lin[first]; radii = r_gm[first]
order = np.argsort(radii)
# choose representative rings across radius range
sel_idx = [order[0], order[len(order)//4], order[len(order)//2],
           order[3*len(order)//4], order[-1]]
sel = [first[i] for i in sel_idx]
sel_r = [radii[i] for i in sel_idx]
sel_ring = [rings[i] for i in sel_idx]

# also a steady circling ADC (Ida==6) from the same TRs for contrast
def same_tr_ida(base_idx, ida_target):
    # base_idx is an Ida==0 event; find Ida==ida_target in same TR block (next few)
    for j in range(base_idx, min(base_idx+12, nacq)):
        if Ida[j] == ida_target and Lin[j] == Lin[base_idx] and Idb[j]==Idb[base_idx]:
            return j
    return None

# ---- Plot 1: launch-track |S(t)| full, several rings ----
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
ax = axs[0,0]
for k, s in enumerate(sel):
    prof = rms(u[:, :, s])
    ax.plot(prof, lw=0.6, label=f"ring Lin={sel_ring[k]}  r_gm={sel_r[k]:.0f}")
ax.set_title("Launch-track ADC (Ida=0): coil-RMS |S| over 4540 samples")
ax.set_xlabel("sample"); ax.set_ylabel("|S| (RMS coils)"); ax.legend(fontsize=7)

ax = axs[0,1]
for k, s in enumerate(sel):
    prof = rms(u[:, :, s])
    ax.plot(prof[:400], lw=0.8, label=f"Lin={sel_ring[k]} r={sel_r[k]:.0f}")
ax.set_title("Launch-track: first 400 samples (k-centre crossing)")
ax.set_xlabel("sample"); ax.set_ylabel("|S| RMS"); ax.legend(fontsize=7)

# ---- Plot 2: contrast first ADC vs steady circling ADC (Ida=6) for mid ring ----
ax = axs[1,0]
smid = sel[len(sel)//2]
ax.plot(rms(u[:, :, smid]), lw=0.6, label=f"Ida=0 (launch)  Lin={Lin[smid]}")
j6 = same_tr_ida(smid, 6)
if j6 is not None:
    ax.plot(rms(u[:, :, j6]), lw=0.6, label=f"Ida=6 (circling) Lin={Lin[j6]}")
ax.set_title("Launch ADC vs steady circling ADC (same ring)")
ax.set_xlabel("sample"); ax.set_ylabel("|S| RMS"); ax.legend(fontsize=7)

# ---- Plot 3: phase of one coil, first samples of launch ADC ----
ax = axs[1,1]
c = 0
for k, s in enumerate(sel[:3]):
    ph = np.unwrap(np.angle(u[:, c, s]))
    ax.plot(ph[:400], lw=0.7, label=f"Lin={sel_ring[k]} r={sel_r[k]:.0f}")
ax.set_title(f"Launch-track phase (coil {c}), first 400 samples")
ax.set_xlabel("sample"); ax.set_ylabel("unwrapped phase [rad]"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "launchtrack_overview.png"), dpi=150)
print("saved launchtrack_overview.png")

# ---- Peak (k-centre) location per ring ----
peak_pos = np.array([np.argmax(rms(u[:, :, s])) for s in first])
peak_val = np.array([rms(u[:, :, s]).max() for s in first])
# also value at very first sample
first_val = np.array([rms(u[:, :, s])[0] for s in first])

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.2))
ax2[0].scatter(radii, peak_pos, s=8)
ax2[0].set_xlabel("ring radius r_gm (ice gradient-moment)")
ax2[0].set_ylabel("sample index of |S| peak (k-centre)")
ax2[0].set_title("Where the launch track crosses k=0 vs ring radius")
ax2[1].scatter(radii, peak_val, s=8, label="peak |S|")
ax2[1].scatter(radii, first_val, s=8, label="|S| at sample 0")
ax2[1].set_xlabel("ring radius r_gm"); ax2[1].set_ylabel("|S| RMS")
ax2[1].set_title("Launch-track central signal vs ring radius"); ax2[1].legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "launchtrack_kcentre.png"), dpi=150)
print("saved launchtrack_kcentre.png")

# ---- Save the extracted launch-track block for later use ----
lt = u[:, :, first].astype(np.complex64)     # (4540,16,nTR)
np.savez_compressed(os.path.join(OUT, "launchtrack_adc.npz"),
                    lt=lt, ring=rings, radius=radii, idb=Idb[first],
                    seg=seg_signed[first], peak_pos=peak_pos)
print("saved launchtrack_adc.npz", lt.shape)

# ---- Numeric summary ----
print("\n=== SUMMARY ===")
print(f"TRs (first-ADC events): {len(first)}")
print(f"ring radius r_gm range: {radii.min():.1f} .. {radii.max():.1f}")
print(f"peak(|S|) sample index: min={peak_pos.min()} med={int(np.median(peak_pos))} max={peak_pos.max()}")
print(f"mean peak|S| / mean |S| at sample0 = {peak_val.mean():.1f} / {first_val.mean():.1f}")
print("per representative ring:")
for k,s in enumerate(sel):
    pr = rms(u[:, :, s])
    print(f"  Lin={Lin[s]:3d} r_gm={r_gm[s]:6.1f}  peak@{np.argmax(pr):4d}  |S0|={pr[0]:.0f}  peak={pr.max():.0f}  tail={pr[-500:].mean():.0f}")

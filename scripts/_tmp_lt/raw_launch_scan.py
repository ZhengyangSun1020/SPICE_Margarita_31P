#!/usr/bin/env python3
"""Raw VD/VE MDH scanner: find & extract the 80/100-sample launch-track ADCs
that precede each 31P PHASCOR readout block (filtered out by mapvbvd)."""
import os, struct, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"
OUT = os.path.dirname(os.path.abspath(__file__))

MASK_ACQEND   = 1 << 0
MASK_SYNCDATA = 1 << 5

f = open(dat, "rb")
first, numMeas = struct.unpack("<II", f.read(8))
# RAID entries (152 bytes): measId(u4) fileId(u4) off(u8) len(u8) pat[64] prot[64]
entries = []
for i in range(numMeas):
    f.seek(8 + i*152)
    b = f.read(152)
    mid, fid = struct.unpack("<II", b[:8])
    off, length = struct.unpack("<QQ", b[8:24])
    entries.append((off, length))
meas_off, meas_len = entries[-1]
print(f"numMeas={numMeas}  last meas off={meas_off} len={meas_len}")

f.seek(meas_off)
hdrLen = struct.unpack("<I", f.read(4))[0]
data_start = meas_off + hdrLen
meas_end   = meas_off + meas_len
print(f"protocol hdrLen={hdrLen}  data_start={data_start}")

pos = data_start
records = []      # (pos, dmaLen, samples, nCha, Lin, Seg_signed, Ida, Idb, evalmask, ice0_3)
n = 0
while pos < meas_end:
    f.seek(pos)
    hdr = f.read(192)
    if len(hdr) < 192:
        break
    dma_and_flags = struct.unpack("<I", hdr[0:4])[0]
    dmaLen = dma_and_flags & 0x01FFFFFF
    eval0, eval1 = struct.unpack("<II", hdr[40:48])
    samples, usedCha = struct.unpack("<HH", hdr[48:52])
    Line, Acq, Sli, Par, Echo, Pha, Rep, Set_, Seg, Ida, Idb, Idc, Idd, Ide = struct.unpack("<14H", hdr[52:80])
    ice = struct.unpack("<24H", hdr[128:176])
    if eval0 & MASK_ACQEND:
        print(f"ACQEND at scan {n}, pos {pos}")
        break
    if dmaLen <= 0:
        break
    seg_signed = Seg - 65536 if Seg > 32768 else Seg
    is_sync = bool(eval0 & MASK_SYNCDATA)
    records.append((pos, dmaLen, samples, usedCha, Line, seg_signed, Ida, Idb, eval0, is_sync, ice[0], ice[1], ice[2], ice[3]))
    pos += dmaLen
    n += 1

print(f"scanned {n} MDH blocks")
rec = records
samples_arr = np.array([r[2] for r in rec])
sync_arr    = np.array([r[9] for r in rec])
# histogram of sample counts (non-sync)
vals, cnts = np.unique(samples_arr[~sync_arr], return_counts=True)
print("\nSamplesInScan histogram (non-sync):")
for v, c in zip(vals, cnts):
    print(f"  samples={v:6d}  count={c}")

# Launch ADCs = non-sync, samples not in {192 (1H), 7560 (phasecor)}, usedCha 16
launch_idx = [i for i, r in enumerate(rec)
              if (not r[9]) and r[2] not in (192, 7560) and r[3] >= 8 and r[2] > 0]
print(f"\n#launch-candidate ADCs = {len(launch_idx)}")
if launch_idx:
    ls = np.array([rec[i][2] for i in launch_idx])
    lv, lc = np.unique(ls, return_counts=True)
    print("  launch sample-length histogram:", dict(zip(lv.tolist(), lc.tolist())))

# context: show 12 blocks around a phasecor transition
print("\n--- sequence context (samples per consecutive block, first 60) ---")
ctx = []
for i, r in enumerate(rec[:60]):
    tag = "SYNC" if r[9] else ("1H" if r[2]==192 else ("PHASCOR" if r[2]==7560 else "LAUNCH?"))
    ctx.append(f"{r[2]}:{tag}")
print("  " + " ".join(ctx))

def sgn(v): return v-65536 if v>32768 else v

# Extract launch ADC complex data (all channels)
def read_adc(i):
    pos, dmaLen, samples, usedCha = rec[i][0], rec[i][1], rec[i][2], rec[i][3]
    f.seek(pos + 192)
    out = np.zeros((samples, usedCha), dtype=np.complex64)
    for c in range(usedCha):
        ch = f.read(32)
        raw = np.frombuffer(f.read(samples*8), dtype=np.float32)
        out[:, c] = raw[0::2] + 1j*raw[1::2]
    return out

if launch_idx:
    # radius per launch from its own ice[0..3]
    lrad = np.array([np.hypot(sgn(rec[i][10])+sgn(rec[i][11])/1e4,
                              sgn(rec[i][12])+sgn(rec[i][13])/1e4) for i in launch_idx])
    lLin = np.array([rec[i][4] for i in launch_idx])
    lsamp= np.array([rec[i][2] for i in launch_idx])
    print(f"\nlaunch ring-radius (from launch ice): min={lrad.min():.1f} max={lrad.max():.1f} "
          f"nuniq={len(np.unique(np.round(lrad,1)))}")
    print(f"launch Lin range {lLin.min()}..{lLin.max()};  sample lengths {np.unique(lsamp).tolist()}")

    # pick representative launches across radius
    order = np.argsort(lrad)
    pick = [order[0], order[len(order)//4], order[len(order)//2], order[3*len(order)//4], order[-1]]
    sel = [launch_idx[i] for i in pick]

    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    for i in sel:
        adc = read_adc(i)
        mag = np.sqrt(np.mean(np.abs(adc)**2, axis=1))
        rr = np.hypot(sgn(rec[i][10])+sgn(rec[i][11])/1e4, sgn(rec[i][12])+sgn(rec[i][13])/1e4)
        ax[0].plot(mag, marker='.', ms=3, lw=0.8, label=f"Lin={rec[i][4]} r={rr:.0f} ({rec[i][2]}pt)")
    ax[0].set_title("Launch-track ADC: coil-RMS |S| across the 80/100 ramp samples")
    ax[0].set_xlabel("sample (gradient ramping k=0 -> ring radius)"); ax[0].set_ylabel("|S| RMS"); ax[0].legend(fontsize=7)
    ax[0].axvline(0, color='k', lw=0.5, ls=':')

    # phase (coil 0)
    for i in sel[:3]:
        adc = read_adc(i)
        ax[1].plot(np.unwrap(np.angle(adc[:,0])), marker='.', ms=3, lw=0.8, label=f"Lin={rec[i][4]}")
    ax[1].set_title("Launch-track phase (coil 0)"); ax[1].set_xlabel("sample"); ax[1].set_ylabel("unwrapped phase")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c_launch_track.png"), dpi=150)
    print("saved c_launch_track.png")

    # Save the full launch-track extract (complex, padded to max length)
    Lmax = int(lsamp.max())
    LT = np.zeros((Lmax, 16, len(launch_idx)), dtype=np.complex64)
    for j, i in enumerate(launch_idx):
        adc = read_adc(i)
        LT[:adc.shape[0], :adc.shape[1], j] = adc
    np.savez_compressed(os.path.join(OUT, "c_launchtrack_extract.npz"),
                        lt=LT, ring=lLin, radius=lrad, samples=lsamp,
                        seg=np.array([rec[i][5] for i in launch_idx]),
                        idb=np.array([rec[i][7] for i in launch_idx]))
    print("saved c_launchtrack_extract.npz", LT.shape)

    # numeric: k-centre (sample 0) magnitude vs steady tail
    print("\n=== launch-track |S| profile (ensemble over launches, per sample-length) ===")
    for L in np.unique(lsamp):
        sub = [i for i in launch_idx if rec[i][2]==L]
        prof = np.zeros(L)
        for i in sub:
            prof += np.sqrt(np.mean(np.abs(read_adc(i))**2, axis=1))
        prof /= len(sub)
        print(f"  {L}-sample launches (n={len(sub)}): |S|[0]={prof[0]:.3e}  [1]={prof[1]:.3e}  "
              f"[5]={prof[5]:.3e}  mid={prof[L//2]:.3e}  end={prof[-1]:.3e}")
f.close()

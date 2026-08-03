"""
Overlay 31P MRSI SNR/PCr maps onto TRA_localizer anatomical images.

Usage:
    python scripts/overlay_31p_localizer.py

Outputs:
    output_7T_iterative/figures/overlay_z*.png  (9 panels, one per 31P slice)
    output_7T_iterative/figures/overlay_summary.png  (3x3 grid of all slices)
"""
import os
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────────────────
LOCALIZER_DIR = "/mnt/f/AliceConceptB1/in_vivo_data_7T/Susuan_0624/0626_invivo/locolizer"
DATA_DIR      = "/home/zsun/SPICE_Margarita/output_7T_iterative/data"
OUT_DIR       = "/home/zsun/SPICE_Margarita/output_7T_iterative/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 31P MRSI geometry ─────────────────────────────────────────────────────────
# Volume center in LPS (mm)
POS_SAG  = -1.82    # L (positive = left)
POS_COR  = -37.54   # P (positive = posterior)
POS_TRA  = -10.0    # S (positive = superior)

FOV_XY   = 240.0    # mm
N_XY     = 10
VOX_XY   = FOV_XY / N_XY   # 24 mm per voxel

FOV_Z    = 240.0    # mm
N_Z      = 9
VOX_Z    = FOV_Z / N_Z   # 26.67 mm per voxel

# Axial orientation (no in-plane rotation):
#   read direction  dGr = [-1, 0, 0]  (-L = R direction)
#   phase direction dGp = [ 0, 1, 0]  (+P direction)
# Voxel (ix, iy) center in LPS:
#   L(ix) = POS_SAG - (ix - (N_XY-1)/2) * VOX_XY
#   P(iy) = POS_COR + (iy - (N_XY-1)/2) * VOX_XY
def voxel_lp(ix, iy):
    # Match the iter_spectra orientation:
    #   ix = x-axis in recon → subplot ROWS in spectra (ix=9 at top, ix=0 at bottom)
    #   iy = y-axis in recon → subplot COLS in spectra (iy=0 at left, iy=9 at right)
    # In the localizer (axial, anterior=top, patient-right=left):
    #   iy → L (col): iy=0 at left (patient's right, -L), iy=9 at right (patient's left, +L)
    #   ix → -P (row): ix=9 at top (anterior, -P), ix=0 at bottom (posterior, +P)
    # finufft DC is at index N//2 = 5 for N_XY=10 (not (N_XY-1)/2 = 4.5)
    ctr = N_XY // 2   # = 5
    L = POS_SAG + (iy - ctr) * VOX_XY
    P = POS_COR - (ix - ctr) * VOX_XY
    return L, P

# 31P slice S-positions (center of each slab)
sp31 = [POS_TRA + (iz - (N_Z-1)/2.0) * VOX_Z for iz in range(N_Z)]
print("31P MRSI S-positions (mm):", [f"{z:.1f}" for z in sp31])

# ── Load TRA_localizer slices (Series 0002: 10 axial slices at -90…+90 mm) ───
all_files = sorted(os.listdir(LOCALIZER_DIR))

# Group by DICOM series number (field[3] in the filename)
from collections import defaultdict
series_files = defaultdict(list)
for fname in all_files:
    series_files[fname.split(".")[3]].append(fname)

# Read all slices in series 0002 (TRA_localizer), keyed by S position
loc_slices = {}   # z_mm (float) → (pixel_array, ds)
for fname in sorted(series_files["0002"]):
    ds = pydicom.dcmread(os.path.join(LOCALIZER_DIR, fname))
    pos = [float(x) for x in ds.ImagePositionPatient]
    z_mm = round(pos[2], 1)
    arr  = ds.pixel_array.astype(np.float32)
    loc_slices[z_mm] = (arr, ds)
    print(f"TRA_localizer: S={z_mm:.1f}mm  {arr.shape}")

loc_z_sorted = sorted(loc_slices.keys())

def find_closest_loc(z31p):
    """Return (z_key, pixel_array, ds) for the closest TRA_localizer to a given S."""
    diffs = {z: abs(z - z31p) for z in loc_z_sorted}
    best_z = min(diffs, key=diffs.get)
    return best_z, loc_slices[best_z][0], loc_slices[best_z][1]

# ── Load 31P reconstruction data ──────────────────────────────────────────────
# recon_fid_fovfix.npy has the FOV-shift axis correction applied (d_ro↔d_pe swap)
_fovfix = os.path.join(DATA_DIR, "recon_fid_fovfix.npy")
_raw    = os.path.join(DATA_DIR, "recon_fid.npy")
recon_fid = np.load(_fovfix if os.path.exists(_fovfix) else _raw).astype(np.complex128)
# shape: (Nx, Ny, Nz, nspec)

dwell_s = 1.0 / 2778.0
larmor  = 120.316   # MHz
n_spec  = recon_fid.shape[-1]
t       = np.arange(n_spec) * dwell_s
apo     = np.exp(-20.0 * np.pi * t)

# Apply 20 Hz apodization and FFT
spec = np.fft.fftshift(np.fft.fft(recon_fid * apo, axis=-1), axes=-1)

FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
PPM  = -FREQ / larmor

# PCr peak region (±2 ppm around 0)
pcr_mask = (PPM >= -2.0) & (PPM <= 2.0)
noise_mask = np.abs(PPM) > 12.0

mag      = np.abs(spec)   # (Nx, Ny, Nz, nspec)
pcr_amp  = mag[:, :, :, pcr_mask].max(axis=-1)   # (Nx, Ny, Nz)

# SNR: peak / std of last 20% of fftshifted spectrum (same as 07b find_best_snr_3d)
noise_start = int(0.8 * n_spec)
noise_std   = mag[:, :, :, noise_start:].std(axis=-1)
snr_map     = np.where(noise_std > 0, pcr_amp / noise_std, 0.0)

print(f"PCr amp range: {pcr_amp.min():.3f} – {pcr_amp.max():.3f}")
print(f"Noise std range: {noise_std.min():.3f} – {noise_std.max():.3f}")
print(f"SNR range: {snr_map.min():.1f} – {snr_map.max():.1f}")

# Global colormap limits across all slices
snr_vmax = np.percentile(snr_map, 99)
pcr_vmax = np.percentile(pcr_amp, 99)

# ── Helper: pixel coords for voxel center on localizer ────────────────────────
def voxel_to_pixel(ix, iy, ds):
    """
    Returns (row, col) on the DICOM image for voxel (ix, iy).
    Localizer: IPP=[-200,-200,z], IOP=[1,0,0,0,1,0], PS=[1.5625,1.5625]
    L = IPP[0] + col * PS[1] * IOP[0]  → col = (L - IPP[0]) / PS[1]
    P = IPP[1] + row * PS[0] * IOP[4]  → row = (P - IPP[1]) / PS[0]
    """
    ipp = [float(x) for x in ds.ImagePositionPatient]
    ps  = [float(x) for x in ds.PixelSpacing]
    L, P = voxel_lp(ix, iy)
    col = (L - ipp[0]) / ps[1]
    row = (P - ipp[1]) / ps[0]
    return row, col

def vox_half_px(ds):
    """Half-width of a 31P voxel in pixels."""
    ps = float(ds.PixelSpacing[0])
    return (VOX_XY / ps) / 2.0

# ── Plot individual overlay figures ───────────────────────────────────────────
cmap_snr = plt.cm.hot
cmap_pcr = plt.cm.viridis

for iz in range(N_Z):
    z31p = sp31[iz]
    z_loc, arr, ds = find_closest_loc(z31p)
    diff_mm = abs(z31p - z_loc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#111111')
    for ax in axes:
        ax.set_facecolor('#111111')

    ipp = [float(x) for x in ds.ImagePositionPatient]
    ps  = float(ds.PixelSpacing[0])
    nr, nc = arr.shape

    # Extent: [xmin, xmax, ymin, ymax] for imshow — in pixel coords
    # row 0 at top, row nr-1 at bottom; col 0 at left, col nc-1 at right
    extent = [-0.5, nc - 0.5, nr - 0.5, -0.5]  # flip y so row0 at top

    for ax_idx, (ax, metric, cmap, vmax, label) in enumerate(zip(
        axes,
        [snr_map[:, :, iz], pcr_amp[:, :, iz]],
        [cmap_snr, cmap_pcr],
        [snr_vmax, pcr_vmax],
        ["SNR", "PCr amplitude"],
    )):
        # Show localizer
        lo, hi = np.percentile(arr, [2, 98])
        ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi,
                  extent=extent, origin="upper", aspect="equal")

        # Draw 31P FOV outline  (geometric center at POS_SAG, POS_COR)
        hw = (N_XY / 2.0) * (VOX_XY / ps)
        r_ctr, c_ctr = voxel_to_pixel(N_XY // 2, N_XY // 2, ds)
        fov_rect = mpatches.Rectangle(
            (c_ctr - hw, r_ctr - hw), 2*hw, 2*hw,
            linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--", alpha=0.6,
        )
        ax.add_patch(fov_rect)

        # Overlay 31P voxels as colored patches
        norm = Normalize(vmin=0, vmax=vmax)
        vh = vox_half_px(ds)
        for ix in range(N_XY):
            for iy in range(N_XY):
                val = metric[ix, iy]
                if val <= 0:
                    continue
                row_c, col_c = voxel_to_pixel(ix, iy, ds)
                color = cmap(norm(val))
                patch = mpatches.Rectangle(
                    (col_c - vh, row_c - vh), 2*vh, 2*vh,
                    linewidth=0.3, edgecolor="none", facecolor=color, alpha=0.55,
                )
                ax.add_patch(patch)

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label(label, color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xlim(-0.5, nc - 0.5)
        ax.set_ylim(nr - 0.5, -0.5)
        ax.set_title(
            f"iz={iz}  S={z31p:.1f} mm  |  Localizer: S={z_loc:.1f} mm  (Δ={diff_mm:.1f} mm)\n"
            f"31P {label} overlay (20 Hz apo)",
            color="white", fontsize=10, pad=6,
        )
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.set_xlabel("col (→L)", color="white", fontsize=8)
        ax.set_ylabel("row (↓P)", color="white", fontsize=8)

    fig.suptitle(f"31P MRSI overlay on TRA_localizer  —  z-slice {iz}",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"overlay_z{iz:02d}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Summary 3×3 grid (SNR only) ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.patch.set_facecolor('#111111')

norm_global = Normalize(vmin=0, vmax=snr_vmax)

for iz in range(N_Z):
    ax = axes[iz // 3, iz % 3]
    ax.set_facecolor('#111111')

    z31p = sp31[iz]
    z_loc, arr, ds = find_closest_loc(z31p)

    ipp = [float(x) for x in ds.ImagePositionPatient]
    ps  = float(ds.PixelSpacing[0])
    nr, nc = arr.shape
    extent = [-0.5, nc - 0.5, nr - 0.5, -0.5]

    lo, hi = np.percentile(arr, [2, 98])
    ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi,
              extent=extent, origin="upper", aspect="equal")

    vh = vox_half_px(ds)
    for ix in range(N_XY):
        for iy in range(N_XY):
            val = snr_map[ix, iy, iz]
            if val <= 0:
                continue
            row_c, col_c = voxel_to_pixel(ix, iy, ds)
            color = cmap_snr(norm_global(val))
            patch = mpatches.Rectangle(
                (col_c - vh, row_c - vh), 2*vh, 2*vh,
                linewidth=0, facecolor=color, alpha=0.55,
            )
            ax.add_patch(patch)

    diff_mm = abs(z31p - z_loc)
    ax.set_title(f"iz={iz}  S={z31p:.1f} mm  (loc {z_loc:.1f}, Δ{diff_mm:.1f})",
                 color="white", fontsize=8)
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.tick_params(colors="white", labelsize=6)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

# Shared colorbar
sm = ScalarMappable(cmap=cmap_snr, norm=norm_global)
sm.set_array([])
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
cb.set_label("SNR", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

fig.suptitle("31P MRSI SNR overlay on TRA_localizer — all 9 z-slices (20 Hz apo)",
             color="white", fontsize=13, y=1.00)
plt.tight_layout()
out_summary = os.path.join(OUT_DIR, "overlay_summary.png")
fig.savefig(out_summary, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_summary}")
print("Done.")

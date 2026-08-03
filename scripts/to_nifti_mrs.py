"""
Convert reconstructed 31P spectra (numpy) to NIfTI-MRS for viewing in FSLeyes.

Usage:
    python scripts/to_nifti_mrs.py [output_dir]

Default output_dir: ./output_phantom

Produces:
    <output_dir>/data/31p_fid.nii.gz   — FID (time domain), for FSLeyes / spec2nii
    <output_dir>/data/31p_spec.nii.gz  — Magnitude spectrum (real 4D), for simple viewing
"""
import sys
import os
import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs

# ── Config ─────────────────────────────────────────────────────────────────────
out_dir  = sys.argv[1] if len(sys.argv) > 1 else "./output_phantom"
data_dir = os.path.join(out_dir, "data")
spec_file = os.path.join(data_dir, "iter3d_recon_spec.npy")

# 31P phantom geometry
POS_SAG = -2.43567119472
POS_COR = 25.1286576107
POS_TRA = 0.0
FOV_MM  = 200.0
N       = 12
VOX_MM  = FOV_MM / N        # 16.667 mm
SLICE_MM = 50.0             # approximate slab thickness for display

dwell_s = 180e-9 * 2       # 360 µs
larmor  = 49.894571         # MHz

PHASE_VEC = np.array([0.0,  1.0, 0.0])   # LPS: +P direction
READ_VEC  = np.array([-1.0, 0.0, 0.0])   # LPS: -L direction
SNV       = np.array([0.0,  0.0, 1.0])   # LPS: +S (axial)

# ── Load & convert spec → FID ─────────────────────────────────────────────────
print(f"Loading {spec_file}")
spec = np.load(spec_file).astype(np.complex128)   # (Nx, Ny, Nz, nspec)
print(f"  spec shape: {spec.shape}")

# spec is stored in fftshifted frequency domain; ifftshift then ifft to get FID
fid = np.fft.ifft(np.fft.ifftshift(spec, axes=-1), axis=-1)

# ── Build affine (voxel → RAS) ────────────────────────────────────────────────
# dim 0 (ix) → PHASE_VEC direction; dim 1 (iy) → READ_VEC direction
# LPS → RAS: negate L and P (first two components)
def lps_to_ras(v):
    return np.array([-v[0], -v[1], v[2]])

# Origin = position of voxel [0, 0, 0] in LPS
center_lps = np.array([POS_SAG, POS_COR, POS_TRA])
origin_lps = (center_lps
              + PHASE_VEC * (0 - (N - 1) / 2.0) * VOX_MM
              + READ_VEC  * (0 - (N - 1) / 2.0) * VOX_MM)

affine = np.eye(4)
affine[:3, 0] = lps_to_ras(PHASE_VEC * VOX_MM)    # step per ix
affine[:3, 1] = lps_to_ras(READ_VEC  * VOX_MM)    # step per iy
affine[:3, 2] = lps_to_ras(SNV       * SLICE_MM)  # step per slice
affine[:3, 3] = lps_to_ras(origin_lps)             # origin in RAS

print(f"Affine:\n{affine}")

# ── Save NIfTI-MRS (FID, complex) ─────────────────────────────────────────────
nmrs = gen_nifti_mrs(fid, dwell_s, larmor, nucleus='31P', affine=affine)
out_fid = os.path.join(data_dir, "31p_fid.nii.gz")
nmrs.save(out_fid)
print(f"Saved FID: {out_fid}")

# ── Save magnitude spectrum as real 4D NIfTI (for simple overlay) ─────────────
import nibabel as nib

mag = np.abs(spec).astype(np.float32)   # (Nx, Ny, Nz, nspec)
nii = nib.Nifti1Image(mag, affine)
nii.header.set_xyzt_units('mm', 'sec')
# set TR = dwell_s so FSLeyes labels the time axis correctly
nii.header['pixdim'][4] = float(dwell_s)
out_mag = os.path.join(data_dir, "31p_spec_mag.nii.gz")
nib.save(nii, out_mag)
print(f"Saved magnitude spectrum: {out_mag}")

print("\nTo open in FSLeyes (run from WSL terminal):")
print(f"  /home/zsun/fsl/share/fsl/bin/fsleyes {out_fid}")
print(f"  or")
print(f"  /home/zsun/fsl/share/fsl/bin/fsleyes {out_mag}")
print("\nIn FSLeyes: Views → Time Series  to see spectrum at clicked voxel")

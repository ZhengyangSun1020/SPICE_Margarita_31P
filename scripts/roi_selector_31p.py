#!/usr/bin/env python3
"""
31P MRSI ROI selector — headless (no display needed).

Workflow:
  Step A: save a numbered grid image
    python scripts/roi_selector_31p.py \
        --localizer X.IMA --dat scan.dat [--matrix-size 12] \
        --save-grid grid.png

  Step B: open grid.png in Windows, note voxel numbers to INCLUDE, then:
    python scripts/roi_selector_31p.py \
        --localizer X.IMA --dat scan.dat [--matrix-size 12] \
        --include 22 23 24 32 33 34 --out roi_mask.npy

  Voxel numbering: left-to-right, bottom-to-top (row 0 = bottom = posterior).
  So voxel 0 = bottom-left, voxel N-1 = bottom-right, voxel N*(N-1) = top-left.

  LPS coordinate convention (matches overlay_31p_localizer.py):
    xi = column index: L(xi) = pos_sag + (xi - N//2) * vox_size
    yi = row index:    P(yi) = pos_cor - (yi - N//2) * vox_size
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── Localizer DICOM loader ─────────────────────────────────────────────────────

def load_localizer_with_geometry(path):
    """
    Load a transverse DICOM localizer and return image + display extent in LPS mm.

    Returns
    -------
    img_norm : (H, W) float32, already flipped so row-0 = bottom (posterior)
    extent   : [L_left, L_right, P_bottom, P_top]
               L increases → patient's left; P_bottom > P_top (y-axis inverted:
               posterior at bottom, anterior at top — standard radiology display)
    ds       : pydicom dataset (for geometry inspection)
    """
    import pydicom
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # Flip so row-0 is the bottom of the displayed image (origin='lower')
    # DICOM row-0 is the top; after [::-1] row-0 becomes the bottom.
    img = img[::-1, :]

    ipp = [float(x) for x in ds.ImagePositionPatient]   # [L, P, S] of first pixel
    ps  = float(ds.PixelSpacing[0])                      # mm / pixel (assume square)
    rows, cols = ds.Rows, ds.Columns

    L0 = ipp[0]   # L coordinate of column 0
    P0 = ipp[1]   # P coordinate of DICOM row 0 (= top)

    L_left  = L0
    L_right = L0 + cols * ps
    # After flipping:
    #   display row-0  = DICOM row (rows-1)  →  P = P0 + (rows-1)*ps  (posterior)
    #   display row-max = DICOM row 0         →  P = P0               (anterior)
    P_bottom = P0 + (rows - 1) * ps    # posterior, at bottom of display
    P_top    = P0                       # anterior,  at top    of display

    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    extent = [L_left, L_right, P_bottom, P_top]
    return img_norm, extent, ds


# ── MRSI geometry from .dat file ───────────────────────────────────────────────

def load_mrsi_geometry(dat_path, matrix_size_override=None):
    """
    Read MRSI position and FOV from a CONCEPT CRT .dat file (2D or 3D).

    Returns (N, pos_sag_mm, pos_cor_mm, fov_mm)
    where pos_sag = L position of VOI center, pos_cor = P position.
    """
    import importlib.util, pathlib
    here = pathlib.Path(__file__).parent

    # 05c uses load_concept_twix_3d from 06c for all 2D/3D dat files
    spec = importlib.util.spec_from_file_location(
        "_06c", here / "06c_iterative_recon_31p_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    info = mod.load_concept_twix_3d(
        dat_path,
        matrix_size_override=matrix_size_override)

    N   = int(info["matrix_size"])
    fov = float(info["fov_mm"])
    ps  = float(info["pos_sag"])
    pc  = float(info["pos_cor"])
    return N, ps, pc, fov


# ── Voxel → LPS coordinate conversion ─────────────────────────────────────────

def voxel_to_lps(xi, yi, N, pos_sag, pos_cor, vox_size):
    """
    Return (L_center, P_center) in mm for voxel at column xi, row yi.

    Convention (matches overlay_31p_localizer.py):
      xi = column index (0 = left = patient's right = −L)
      yi = row    index (0 = bottom = posterior = +P)

      L(xi) = pos_sag + (xi - (N-1)/2) * vox_size
      P(yi) = pos_cor − (yi − (N-1)/2) * vox_size
    """
    ctr = (N - 1) / 2.0   # voxel-center convention (correct for even N; matches MATLAB gold-standard grid)
    L = pos_sag + (xi - ctr) * vox_size
    P = pos_cor - (yi - ctr) * vox_size
    return L, P


# ── Common figure setup ────────────────────────────────────────────────────────

def _setup_figure(img, extent):
    """Return (fig, ax) with localizer image displayed in LPS world coordinates."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='k')
    ax.set_facecolor('k')

    # extent = [L_left, L_right, P_bottom, P_top]
    # P_bottom > P_top  →  y-axis inverted (posterior at bottom, anterior at top)
    ax.imshow(img, cmap='gray', origin='lower',
              extent=extent, aspect='equal')

    # Keep axis limits tight to the image
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])   # inverted: posterior at bottom

    ax.set_xlabel("L (mm, patient left = +)", color='gray', fontsize=8)
    ax.set_ylabel("P (mm, posterior = +, appears at bottom)", color='gray', fontsize=8)
    ax.tick_params(colors='gray', labelsize=6)
    return fig, ax


# ── Step A: save numbered grid image ──────────────────────────────────────────

def save_grid(localizer_path, N, out_png,
              dat_path=None,
              fov_mm_override=None, pos_sag_override=None, pos_cor_override=None,
              matrix_size_override=None):

    img, extent, _ds = load_localizer_with_geometry(localizer_path)

    # Get MRSI geometry
    if dat_path is not None:
        N_mrsi, pos_sag, pos_cor, fov_mm = load_mrsi_geometry(
            dat_path, matrix_size_override=matrix_size_override or N)
        N = N_mrsi  # trust the dat file
    else:
        fov_mm  = fov_mm_override  or 200.0
        pos_sag = pos_sag_override or 0.0
        pos_cor = pos_cor_override or 0.0

    vox_size = fov_mm / N

    print(f"[roi] N={N}, FOV={fov_mm:.0f} mm, vox={vox_size:.2f} mm")
    print(f"[roi] MRSI center: L={pos_sag:.1f} mm, P={pos_cor:.1f} mm")
    print(f"[roi] Localizer extent: L=[{extent[0]:.0f}, {extent[1]:.0f}], "
          f"P=[{extent[3]:.0f}..{extent[2]:.0f}] mm")

    fig, ax = _setup_figure(img, extent)

    for yi in range(N):
        for xi in range(N):
            voxel_id = yi * N + xi
            L_ctr, P_ctr = voxel_to_lps(xi, yi, N, pos_sag, pos_cor, vox_size)

            rect = patches.Rectangle(
                (L_ctr - vox_size / 2, P_ctr - vox_size / 2),
                vox_size, vox_size,
                linewidth=0.8, edgecolor='cyan', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            ax.text(L_ctr, P_ctr, str(voxel_id), color='yellow',
                    fontsize=max(4, int(vox_size * 0.3)),
                    ha='center', va='center', fontweight='bold')

    ax.set_title(
        f"Voxel numbers (0=bottom-left, {N*N-1}=top-right)  "
        f"FOV={fov_mm:.0f}mm  N={N}  vox={vox_size:.1f}mm\n"
        f"MRSI center: L={pos_sag:.1f}, P={pos_cor:.1f} mm  "
        f"(anterior=top, posterior=bottom, patient-left=right)",
        color='white', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='k')
    plt.close(fig)
    print(f"Saved grid image -> {out_png}")
    print(f"Open it, note which voxel numbers to INCLUDE in your ROI, then run:")
    print(f"  python scripts/roi_selector_31p.py --localizer \"...\" --dat \"...\" "
          f"--include <ids> --out roi_mask.npy")


# ── Step B: save mask + confirmation image ────────────────────────────────────

def save_mask(localizer_path, N, include_ids, out_npy,
              dat_path=None,
              fov_mm_override=None, pos_sag_override=None, pos_cor_override=None,
              matrix_size_override=None):

    # Get MRSI geometry
    if dat_path is not None:
        N_mrsi, pos_sag, pos_cor, fov_mm = load_mrsi_geometry(
            dat_path, matrix_size_override=matrix_size_override or N)
        N = N_mrsi
    else:
        fov_mm  = fov_mm_override  or 200.0
        pos_sag = pos_sag_override or 0.0
        pos_cor = pos_cor_override or 0.0

    vox_size = fov_mm / N

    mask = np.zeros((N, N), dtype=bool)
    for vid in include_ids:
        yi, xi = divmod(int(vid), N)
        if 0 <= xi < N and 0 <= yi < N:
            mask[xi, yi] = True          # mask[xi, yi] = col-first convention
        else:
            print(f"[warn] voxel {vid} out of range, skipped")

    np.save(out_npy, mask)
    n_in = int(mask.sum())
    print(f"Saved ROI mask ({n_in}/{N*N} included) -> {out_npy}")

    # Confirmation image
    img, extent, _ds = load_localizer_with_geometry(localizer_path)
    fig, ax = _setup_figure(img, extent)

    for yi in range(N):
        for xi in range(N):
            voxel_id = yi * N + xi
            included = bool(mask[xi, yi])
            L_ctr, P_ctr = voxel_to_lps(xi, yi, N, pos_sag, pos_cor, vox_size)

            color = 'cyan' if included else 'red'
            face  = 'none' if included else 'red'
            alpha = 0.6   if included else 0.25
            rect = patches.Rectangle(
                (L_ctr - vox_size / 2, P_ctr - vox_size / 2),
                vox_size, vox_size,
                linewidth=0.8, edgecolor=color, facecolor=face, alpha=alpha)
            ax.add_patch(rect)
            ax.text(L_ctr, P_ctr, str(voxel_id), color='yellow',
                    fontsize=max(4, int(vox_size * 0.3)),
                    ha='center', va='center', fontweight='bold')

    ax.set_title(
        f"ROI mask: {n_in}/{N*N} included (cyan) | excluded (red)\n"
        f"FOV={fov_mm:.0f}mm  N={N}  vox={vox_size:.1f}mm  "
        f"MRSI center: L={pos_sag:.1f}, P={pos_cor:.1f} mm",
        color='white', fontsize=8)

    confirm_png = out_npy.replace('.npy', '_confirm.png')
    fig.tight_layout()
    fig.savefig(confirm_png, dpi=150, bbox_inches='tight', facecolor='k')
    plt.close(fig)
    print(f"Saved confirmation image -> {confirm_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--localizer", required=True,
                   help="Path to a transverse DICOM file (single .IMA)")
    p.add_argument("--dat", default=None,
                   help="CONCEPT CRT .dat file — reads MRSI FOV and center position")
    p.add_argument("--matrix-size", type=int, default=None,
                   help="MRSI matrix size (override; default: read from --dat)")
    p.add_argument("--fov-mm", type=float, default=200.0,
                   help="MRSI FOV in mm (used only if --dat is not given)")
    p.add_argument("--pos-sag", type=float, default=0.0,
                   help="MRSI VOI center L offset in mm (used only if --dat not given)")
    p.add_argument("--pos-cor", type=float, default=0.0,
                   help="MRSI VOI center P offset in mm (used only if --dat not given)")
    p.add_argument("--save-grid", default=None,
                   help="(Step A) Save numbered grid PNG to this path")
    p.add_argument("--include", type=int, nargs="*", default=None,
                   help="(Step B) Voxel IDs to include in ROI")
    p.add_argument("--out", default="roi_mask.npy",
                   help="Output mask path (Step B)")
    args = p.parse_args()

    N = args.matrix_size or 12   # fallback

    kwargs = dict(
        dat_path=args.dat,
        fov_mm_override=args.fov_mm,
        pos_sag_override=args.pos_sag,
        pos_cor_override=args.pos_cor,
        matrix_size_override=args.matrix_size,
    )

    if args.save_grid:
        save_grid(args.localizer, N, args.save_grid, **kwargs)
    elif args.include is not None:
        save_mask(args.localizer, N, args.include, args.out, **kwargs)
    else:
        grid_png = args.out.replace('.npy', '_grid.png')
        print("[roi] No mode specified — saving numbered grid image ...")
        save_grid(args.localizer, N, grid_png, **kwargs)


if __name__ == "__main__":
    main()

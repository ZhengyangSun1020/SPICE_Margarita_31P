#!/usr/bin/env python3
"""08 — 31P spectral fitting via FSL-MRS (fsl_mrsi), mirroring the 1H specfit_01.

Pipeline (same idea as scripts/specfitting/specfit_01_fsl_mrsi_fit.py, adapted to 31P):
  1. load a 05c/06c reconstruction FID  (iter3d_recon_fid_repNN.npy, complex Nx,Ny,Nz,ns)
  2. wrap it as NIFTI-MRS (nucleus 31P, centralFreq from .dat, dwell = 1/BW)
  3. build a PCr-amplitude mask (the 31P analogue of the 1H water/brain mask)
  4. run  fsl_mrsi  with a 31P basis (make_31p_basis.py)
  5. load metabolite concentration maps, plot, and symlink results for FSLeyes

Spectral axis is set only by centralFrequency and BW (fixed 2778 Hz), as requested.

Prereq (once):
    python scripts/specfitting/make_31p_basis.py --larmor <MHz> --out basis_31p_3T

Usage:
    python scripts/08_fsl_mrsi_fit_31p.py \
        --dir output_dynamic_3T_MID00644_10x10_5rings_dft_adj --rep 26 \
        --dat "/mnt/f/.../meas_MID00644_...5rings.dat" \
        --basis basis_31p_3T --ppmlim -20 10
"""
import argparse, os, shutil, subprocess, sys, importlib.util as ilu
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from warnings import filterwarnings; filterwarnings("ignore")
from nifti_mrs.create_nmrs import gen_nifti_mrs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def FIDToSpec(x, axis=-1):
    return np.fft.fftshift(np.fft.fft(x, axis=axis), axes=axis)

def load_geom(dat):
    """Return (N, pos_sag L, pos_cor P, fov mm, pos_tra S, larmor MHz, fov_z mm)."""
    s = ilu.spec_from_file_location("_06c", os.path.join(ROOT,"scripts","06c_iterative_recon_31p_3d.py"))
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    info = m.load_concept_twix_3d(dat)
    return (int(info["matrix_size"]), float(info["pos_sag"]), float(info["pos_cor"]),
            float(info["fov_mm"]), float(info["pos_tra"]),
            float(info["larmor_freq_mhz"]), float(info.get("fov_z_mm", 25.0)))

def build_localizer_nifti(dicom_path, out_path):
    """Convert a localizer DICOM to NIfTI with an LPS->RAS affine built the SAME way
    as build_affine (from IPP/IOP/PixelSpacing), so it co-registers with the MRSI map."""
    import pydicom
    d = pydicom.dcmread(dicom_path)
    px = d.pixel_array.astype(np.float32)                 # [rows(i), cols(j)]
    iop = np.array(d.ImageOrientationPatient, float)
    ipp = np.array(d.ImagePositionPatient, float)
    ps_r, ps_c = float(d.PixelSpacing[0]), float(d.PixelSpacing[1])
    e_col = iop[0:3]                                       # dir of increasing column (j)
    e_row = iop[3:6]                                       # dir of increasing row (i)
    thick = float(getattr(d, "SliceThickness", 5.0))
    n = np.cross(e_col, e_row)                            # slice normal (+S for transverse)
    L2R = np.diag([-1.0, -1.0, 1.0])                      # LPS -> RAS
    A = np.eye(4)
    A[:3, 0] = L2R @ (ps_r * e_row)
    A[:3, 1] = L2R @ (ps_c * e_col)
    A[:3, 2] = L2R @ (thick * n)
    A[:3, 3] = L2R @ ipp
    nib.save(nib.Nifti1Image(px[:, :, np.newaxis], A), out_path)
    return A


def build_affine(N, ps, pc, S, vox, thick):
    """voxel(i,j,k)->RAS mm. Symmetric even-N centering c=(N-1)/2 so the N-voxel grid
    tiles the FOV symmetrically (edges at +/-N*vox/2 about the VOI centre, matching the
    scanner localizer FOV). L=ps+(i-c)vox, P=pc-(j-c)vox, S const; RAS=(-L,-P,S)."""
    c = (N - 1) / 2.0
    A = np.eye(4)
    A[0,0] = -vox; A[0,3] = -ps + c*vox
    A[1,1] =  vox; A[1,3] = -pc - c*vox
    A[2,2] =  thick; A[2,3] = S
    return A

def main():
    ap = argparse.ArgumentParser(description="08 - 31P fsl_mrsi fitting wrapper")
    ap.add_argument("--dir", default=None, help="06c out-dir (uses data/iter3d_recon_fid_repNN.npy)")
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--fid", default=None, help="explicit recon FID npy (Nx,Ny[,Nz],ns) complex")
    ap.add_argument("--dat", default=None, help=".dat for centralFreq + geometry")
    ap.add_argument("--larmor", type=float, default=None, help="override central freq MHz")
    ap.add_argument("--bw", type=float, default=2778.0)
    ap.add_argument("--basis", required=True, help="31P basis dir (make_31p_basis.py)")
    ap.add_argument("--ppmlim", type=float, nargs=2, default=[-20.0, 10.0])
    ap.add_argument("--baseline", default="poly, 2")
    ap.add_argument("--mask-frac", type=float, default=0.30)
    ap.add_argument("--internal-ref", default="PCr", help="internal conc reference (31P: PCr)")
    ap.add_argument("--metab-groups", default="separate_all",
                    help="fsl_mrsi metabolite grouping. 31P NEEDS separate groups: PCr is narrow "
                         "and Pi/ATP broader, so one shared linewidth suppresses them (fit -> 0).")
    ap.add_argument("--no-rescale", action="store_true",
                    help="pass --no_rescale to fsl_mrsi (default: rescale ON, helps small peaks fit)")
    ap.add_argument("--localizer", default=None, help="localizer DICOM (.IMA) -> NIfTI for FSLeyes overlay")
    ap.add_argument("--no-transpose-xy", action="store_true",
                    help="do NOT transpose the FID xy into the (xi=L, yi=P) frame that build_affine expects")
    ap.add_argument("--combine", nargs="+", action="append", default=[],
                    help="combine groups, repeatable (e.g. --combine ATP_gamma ATP_alpha ATP_beta)")
    ap.add_argument("--conj-data", action="store_true", help="conjugate FID before NIFTI-MRS")
    ap.add_argument("--conj-basis", action="store_true", help="pass --conj_basis to fsl_mrsi")
    ap.add_argument("--no-conj-fid", action="store_true", help="pass --no_conj_fid to fsl_mrsi")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ── load recon FID ──
    if args.fid:
        fid = np.load(args.fid)
    elif args.dir is not None and args.rep is not None:
        fid = np.load(os.path.join(args.dir,"data",f"iter3d_recon_fid_rep{args.rep:02d}.npy"))
    else:
        ap.error("provide --fid, or --dir with --rep")
    fid = np.asarray(fid)
    if fid.ndim == 3: fid = fid[:, :, None, :]
    if not args.no_transpose_xy:
        fid = fid.transpose(1, 0, 2, 3)          # native -> (xi=L, yi=P) to match build_affine / roi_selector
    Nx, Ny, Nz, ns = fid.shape
    if args.conj_data: fid = fid.conj()

    # ── geometry / central freq ──
    if args.dat:
        N, ps, pc, fov, S, larmor_dat, fov_z = load_geom(args.dat)
    else:
        N, ps, pc, fov, S, larmor_dat, fov_z = Nx, 0.0, 0.0, float(Nx*20), 0.0, None, 25.0
    larmor = args.larmor or larmor_dat
    if larmor is None: ap.error("need --larmor or --dat")
    vox = fov / N
    affine = build_affine(N, ps, pc, S, vox, fov_z)
    dwell = 1.0 / args.bw
    out = args.out or ((args.dir or ".").rstrip("/") + f"_fit31p_rep{args.rep if args.rep is not None else 0}")
    os.makedirs(out, exist_ok=True)
    print(f"[08] grid {Nx}x{Ny}x{Nz} ns={ns} BW={args.bw:.0f} larmor={larmor:.3f}MHz vox={vox:.1f} -> {out}")

    # ── NIFTI-MRS data ──
    nmrs = gen_nifti_mrs(fid.astype(np.complex64), dwell, larmor, nucleus="31P", affine=affine)
    data_nii = os.path.join(out, "spice31p.nii.gz")
    nmrs.save(data_nii)
    print(f"[08] wrote NIFTI-MRS -> {data_nii}")

    # ── PCr-amplitude mask ──
    PPM = -np.fft.fftshift(np.fft.fftfreq(ns, d=dwell)) / larmor
    spec = np.abs(FIDToSpec(fid.conj()))
    pcr_m = np.abs(PPM) <= 0.8
    pcr_amp = spec[:, :, :, pcr_m].max(axis=-1)          # (Nx,Ny,Nz)
    mask = pcr_amp >= args.mask_frac * pcr_amp.max()
    mask_nii = os.path.join(out, "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), mask_nii)
    print(f"[08] mask: {int(mask.sum())} voxels (frac={args.mask_frac}) -> {mask_nii}")

    # ── run fsl_mrsi ──
    fsl_out = os.path.join(out, "fit")
    if os.path.exists(fsl_out): shutil.rmtree(fsl_out)
    cmd = ["fsl_mrsi", "--data", data_nii, "--basis", args.basis, "--mask", mask_nii,
           "--baseline", args.baseline, "--ppmlim", str(args.ppmlim[0]), str(args.ppmlim[1]),
           "--output", fsl_out, "--overwrite", "--report",
           "--internal_ref", args.internal_ref, "--metab_groups", args.metab_groups]
    if args.no_rescale:  cmd.append("--no_rescale")
    if args.conj_basis:  cmd.append("--conj_basis")
    if args.no_conj_fid: cmd.append("--no_conj_fid")
    for g in (args.combine or []): cmd += ["--combine"] + list(g)
    print("[08] running:", " ".join(cmd))
    subprocess.run(cmd, env=os.environ.copy(), check=False)   # tolerate Dask cleanup errors
    if not os.path.isdir(os.path.join(fsl_out, "concs")):
        raise RuntimeError("fsl_mrsi produced no concs/ — check the log above")
    print(f"[08] fsl_mrsi done -> {fsl_out}")

    # ── load & plot concentration maps ──
    from pathlib import Path
    conc_dir = Path(fsl_out)/"concs"/"internal"
    conc = {}
    if conc_dir.exists():
        for f in sorted(conc_dir.glob("*.nii*")):
            nm = f.name.split(".nii")[0]
            conc[nm] = np.squeeze(nib.load(str(f)).get_fdata())
        print("[08] fitted metabolites:", sorted(conc.keys()))
        show = [k for k in ["PCr","Pi","Pi2","PDE","ATP_gamma","ATP_alpha","ATP_beta"] if k in conc]
        n = len(show) or 1
        fig, axs = plt.subplots(1, n, figsize=(3.2*n, 3.2), squeeze=False)
        for ax, k in zip(axs[0], show):
            M = np.where(mask.squeeze(), conc[k].squeeze(), np.nan)
            im = ax.imshow(M.T, origin="lower", cmap="viridis",
                           vmax=np.nanpercentile(M,98) if np.isfinite(M).any() else None)
            ax.set_title(k, fontsize=9); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"08 31P fsl_mrsi — {os.path.basename(out)}", fontsize=11); fig.tight_layout()
        fig.savefig(os.path.join(out,"conc_maps.png"), dpi=140); plt.close(fig)
        print(f"[08] conc maps -> {out}/conc_maps.png")
    else:
        print(f"[warn] no concs dir at {conc_dir}")

    # ── localizer -> NIfTI (dcm2niix) for FSLeyes overlay ──
    loc_nii = None
    if args.localizer:
        loc_nii = os.path.join(out, "localizer.nii.gz")
        build_localizer_nifti(args.localizer, loc_nii)   # affine matches build_affine (co-registers)
        if loc_nii:
            print(f"[08] localizer NIfTI -> {loc_nii}")
            # registration check: place fit PCr map by its OWN affine (RAS->LPS) on the
            # localizer in world-mm — the same frame FSLeyes uses (trustworthy, not a display transpose)
            try:
                rs_s = ilu.spec_from_file_location("rs", os.path.join(ROOT,"scripts","roi_selector_31p.py"))
                rs = ilu.module_from_spec(rs_s); rs_s.loader.exec_module(rs)
                pcr_path = os.path.join(fsl_out, "concs", "internal", "PCr.nii.gz")
                img_l, ext_l, _ = rs.load_localizer_with_geometry(args.localizer)
                pimg = nib.load(pcr_path); Aff = pimg.affine; M = np.squeeze(pimg.get_fdata())
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(img_l, cmap="gray", origin="lower", extent=ext_l, aspect="equal")
                ax.set_xlim(ps+fov/2, ps-fov/2) if ext_l[0]>ext_l[1] else ax.set_xlim(ps-fov/2, ps+fov/2)
                ax.set_ylim(pc+fov/2, pc-fov/2) if ext_l[2]>ext_l[3] else ax.set_ylim(pc-fov/2, pc+fov/2)
                g = np.nanmax(M)+1e-12
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        ras = Aff @ np.array([i, j, 0, 1.0]); Lw = -ras[0]; Pw = -ras[1]
                        v = M[i, j]/g
                        if v > 0.3:
                            ax.add_patch(plt.Rectangle((Lw-vox/2, Pw-vox/2), vox, vox, color='red', alpha=0.3*v+0.15))
                ax.set_title("PCr on localizer (world-coord affine check = FSLeyes frame)", fontsize=9)
                ax.set_facecolor('k'); ax.tick_params(colors='w')
                fig.savefig(os.path.join(out,"reg_check.png"), dpi=140, bbox_inches="tight", facecolor='k'); plt.close(fig)
                print(f"[08] registration-check overlay -> {out}/reg_check.png")
            except Exception as e:
                print(f"[warn] reg-check overlay failed: {e}")

    print(f"\n[08] To visualise in FSLeyes (WSLg):")
    if loc_nii:
        print(f"    fsleyes {loc_nii} {os.path.join(fsl_out,'concs','internal','PCr.nii.gz')} -cm hot -a 50 "
              f"{os.path.join(fsl_out,'concs','internal','Pi.nii.gz')} -cm blue-lightblue -a 50 &")
    print(f"    fsleyes {fsl_out} &     # full MRS-plugin view (data/fit/residual per voxel)")
    print(f"    (HTML report: {os.path.join(fsl_out,'report.html')})")

if __name__ == "__main__":
    main()

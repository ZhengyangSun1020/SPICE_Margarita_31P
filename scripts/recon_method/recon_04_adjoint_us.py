#!/usr/bin/env python3
"""
recon_04_adjoint_us.py — Monte Carlo adjoint recon for undersampled k-space.

Takes a reference undersampled dataset (e.g. invivo_260623_01_us25) and a list
of full-data source subjects.  For each source, applies the SAME ring mask as
the reference, then runs lipid removal and adjoint reconstruction.  Results land
in a single output folder for downstream MC uncertainty analysis.

The ring mask is read from <ref-us-dir>/ring_mask.npy (written by
data_proc_06_undersample.py).  If that file is absent (old runs), the mask is
re-derived by matching k-space trajectory fingerprints.

Folder layout
-------------
output/<ref_name>/adjoint_recon_us/
  ring_mask.npy
  <src_name>/                  one per MC source subject
    data/                      temp processed-data directory
      mrsi_data.npy            undersampled, same ring mask as ref
      mrsi_ksp.npy             undersampled
      wref_o.npy  -> ...       symlinks to src processed-data
      affine.npy  -> ...
      scan_params.json -> ...
      sigma_noise.npy -> ...
    coilmap  ->  ../../coilmap   symlink to ref output coilmap
    b0map    ->  ../../b0map     symlink to ref output b0map
    lipid_removal/               output of data_proc_04
    adjoint_test/                output of recon_02

Usage:
    python scripts/recon_method/recon_04_adjoint_us.py \
        --ref-us-dir   data/processed/invivo_260623_01_us25 \
        --mc-src-dirs  data/processed/invivo_260623_01 \
                       data/processed/invivo_260623_02 \
                       data/processed/invivo_260623_03 \
                       data/processed/invivo_260623_04 \
                       data/processed/invivo_260623_05 \
        --basis-dir    ./basis/ \
        --out-root     output \
        --n-coils      32 \
        --dim          64 64 \
        --n-seq-points 300 \
        --brain-threshold 0.16 \
        --rank         20

    # skip lipid removal if already done (re-run adjoint recon only)
    python scripts/recon_method/recon_04_adjoint_us.py \
        --ref-us-dir   data/processed/invivo_260623_01_us25 \
        --mc-src-dirs  data/processed/invivo_260623_02 \
        --basis-dir    ./basis/ \
        --skip-lprm
"""

import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path

_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_SYMLINKS = [
    "wref_o.npy",
    "wref_data.npy",
    "wref_ksp.npy",
    "wref_o_check.png",
    "affine.npy",
    "sigma_noise.npy",
    "scan_params.json",
]


# ── ring mask extraction (fallback when ring_mask.npy is absent) ──────────────

def _extract_ring_mask(ref_ksp, full_ksp):
    """Find which columns of full_ksp appear in ref_ksp.

    Uses a random-projection fingerprint (robust to rings with equal radius).
    Returns keep_idx of shape (N_keep,).
    """
    K        = full_ksp.shape[1]
    N_total  = full_ksp.shape[2]
    N_keep   = ref_ksp.shape[2]

    rng     = np.random.default_rng(0)
    weights = rng.random((2, K))                              # (2, K)
    fp_full = np.einsum('tk,tkn->n', weights, full_ksp[:2])  # (N_total,)
    fp_ref  = np.einsum('tk,tkn->n', weights, ref_ksp[:2])   # (N_keep,)

    keep_idx = np.array([np.argmin(np.abs(fp_full - fp)) for fp in fp_ref])
    n_unique = len(np.unique(keep_idx))
    if n_unique != N_keep:
        raise RuntimeError(
            f"Ring mask extraction: expected {N_keep} unique matches but got {n_unique}. "
            "Re-run data_proc_06_undersample.py to regenerate ring_mask.npy.")
    return keep_idx


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MC adjoint recon for undersampled k-space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref-us-dir",  required=True,
                   help="Reference undersampled processed-data folder "
                        "(e.g. data/processed/invivo_260623_01_us25); "
                        "defines the ring mask and the coilmap/b0map to reuse")
    p.add_argument("--mc-src-dirs", nargs="+", required=True,
                   help="Full-data processed-data folders for each MC source subject "
                        "(e.g. data/processed/invivo_260623_01 02 03 04 05)")
    p.add_argument("--out-root",    default="output",
                   help="Pipeline output root")
    p.add_argument("--basis-dir",   default="./basis/",
                   help="FSL-MRS basis directory for adjoint recon xcorr alignment")

    # pipeline control
    p.add_argument("--skip-lprm",   action="store_true",
                   help="Skip lipid removal (use existing lipid_removal/ outputs)")
    p.add_argument("--skip-adj",    action="store_true",
                   help="Skip adjoint recon (re-run only lipid removal)")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-run stages even if outputs already exist")

    # lipid removal params (passed through to data_proc_04)
    p.add_argument("--n-coils",         type=int,   default=32)
    p.add_argument("--dim",             type=int,   nargs=2, default=[64, 64])
    p.add_argument("--n-seq-points",    type=int,   default=300)
    p.add_argument("--ppm-center",      type=float, default=3.027)
    p.add_argument("--brain-threshold", type=float, default=0.16)
    p.add_argument("--lipid-beta",      type=float, default=200.0)
    p.add_argument("--n-lipid-voxels",  type=int,   default=500)
    p.add_argument("--nsigma-gmm",      type=float, default=0.2)
    p.add_argument("--lipid-rank",      type=int,   default=10)
    p.add_argument("--topn-fallback",   type=int,   default=100)
    p.add_argument("--phase-ppmlim",    type=float, nargs=2, default=[3.5, 3.9])
    p.add_argument("--phase-method",    default="none")
    p.add_argument("--plot-voxel",      type=int,   nargs=2, default=[41, 24])
    p.add_argument("--save-plots",      action="store_true")

    # adjoint recon params (passed through to recon_02)
    p.add_argument("--rank",            type=int,   default=20)
    p.add_argument("--align-method",    default="xcorr", choices=["xcorr", "phase_corr"])
    p.add_argument("--no-b0",           action="store_true")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _symlink(src, dst):
    """Create relative symlink dst → src, replacing any existing link."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    target = os.path.relpath(str(src), str(dst.parent))
    dst.symlink_to(target)


def _symlink_dir(src, dst):
    """Symlink a directory dst → src, removing any existing dir/link."""
    if dst.is_symlink():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    if src.exists():
        target = os.path.relpath(str(src), str(dst.parent))
        dst.symlink_to(target)
        return True
    return False


def _run(cmd, label):
    print(f"\n[adj-us] {label}")
    print(f"[adj-us] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(_root), check=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    ref_us_dir = Path(args.ref_us_dir).resolve()
    out_root   = Path(args.out_root)

    if not ref_us_dir.is_dir():
        raise FileNotFoundError(f"--ref-us-dir not found: {ref_us_dir}")

    # ── Output root for MC collection ────────────────────────────────────────
    ref_us_name = ref_us_dir.name
    ref_out     = out_root / ref_us_name          # e.g. output/invivo_260623_01_us25
    adj_us_dir  = ref_out / "adjoint_recon_us"
    adj_us_dir.mkdir(parents=True, exist_ok=True)
    print(f"[adj-us] Ref US dir:  {ref_us_dir}")
    print(f"[adj-us] MC output:   {adj_us_dir}")

    # ── Ring mask ────────────────────────────────────────────────────────────
    ring_mask_path = ref_us_dir / "ring_mask.npy"
    if ring_mask_path.exists():
        keep_idx = np.load(str(ring_mask_path))
        print(f"[adj-us] Ring mask from ring_mask.npy: {len(keep_idx)} rings kept")
    else:
        print("[adj-us] ring_mask.npy not found — extracting from ksp fingerprint …")
        ref_ksp  = np.load(str(ref_us_dir / "mrsi_ksp.npy"))
        full_ksp = np.load(str(Path(args.mc_src_dirs[0]).resolve() / "mrsi_ksp.npy"))
        keep_idx = _extract_ring_mask(ref_ksp, full_ksp)
        print(f"[adj-us] Extracted ring mask: {len(keep_idx)} rings kept  "
              f"(tip: re-run data_proc_06 to save ring_mask.npy)")

    # Save ring mask into adjoint_recon_us for reference
    np.save(str(adj_us_dir / "ring_mask.npy"), keep_idx)
    print(f"[adj-us] Saved ring_mask.npy → {adj_us_dir}/ring_mask.npy")

    # ── Coilmap / b0map from ref output ──────────────────────────────────────
    ref_coilmap = ref_out / "coilmap"
    ref_b0map   = ref_out / "b0map"
    if not ref_coilmap.exists():
        print(f"[adj-us] WARNING: coilmap not found at {ref_coilmap}. "
              "Run data_proc_02/03 on --ref-us-dir first.")
    if not ref_b0map.exists():
        print(f"[adj-us] WARNING: b0map not found at {ref_b0map}. "
              "Run data_proc_03 on --ref-us-dir first.")

    # ── Per-subject MC loop ───────────────────────────────────────────────────
    for src_path in args.mc_src_dirs:
        src_dir  = Path(src_path).resolve()
        mc_name  = src_dir.name
        mc_dir   = adj_us_dir / mc_name
        data_dir = mc_dir / "data"
        mc_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[adj-us] MC subject: {mc_name}")
        print(f"[adj-us] MC out:     {mc_dir}")

        # ── Undersample data ─────────────────────────────────────────────────
        us_data_path = data_dir / "mrsi_data.npy"
        us_ksp_path  = data_dir / "mrsi_ksp.npy"

        if args.overwrite or not us_data_path.exists():
            mrsi_data = np.load(str(src_dir / "mrsi_data.npy"))
            mrsi_ksp  = np.load(str(src_dir / "mrsi_ksp.npy"))
            us_data   = mrsi_data[:, :, keep_idx, :]
            us_ksp    = mrsi_ksp[:, :, keep_idx]
            np.save(str(us_data_path), us_data)
            np.save(str(us_ksp_path),  us_ksp)
            print(f"[adj-us] mrsi_data: {mrsi_data.shape} → {us_data.shape}")
            print(f"[adj-us] mrsi_ksp:  {mrsi_ksp.shape}  → {us_ksp.shape}")
        else:
            print(f"[adj-us] data/ already exists — skip slicing (use --overwrite to redo)")

        # ── Symlink shared files from source ─────────────────────────────────
        for fname in DATA_SYMLINKS:
            src_file = src_dir / fname
            link     = data_dir / fname
            if src_file.exists() or src_file.is_symlink():
                _symlink(src_file, link)

        # ── Symlink coilmap + b0map from ref output ───────────────────────────
        for subdir, ref_path in [("coilmap", ref_coilmap), ("b0map", ref_b0map)]:
            ok = _symlink_dir(ref_path, mc_dir / subdir)
            if ok:
                print(f"[adj-us] {subdir} → {ref_path}")
            else:
                print(f"[adj-us] WARNING: {subdir} target {ref_path} does not exist")

        # ── Lipid removal ────────────────────────────────────────────────────
        lprm_done = (mc_dir / "lipid_removal" / "kt_mrsi_lprm.npy").exists()
        if args.skip_lprm:
            print("[adj-us] --skip-lprm: skipping lipid removal")
        elif lprm_done and not args.overwrite:
            print("[adj-us] lipid_removal/ already done — skip (use --overwrite to redo)")
        else:
            lprm_cmd = [
                sys.executable, "scripts/data_preproc/data_proc_04_lipid_removal.py",
                "--data-dir",        os.path.relpath(str(data_dir), str(_root)),
                "--out-dir",         os.path.relpath(str(mc_dir),   str(_root)),
                "--n-coils",         str(args.n_coils),
                "--dim",             str(args.dim[0]), str(args.dim[1]),
                "--n-seq-points",    str(args.n_seq_points),
                "--ppm-center",      str(args.ppm_center),
                "--brain-threshold", str(args.brain_threshold),
                "--lipid-beta",      str(args.lipid_beta),
                "--n-lipid-voxels",  str(args.n_lipid_voxels),
                "--nsigma-gmm",      str(args.nsigma_gmm),
                "--lipid-rank",      str(args.lipid_rank),
                "--topn-fallback",   str(args.topn_fallback),
                "--phase-ppmlim",    str(args.phase_ppmlim[0]), str(args.phase_ppmlim[1]),
                "--phase-method",    args.phase_method,
                "--plot-voxel",      str(args.plot_voxel[0]), str(args.plot_voxel[1]),
            ]
            if args.save_plots:
                lprm_cmd.append("--save-plots")
            _run(lprm_cmd, f"Lipid removal — {mc_name}")

        # ── Adjoint recon ────────────────────────────────────────────────────
        adj_done = (mc_dir / "adjoint_test" / "adj_recon_aligned.nii.gz").exists()
        if args.skip_adj:
            print("[adj-us] --skip-adj: skipping adjoint recon")
        elif adj_done and not args.overwrite:
            print("[adj-us] adjoint_test/ already done — skip (use --overwrite to redo)")
        else:
            adj_cmd = [
                sys.executable, "scripts/recon_method/recon_02_adjoint_recon.py",
                "--data-dir",        os.path.relpath(str(data_dir), str(_root)),
                "--out-dir",         os.path.relpath(str(mc_dir),   str(_root)),
                "--basis-dir",       args.basis_dir,
                "--rank",            str(args.rank),
                "--dim",             str(args.dim[0]), str(args.dim[1]),
                "--n-seq-points",    str(args.n_seq_points),
                "--brain-threshold", str(args.brain_threshold),
                "--align-method",    args.align_method,
            ]
            if args.no_b0:
                adj_cmd.append("--no-b0")
            _run(adj_cmd, f"Adjoint recon — {mc_name}")

        print(f"[adj-us] Done: {mc_name}")

    print(f"\n{'='*60}")
    print(f"[adj-us] All MC subjects complete.")
    print(f"[adj-us] Results in: {adj_us_dir}")
    print(f"[adj-us] Next: run Uncert_08_MC_adjoint_us.py --adj-us-dir {adj_us_dir}")


if __name__ == "__main__":
    main()

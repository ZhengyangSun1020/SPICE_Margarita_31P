#!/usr/bin/env python3
"""
data_proc_06_undersample.py — Simulate k-space ring undersampling.

Loads mrsi_data.npy and mrsi_ksp.npy from a processed-data folder,
randomly discards a fraction of rings, and writes a new processed-data
folder with the undersampled data. All shared files (wref, scan_params,
sigma_noise, affine) are symlinked from the source folder so the full
downstream pipeline works unchanged.

Ring selection
--------------
The --n-protect innermost rings (by mean kx/ky radius) are never
discarded.  From the remaining pool, floor(rate * N_total_rings) rings
are discarded at random.

Folder naming
-------------
Default: <src_name>_us<int(rate*100)>
  e.g.  invivo_260623_01  + rate=0.25  →  invivo_260623_01_us25
Override with --out-dir.

Usage:
    python scripts/data_preproc/data_proc_06_undersample.py \
        --src-dir  data/processed/invivo_260623_01 \
        --rate     0.25 \
        --seed     42

    # combine step output as source
    python scripts/data_preproc/data_proc_06_undersample.py \
        --src-dir  data/processed/invivo_260623_comb_01_02_03 \
        --rate     0.50 \
        --seed     0

    # custom output location
    python scripts/data_preproc/data_proc_06_undersample.py \
        --src-dir  data/processed/invivo_260623_01 \
        --rate     0.25 \
        --out-dir  data/processed/invivo_260623_01_us25_seed7 \
        --seed     7
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

SYMLINK_FILES = [
    "wref_data.npy",
    "wref_ksp.npy",
    "wref_o.npy",
    "wref_o_check.png",
    "affine.npy",
    "sigma_noise.npy",
    "scan_params.json",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate k-space ring undersampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src-dir",   required=True,
                   help="Source processed-data folder "
                        "(e.g. data/processed/invivo_260623_01)")
    p.add_argument("--rate",      type=float, required=True,
                   help="Fraction of rings to discard, e.g. 0.25 discards "
                        "25%% of total rings (protected centre rings excluded "
                        "from discarding)")
    p.add_argument("--n-protect", type=int, default=5,
                   help="Number of innermost k-space rings to always keep")
    p.add_argument("--seed",      type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--out-dir",   default=None,
                   help="Override output folder path (default: "
                        "<parent>/<src_name>_us<int(rate*100)>)")
    p.add_argument("--data-root", default=None,
                   help="Parent directory for the output folder when --out-dir "
                        "is not set (default: same parent as --src-dir)")
    return p.parse_args()


def main():
    args    = parse_args()
    src_dir = Path(args.src_dir).resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not (0.0 < args.rate < 1.0):
        raise ValueError(f"--rate must be in (0, 1), got {args.rate}")

    # ── Output directory ──────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        data_root = Path(args.data_root).resolve() if args.data_root else src_dir.parent
        suffix    = f"_us{int(args.rate * 100)}"
        out_dir   = data_root / (src_dir.name + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[undersample] Source:  {src_dir}")
    print(f"[undersample] Output:  {out_dir}")
    print(f"[undersample] Rate:    {args.rate}  ({args.rate*100:.0f}% of total rings discarded)")
    print(f"[undersample] Protect: {args.n_protect} innermost rings")
    if args.seed is not None:
        print(f"[undersample] Seed:    {args.seed}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[undersample] Loading mrsi_data.npy and mrsi_ksp.npy …")
    mrsi_data = np.load(str(src_dir / "mrsi_data.npy"))   # (1, K, N_rings, C)
    mrsi_ksp  = np.load(str(src_dir / "mrsi_ksp.npy"))    # (3, K, N_rings)

    n_rings = mrsi_data.shape[2]
    if mrsi_ksp.shape[2] != n_rings:
        raise ValueError(
            f"Ring count mismatch: mrsi_data has {n_rings} rings "
            f"but mrsi_ksp has {mrsi_ksp.shape[2]}")

    print(f"[undersample] mrsi_data  shape={mrsi_data.shape}  dtype={mrsi_data.dtype}")
    print(f"[undersample] mrsi_ksp   shape={mrsi_ksp.shape}   dtype={mrsi_ksp.dtype}")

    # ── Identify protected (innermost) rings by mean k-space radius ───────────
    kx = mrsi_ksp[0]   # (K, N_rings)
    ky = mrsi_ksp[1]   # (K, N_rings)
    ring_radius = np.sqrt(kx**2 + ky**2).mean(axis=0)   # (N_rings,) mean radius

    sorted_by_radius = np.argsort(ring_radius)
    n_protect  = min(args.n_protect, n_rings)
    protected  = set(sorted_by_radius[:n_protect].tolist())
    candidates = [i for i in range(n_rings) if i not in protected]

    print(f"\n[undersample] Ring radii (min→max): "
          f"{ring_radius.min():.4f} … {ring_radius.max():.4f}")
    print(f"[undersample] Protected ring indices: {sorted(protected)}")
    print(f"[undersample] Protected ring radii:   "
          f"{ring_radius[sorted_by_radius[:n_protect]]}")

    # ── Choose rings to discard ───────────────────────────────────────────────
    n_discard = int(args.rate * n_rings)   # floor
    if n_discard > len(candidates):
        raise ValueError(
            f"Cannot discard {n_discard} rings: only {len(candidates)} "
            f"non-protected rings available "
            f"(total={n_rings}, protected={n_protect}). "
            f"Reduce --rate or --n-protect.")

    rng = np.random.default_rng(args.seed)
    discard_idx = set(rng.choice(candidates, size=n_discard, replace=False).tolist())
    keep_idx    = sorted(i for i in range(n_rings) if i not in discard_idx)
    n_keep      = len(keep_idx)

    print(f"\n[undersample] Total rings:     {n_rings}")
    print(f"[undersample] Discarded rings: {n_discard}")
    print(f"[undersample] Kept rings:      {n_keep}")

    # ── Slice and save ────────────────────────────────────────────────────────
    keep_arr = np.array(keep_idx)
    us_data  = mrsi_data[:, :, keep_arr, :]   # (1, K, n_keep, C)
    us_ksp   = mrsi_ksp[:, :, keep_arr]       # (3, K, n_keep)

    print(f"\n[undersample] Before → mrsi_data: {mrsi_data.shape}  mrsi_ksp: {mrsi_ksp.shape}")
    print(f"[undersample] After  → mrsi_data: {us_data.shape}   mrsi_ksp: {us_ksp.shape}")

    np.save(str(out_dir / "mrsi_data.npy"),  us_data)
    np.save(str(out_dir / "mrsi_ksp.npy"),   us_ksp)
    np.save(str(out_dir / "ring_mask.npy"),  keep_arr)
    print(f"[undersample] Saved mrsi_data.npy  dtype={us_data.dtype}")
    print(f"[undersample] Saved mrsi_ksp.npy   dtype={us_ksp.dtype}")
    print(f"[undersample] Saved ring_mask.npy  kept indices: {keep_arr[:5]} … (n={len(keep_arr)})")

    # ── Symlink shared files ──────────────────────────────────────────────────
    print(f"\n[undersample] Symlinking shared files from {src_dir.name} …")
    for fname in SYMLINK_FILES:
        src_file = src_dir / fname
        link     = out_dir / fname
        if link.exists() or link.is_symlink():
            link.unlink()
        if src_file.exists() or src_file.is_symlink():
            target = os.path.relpath(str(src_file), str(out_dir))
            link.symlink_to(target)
            print(f"  {fname} -> {target}")
        else:
            print(f"  [skip] {fname} not found in {src_dir.name}")

    print(f"\n[undersample] Done.")
    print(f"[undersample] Output: {out_dir}")
    print(f"[undersample] Run downstream pipeline with --data-dir {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
data_proc_05 — Average k-space MRSI data across selected 260623 subjects.

Loads mrsi_data.npy from each selected subject, averages them along the
subject axis, and writes a new processed-data folder with the same layout
as a single subject so all downstream scripts (lipid removal, SPICE, etc.)
work unchanged.

Averaged noise:  sigma_comb = sigma_noise / sqrt(N_selected)

New folder name: <prefix>_comb_<id1>_<id2>_...
  e.g. invivo_260623_comb_01_02_03

New folder contents (data/processed/<name>/):
  mrsi_data.npy      ← averaged (new, complex64)
  sigma_noise.npy    ← sigma_noise / sqrt(N)  (new)
  mrsi_ksp.npy       ← symlink to ref subject
  scan_params.json   ← symlink to ref subject
  wref_data.npy      ← symlink to ref subject
  wref_ksp.npy       ← symlink to ref subject
  wref_o.npy         ← symlink to ref subject
  wref_o_check.png   ← symlink to ref subject
  affine.npy         ← symlink to ref subject

Also symlinks output/<name>/coilmap and output/<name>/b0map as directory
symlinks to the ref subject's output folders (both are derived from the
shared wref scan and are identical across subjects).

Usage:
    # combine subjects 01 02 03
    python scripts/data_preproc/data_proc_05_combine_mrsi.py \
        --subject-ids 01 02 03 \
        --data-root   data/processed \
        --out-root    output \
        --prefix      invivo_260623

    # combine all five
    python scripts/data_preproc/data_proc_05_combine_mrsi.py \
        --subject-ids 01 02 03 04 05 \
        --data-root   data/processed \
        --out-root    output \
        --prefix      invivo_260623
"""

import argparse
import os
import shutil
import sys
import numpy as np
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

SYMLINK_FILES = [
    "mrsi_ksp.npy",
    "scan_params.json",
    "wref_data.npy",
    "wref_ksp.npy",
    "wref_o.npy",
    "wref_o_check.png",
    "affine.npy",
]


def parse_args():
    p = argparse.ArgumentParser(description="Average MRSI k-space data across subjects")
    p.add_argument("--subject-ids", nargs="+", required=True,
                   help="Subject IDs to combine, e.g. 01 02 03")
    p.add_argument("--data-root",   default="data/processed",
                   help="Root for processed data folders (default: data/processed)")
    p.add_argument("--out-root",    default="output",
                   help="Root for pipeline output folders (default: output)")
    p.add_argument("--prefix",      default="invivo_260623",
                   help="Common folder prefix (default: invivo_260623)")
    p.add_argument("--ref-id",      default=None,
                   help="Subject ID to use as reference for symlinks (default: first in --subject-ids)")
    p.add_argument("--out-dir",     default=None,
                   help="Override processed-data output path (default: <data-root>/<prefix>_comb_<ids>)")
    return p.parse_args()


def main():
    args  = parse_args()
    ids   = args.subject_ids
    ref_id = args.ref_id or ids[0]
    N     = len(ids)

    ref_dir = Path(args.data_root) / f"{args.prefix}_{ref_id}"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference subject directory not found: {ref_dir}")

    # Output folder
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = "_".join(ids)
        out_dir = Path(args.data_root) / f"{args.prefix}_comb_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[combine] Subjects: {ids}  (N={N})")
    print(f"[combine] Reference: {ref_dir}")
    print(f"[combine] Output:    {out_dir}")

    # ── Average mrsi_data ─────────────────────────────────────────────────────
    print("\n[combine] Loading and averaging mrsi_data.npy …")
    accum   = None
    shapes  = []
    for sid in ids:
        src = Path(args.data_root) / f"{args.prefix}_{sid}" / "mrsi_data.npy"
        if not src.exists():
            raise FileNotFoundError(f"Missing: {src}")
        d = np.load(str(src))
        shapes.append(d.shape)
        if accum is None:
            accum = d.astype(np.complex128)
        else:
            if d.shape != accum.shape:
                raise ValueError(f"Shape mismatch: {d.shape} vs {accum.shape}")
            accum += d.astype(np.complex128)
        print(f"  loaded {src.name} from {args.prefix}_{sid}  shape={d.shape}")

    averaged = (accum / N).astype(np.complex64)
    out_data = out_dir / "mrsi_data.npy"
    np.save(str(out_data), averaged)
    print(f"[combine] Saved mrsi_data.npy  shape={averaged.shape}  dtype={averaged.dtype}")

    # ── sigma_noise ───────────────────────────────────────────────────────────
    sigma_ref = float(np.load(str(ref_dir / "sigma_noise.npy")))
    sigma_comb = np.float32(sigma_ref / np.sqrt(N))
    np.save(str(out_dir / "sigma_noise.npy"), sigma_comb)
    print(f"[combine] sigma_noise: {sigma_ref:.4e} / sqrt({N}) = {sigma_comb:.4e}")

    # ── Symlink shared files from reference subject ───────────────────────────
    print(f"\n[combine] Symlinking shared files from {ref_dir.name} …")
    for fname in SYMLINK_FILES:
        src  = ref_dir / fname
        link = out_dir / fname
        if link.exists() or link.is_symlink():
            link.unlink()
        if src.exists():
            target = os.path.relpath(str(src), str(out_dir))
            link.symlink_to(target)
            print(f"  {fname} -> {target}")
        else:
            print(f"  [skip] {fname} not found in {ref_dir.name}")

    # ── Symlink output subdirs (coilmap, b0map) from ref subject ─────────────
    tag        = out_dir.name          # e.g. invivo_260623_comb_01_02_03
    ref_out    = Path(args.out_root) / f"{args.prefix}_{ref_id}"
    comb_out   = Path(args.out_root) / tag
    comb_out.mkdir(parents=True, exist_ok=True)

    for subdir in ("coilmap", "b0map"):
        src  = ref_out / subdir
        link = comb_out / subdir
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        if src.exists():
            target = os.path.relpath(str(src), str(comb_out))
            link.symlink_to(target)
            print(f"[combine] output/{tag}/{subdir} -> {target}")
        else:
            print(f"[combine] [skip] {subdir} not found in {ref_out.name} — run earlier steps first")

    print(f"\n[combine] Done.  Data:   {out_dir}")
    print(f"[combine]         Output: {comb_out}")
    print(f"[combine] Run downstream pipeline as usual with --data-dir {out_dir}")


if __name__ == "__main__":
    main()

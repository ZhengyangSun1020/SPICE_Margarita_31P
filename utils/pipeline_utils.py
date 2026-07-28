"""Shared pipeline utilities."""

import glob
import os
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, label


def make_brain_mask(wref_img: np.ndarray, threshold: float, erosion: int = 3,
                     cleanup: bool = False):
    """Brain mask from normalised absolute wref_o.

    cleanup : if True, run an extra cleanup pass on the thresholded mask —
        keep only the largest connected component (discards disconnected
        noise blobs outside the brain) and fill any enclosed holes (voxels
        surrounded by brain that fell under threshold, e.g. a signal void in
        the centre). Default off, so existing datasets are unaffected.

    Returns (wref_norm, brain_mask, brain_mask_inner).
    """
    wref_2d = np.abs(wref_img.squeeze(-1))
    wref_norm = (wref_2d - wref_2d.min()) / (wref_2d.max() - wref_2d.min() + 1e-12)
    brain_mask = wref_norm > threshold
    if cleanup:
        labeled, n_components = label(brain_mask)
        if n_components > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # ignore background label
            brain_mask = labeled == sizes.argmax()
        brain_mask = binary_fill_holes(brain_mask)
    brain_mask_inner = binary_erosion(brain_mask, iterations=erosion)
    return wref_norm, brain_mask, brain_mask_inner


def try_symlink_shared_output(data_dir, subject_out_dir, subdir):
    """If wref_o.npy in data_dir is a symlink, the wref is shared from a primary
    subject whose coilmap/b0map/... results are identical for all subjects.
    In that case, create subject_out_dir/<subdir> as a relative symlink pointing
    to the primary subject's equivalent directory instead of recomputing.

    Returns True (caller should return early) if symlinked successfully.
    Returns False if wref is not a symlink, or primary output not found yet.
    """
    wref_o_path = os.path.join(data_dir, "wref_o.npy")
    if not os.path.islink(wref_o_path):
        return False

    primary_data_dir = os.path.dirname(os.path.realpath(wref_o_path))
    primary_subj_id  = os.path.basename(primary_data_dir)

    out_root       = os.path.dirname(os.path.abspath(subject_out_dir))
    primary_subdir = os.path.join(out_root, primary_subj_id, subdir)

    if not os.path.isdir(primary_subdir):
        print(f"[pipeline] wref files are symlinked from primary subject {primary_subj_id!r}.")
        print(f"[pipeline] Primary {subdir}/ not found at: {primary_subdir}")
        print(f"[pipeline] Run this script on the primary subject first, then re-run here.")
        return False

    os.makedirs(subject_out_dir, exist_ok=True)
    link_path = os.path.join(subject_out_dir, subdir)

    if os.path.lexists(link_path):
        if os.path.islink(link_path):
            os.remove(link_path)
        else:
            print(f"[pipeline] {link_path} exists and is not a symlink — skipping auto-symlink.")
            return False

    rel = os.path.relpath(primary_subdir, subject_out_dir)
    os.symlink(rel, link_path)
    print(f"[pipeline] wref files are symlinked from primary subject {primary_subj_id!r}.")
    print(f"[pipeline] Symlinked {subdir}/  →  {rel}  (skipping recompute).")
    return True


def patch_spicefit_tree(out_dir: str, links: list, subdir: str = "data") -> None:
    """Symlink NIfTI outputs into any existing spice_fit/<subdir>/ directories
    and patch their mrsi.tree.  No-op when no fitting directory exists yet or
    when a source file is missing.

    out_dir : base scan output directory (e.g. output/invivo_260623_01/)
    links   : [(dst_name, src_path, tree_label), ...]
    subdir  : 'data' (default) or 'fit'
    """
    spice_fit_dirs = glob.glob(os.path.join(out_dir, "fitting_*/spice_fit"))
    if not spice_fit_dirs:
        return

    for fsl_out in spice_fit_dirs:
        link_dir = os.path.join(fsl_out, subdir)
        os.makedirs(link_dir, exist_ok=True)

        created = []
        for dst_name, src, label in links:
            if not os.path.exists(src):
                continue
            dst = os.path.join(link_dir, dst_name)
            if os.path.islink(dst) or os.path.exists(dst):
                os.unlink(dst)
            os.symlink(os.path.abspath(src), dst)
            created.append((dst_name, label))

        if not created:
            continue

        tree_path = os.path.join(fsl_out, "mrsi.tree")
        if not os.path.exists(tree_path):
            print(f"[filetree] Symlinked {len(created)} file(s) → {fsl_out}/{subdir}/")
            continue

        with open(tree_path) as f:
            tree_txt = f.read()

        new_entries = "".join(
            f"    {name:<38} ({label})\n"
            for name, label in created
            if name not in tree_txt
        )

        if new_entries:
            if subdir == "fit":
                anchor = "data\n" if "data\n" in tree_txt else "uncertainties\n"
                tree_txt = tree_txt.replace(anchor, new_entries + anchor)
            elif f"{subdir}\n" in tree_txt:
                tree_txt = tree_txt.replace(f"{subdir}\n", f"{subdir}\n" + new_entries)
            else:
                tree_txt = tree_txt.replace("uncertainties\n",
                                            f"{subdir}\n" + new_entries + "uncertainties\n")
            with open(tree_path, "w") as f:
                f.write(tree_txt)

        print(f"[filetree] Symlinked {len(created)} file(s) → {fsl_out}/{subdir}/ "
              f"({'patched mrsi.tree' if new_entries else 'tree already up to date'})")

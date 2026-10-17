#!/usr/bin/env python3
"""Draw a 2-D ROI on one MRSI slice and broadcast it to every slice.

The mask is drawn on the *native-frame* spatial prior
(``data/localizer_spatial_prior_native_3d.npy``) -- the exact array the
reconstruction multiplies the ROI into -- so no orientation transform is
needed and the saved mask lines up with 05d's ``--roi-mask`` input.

Output: a bool array of shape (Nx, Ny, Nz) where the 2-D drawing is repeated
across all Z slices.

Usage
-----
    python draw_roi_mask.py OUTPUT_DIR [--tag liver] [--z mean|max|<int>]

Then feed it back into the reconstruction, e.g.:
    python 05d_spice_recon_31p_3d.py ... \
        --roi-mask OUTPUT_DIR/roi_mask_liver.npy --roi-strength strong

Controls (interactive window)
-----------------------------
    left-drag / left-click  : paint ROI (turn voxels ON)
    right-drag / right-click: erase (turn voxels OFF)
    r : reset (clear)        f : fill all
    s or Enter : save + close        q or Esc : cancel without saving
"""
import argparse
import os
import sys
import numpy as np


def _load_native_prior(out_dir):
    p = os.path.join(out_dir, "data", "localizer_spatial_prior_native_3d.npy")
    if not os.path.isfile(p):
        sys.exit(f"[draw_roi] native prior not found: {p}\n"
                 f"           run 05d once (with a localizer) so it gets written.")
    prior = np.load(p).astype(np.float32)
    if prior.ndim != 3:
        sys.exit(f"[draw_roi] expected 3-D prior, got shape {prior.shape}")
    return prior


def _background_2d(prior, z_mode):
    """Native-frame 2-D image to draw on. Rows = x (axis0), cols = y (axis1)."""
    if z_mode == "mean":
        return prior.mean(axis=2)
    if z_mode == "max":
        return prior.max(axis=2)
    z = int(z_mode)
    if not (0 <= z < prior.shape[2]):
        sys.exit(f"[draw_roi] --z {z} out of range 0..{prior.shape[2]-1}")
    return prior[:, :, z]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="05d output directory (contains data/ and figures/)")
    ap.add_argument("--tag", default="liver", help="mask name -> roi_mask_<tag>.npy")
    ap.add_argument("--z", default="mean",
                    help="slice to display: 'mean', 'max', or a z index (default mean)")
    ap.add_argument("--out", default=None,
                    help="explicit output .npy path (default OUT_DIR/roi_mask_<tag>.npy)")
    args = ap.parse_args()

    prior = _load_native_prior(args.out_dir)
    nx, ny, nz = prior.shape
    bg = _background_2d(prior, args.z)
    # normalise background just for display contrast
    bg_disp = bg - bg.min()
    bg_disp = bg_disp / (bg_disp.max() + 1e-12)

    out_path = args.out or os.path.join(args.out_dir, f"roi_mask_{args.tag}.npy")

    mask2d = np.zeros((nx, ny), dtype=bool)

    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(bg_disp, origin="upper", cmap="gray", interpolation="nearest")
    # red translucent overlay for the current mask
    overlay_cmap = ListedColormap([(1, 0, 0, 0.0), (1, 0, 0, 0.45)])
    ov = ax.imshow(mask2d.astype(int), origin="upper", cmap=overlay_cmap,
                   vmin=0, vmax=1, interpolation="nearest")
    # grid lines on cell boundaries
    ax.set_xticks(np.arange(-0.5, ny, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nx, 1), minor=True)
    ax.grid(which="minor", color=(0.3, 0.6, 1.0, 0.5), linewidth=0.7)
    ax.set_xlabel("native y index (axis 1)")
    ax.set_ylabel("native x index (axis 0)")

    def _refresh():
        ov.set_data(mask2d.astype(int))
        ax.set_title(f"ROI on native prior (z={args.z})  |  {int(mask2d.sum())} voxels"
                     f"  |  L=paint  R=erase  r=reset  f=fill  s=save  q=cancel",
                     fontsize=9)
        fig.canvas.draw_idle()

    _refresh()

    painting = {"active": False, "value": True}

    def _paint_at(event):
        if event.xdata is None or event.ydata is None:
            return
        iy = int(round(event.xdata))  # column -> native y
        ix = int(round(event.ydata))  # row    -> native x
        if 0 <= ix < nx and 0 <= iy < ny:
            if mask2d[ix, iy] != painting["value"]:
                mask2d[ix, iy] = painting["value"]
                _refresh()

    def on_press(event):
        if event.inaxes is not ax:
            return
        if event.button == 1:
            painting["active"], painting["value"] = True, True
        elif event.button == 3:
            painting["active"], painting["value"] = True, False
        else:
            return
        _paint_at(event)

    def on_move(event):
        if painting["active"] and event.inaxes is ax:
            _paint_at(event)

    def on_release(event):
        painting["active"] = False

    saved = {"ok": False}

    def on_key(event):
        if event.key in ("r",):
            mask2d[:] = False
            _refresh()
        elif event.key in ("f",):
            mask2d[:] = True
            _refresh()
        elif event.key in ("s", "enter"):
            saved["ok"] = True
            plt.close(fig)
        elif event.key in ("q", "escape"):
            saved["ok"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("[draw_roi] draw the ROI, then press 's' (or Enter) to save, 'q' to cancel.")
    plt.show()

    if not saved["ok"]:
        print("[draw_roi] cancelled, nothing saved.")
        return
    if mask2d.sum() == 0:
        print("[draw_roi] mask is empty, nothing saved.")
        return

    mask3d = np.repeat(mask2d[:, :, None], nz, axis=2)
    np.save(out_path, mask3d)
    print(f"[draw_roi] saved {out_path}")
    print(f"[draw_roi] shape={mask3d.shape} dtype={mask3d.dtype} "
          f"kept {int(mask3d.sum())}/{mask3d.size} voxels "
          f"({int(mask2d.sum())}/{nx*ny} per slice)")

    # save a quick PNG preview next to the mask
    try:
        import matplotlib.pyplot as plt2
        fig2, ax2 = plt2.subplots(figsize=(6, 6))
        ax2.imshow(bg_disp, origin="upper", cmap="gray", interpolation="nearest")
        ax2.imshow(np.ma.masked_where(~mask2d, mask2d), origin="upper",
                   cmap=ListedColormap([(1, 0, 0, 0.45)]), interpolation="nearest")
        ax2.set_title(f"roi_mask_{args.tag} (broadcast to {nz} slices)")
        png = os.path.splitext(out_path)[0] + "_preview.png"
        fig2.savefig(png, dpi=150, bbox_inches="tight")
        print(f"[draw_roi] preview -> {png}")
    except Exception as e:
        print(f"[draw_roi] preview skipped: {e}")


if __name__ == "__main__":
    main()

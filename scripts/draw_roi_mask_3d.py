#!/usr/bin/env python3
"""
Interactive ROI drawer for 3-D CRT MRSI (12x12x11-style native grids).

Loads the localizer prior with exactly the same geometry that
`scripts/05d_spice_recon_31p_3d.py` uses, shows it slice-by-slice in the
DICOM row/col frame (same axes as `spatial_prior_map.png`), and lets you
left/right-click voxels to toggle ROI membership.

Save produces a bool `.npy` of shape (Nx, Ny, Nz) in the NATIVE MRSI axes
that `--roi-mask` in 05d expects.  Combine with `--roi-edge-scale` /
`--roi-hard-boundary` / `--roi-renorm` to make outside-mask weights very
small (like the 05c workflow).

Example
-------
    python scripts/draw_roi_mask_3d.py \
        --dat "/mnt/f/.../meas_MID00073_FID95804_..._30r.dat" \
        --localizer "/mnt/f/.../localizer_TRA_29_None" \
        --out roi_mask_MID00073.npy

    # then in 05d:
    python scripts/05d_spice_recon_31p_3d.py ... \
        --roi-mask roi_mask_MID00073.npy \
        --roi-edge-scale 50 --roi-dilation 1 --roi-renorm
"""

import argparse
import importlib.util as _ilu
import os
import pathlib as _pl
import sys

import numpy as np

_HERE = _pl.Path(__file__).parent
sys.path.insert(0, str(_HERE))

# 05d hard-codes matplotlib.use("Agg") for headless figure output, which would
# make plt.show() a no-op here. Import it first, then switch to a GUI backend.
_spec_05d = _ilu.spec_from_file_location("_05d", _HERE / "05d_spice_recon_31p_3d.py")
_05d = _ilu.module_from_spec(_spec_05d)
_spec_05d.loader.exec_module(_05d)

import matplotlib

_PREFERRED_BACKENDS = ["QtAgg", "Qt5Agg", "TkAgg", "GTK3Agg", "MacOSX"]
_chosen_backend = None
for _bk in _PREFERRED_BACKENDS:
    try:
        matplotlib.use(_bk, force=True)
        _chosen_backend = _bk
        break
    except Exception:
        continue
if _chosen_backend is None:
    print("[roi] WARNING: no interactive matplotlib backend available.")
    print("[roi] On WSL, install one of:  Tk (sudo apt install python3-tk)")
    print("[roi] or PyQt5 (pip install pyqt5), and ensure $DISPLAY is set.")
    print(f"[roi] current DISPLAY = {os.environ.get('DISPLAY', '<unset>')}")
    sys.exit(1)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print(f"[roi] matplotlib backend: {matplotlib.get_backend()}   "
      f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}")

load_localizer_prior_3d = _05d.load_localizer_prior_3d
load_interleaved_31p = _05d.load_interleaved_31p
_fGSLCalcPRS = _05d._fGSLCalcPRS


def parse_args():
    p = argparse.ArgumentParser(description="Interactive 3-D ROI drawer for MRSI")
    p.add_argument("--dat", required=True, help="Siemens Twix .dat (same as 05d)")
    p.add_argument("--localizer", required=True, help="Localizer DICOM folder")
    p.add_argument("--out", required=True, help="Output ROI mask .npy path")
    p.add_argument("--init", default=None,
                   help="Optional existing mask .npy to start from")
    p.add_argument("--localizer-percentiles", type=float, nargs=2,
                   default=[2.0, 99.5])
    p.add_argument("--localizer-contrast-gamma", type=float, default=0.7)
    p.add_argument("--prior-flip-native-y",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Match the 05d default (False).")
    p.add_argument("--nx", type=int, default=None,
                   help="Override MRSI Nx (default: header base resolution)")
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz", type=int, default=None,
                   help="Override MRSI Nz (default: header lPartitions)")
    p.add_argument("--single-slice", type=int, nargs="?", const=-1, default=None,
                   help="Draw ROI on ONE slice only; the resulting 2-D mask is "
                        "tiled across all z on save. Optional int selects which "
                        "slice to display (default: middle).")
    return p.parse_args()


def build_view_transform(info, ref_record, flip_native_y):
    """Return (view_fn, inv_fn, axes_label).

    view_fn(arr_xy)  : (nx, ny) -> (H_view, W_view) DICOM-oriented display
    inv_fn(row_view, col_view) -> (x_native, y_native)
    axes_label : (ylabel, xlabel) for the display
    """
    normal = np.array([info["snv_x"], info["snv_y"], info["snv_z"]], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-20
    phase_vec, read_vec = _fGSLCalcPRS(normal, info["in_plane_rotation"])
    phase_vec = phase_vec / (np.linalg.norm(phase_vec) + 1e-20)
    read_vec = read_vec / (np.linalg.norm(read_vec) + 1e-20)
    y_axis_vec = (-read_vec) if flip_native_y else read_vec

    row_dir = ref_record["row_dir"]
    col_dir = ref_record["col_dir"]
    a_row = float(np.dot(phase_vec, row_dir))
    a_col = float(np.dot(phase_vec, col_dir))
    b_row = float(np.dot(y_axis_vec, row_dir))
    b_col = float(np.dot(y_axis_vec, col_dir))

    # phase (x) on rows if it aligns more with row_dir than col_dir.
    x_on_rows = abs(a_row) >= abs(a_col)
    flip_row = (a_row < 0) if x_on_rows else (b_row < 0)
    flip_col = (b_col < 0) if x_on_rows else (a_col < 0)

    def view_fn(arr_xy):
        out = arr_xy if x_on_rows else arr_xy.T
        if flip_row:
            out = out[::-1, :]
        if flip_col:
            out = out[:, ::-1]
        return out

    def inv_fn(row_view, col_view, H, W):
        r = (H - 1 - row_view) if flip_row else row_view
        c = (W - 1 - col_view) if flip_col else col_view
        if x_on_rows:
            return int(r), int(c)      # row=x, col=y
        else:
            return int(c), int(r)      # row=y, col=x  ->  x=c, y=r

    if x_on_rows:
        axes_label = ("MRSI x (phase)", "MRSI y (read)")
    else:
        axes_label = ("MRSI y (read)", "MRSI x (phase)")
    return view_fn, inv_fn, axes_label


class RoiDrawer:
    def __init__(self, prior, view_fn, inv_fn, axes_label, out_path,
                 init_mask=None, single_slice=None):
        self.prior = prior                                # (nx, ny, nz) native
        self.nx, self.ny, self.nz = prior.shape
        self.view_fn = view_fn
        self.inv_fn = inv_fn
        self.axes_label = axes_label
        self.out_path = out_path
        # single_slice: None = full 3-D; int = lock to that z, tile on save.
        self.single_slice = single_slice
        self.mask = np.zeros_like(prior, dtype=bool)
        if init_mask is not None:
            if init_mask.shape != prior.shape:
                sys.exit(f"[roi] init mask shape {init_mask.shape} != {prior.shape}")
            self.mask = init_mask.astype(bool).copy()

        self.z = self.single_slice if self.single_slice is not None else self.nz // 2
        self.z = max(0, min(self.nz - 1, int(self.z)))
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._prior_view0 = view_fn(prior[:, :, 0])
        self.H, self.W = self._prior_view0.shape
        self.cell_patches = np.empty((self.H, self.W), dtype=object)
        self._init_display()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _init_display(self):
        self.ax.clear()
        view = self.view_fn(self.prior[:, :, self.z])
        self.im = self.ax.imshow(view, origin="upper", cmap="Greens",
                                 vmin=0, vmax=max(1e-12, float(self.prior.max())))
        self.ax.set_xticks(np.arange(-0.5, self.W, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.H, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linewidth=0.4, alpha=0.5)
        self.ax.set_xlabel(self.axes_label[1])
        self.ax.set_ylabel(self.axes_label[0])
        self._refresh_overlay()
        self._update_title()

    def _refresh_overlay(self):
        # Clear old rectangles
        for r in range(self.H):
            for c in range(self.W):
                if self.cell_patches[r, c] is not None:
                    try:
                        self.cell_patches[r, c].remove()
                    except Exception:
                        pass
                    self.cell_patches[r, c] = None
        mask_view = self.view_fn(self.mask[:, :, self.z].astype(np.uint8)).astype(bool)
        for r in range(self.H):
            for c in range(self.W):
                if mask_view[r, c]:
                    patch = Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                                      linewidth=1.2, edgecolor="red",
                                      facecolor=(1, 0, 0, 0.25))
                    self.ax.add_patch(patch)
                    self.cell_patches[r, c] = patch
        self.fig.canvas.draw_idle()

    def _update_title(self):
        n_in = int(self.mask[:, :, self.z].sum())
        n_tot = int(self.mask.sum())
        mode = (f"single-slice (z={self.z + 1} tiled to all {self.nz} on save)"
                if self.single_slice is not None else "3-D")
        self.ax.set_title(
            f"[{mode}]  z={self.z + 1}/{self.nz}  |  in-slice ROI={n_in}  |  total ROI={n_tot}\n"
            "click: toggle voxel  |  n/p: next/prev z  |  c: clear slice  |  "
            "C: clear all  |  a: fill slice  |  A: fill all  |  "
            "R: replicate slice to all z  |  s: save  |  q: quit",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        col_view = int(round(event.xdata))
        row_view = int(round(event.ydata))
        if not (0 <= row_view < self.H and 0 <= col_view < self.W):
            return
        x, y = self.inv_fn(row_view, col_view, self.H, self.W)
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            return
        if event.button == 1:   # left = toggle
            self.mask[x, y, self.z] = not self.mask[x, y, self.z]
        elif event.button == 3:  # right = clear
            self.mask[x, y, self.z] = False
        self._refresh_overlay()
        self._update_title()

    def _save(self):
        out_mask = self.mask.copy()
        if self.single_slice is not None:
            # Tile the drawn slice across every z.
            out_mask[:, :, :] = self.mask[:, :, self.z][:, :, None]
        np.save(self.out_path, out_mask.astype(bool))
        n_in = int(out_mask.sum())
        print(f"[roi] saved mask {out_mask.shape}  in-ROI={n_in}  "
              f"-> {self.out_path}"
              + ("  (single-slice tiled across all z)"
                 if self.single_slice is not None else ""))

    def on_key(self, event):
        k = event.key
        if k in ("n", "right", "pagedown"):
            if self.single_slice is not None:
                return
            self.z = (self.z + 1) % self.nz
            self._init_display()
        elif k in ("p", "left", "pageup"):
            if self.single_slice is not None:
                return
            self.z = (self.z - 1) % self.nz
            self._init_display()
        elif k == "c":
            self.mask[:, :, self.z] = False
            self._refresh_overlay(); self._update_title()
        elif k == "C":
            self.mask[:] = False
            self._refresh_overlay(); self._update_title()
        elif k == "a":
            self.mask[:, :, self.z] = True
            self._refresh_overlay(); self._update_title()
        elif k == "A":
            self.mask[:] = True
            self._refresh_overlay(); self._update_title()
        elif k in ("R", "r"):
            self.mask[:, :, :] = self.mask[:, :, self.z][:, :, None]
            print(f"[roi] replicated z={self.z + 1} mask to all {self.nz} slices")
            self._refresh_overlay(); self._update_title()
        elif k == "s":
            self._save()
        elif k == "q":
            plt.close(self.fig)


def main():
    args = parse_args()

    info = load_interleaved_31p(args.dat)
    nx = args.nx or info["matrix_size"]
    ny = args.ny or info["matrix_size"]
    nz = args.nz or info["Nz"]
    print(f"[roi] MRSI native grid: ({nx}, {ny}, {nz})")

    prior, selected = load_localizer_prior_3d(
        args.localizer, info, (nx, ny, nz),
        percentiles=args.localizer_percentiles,
        gamma=args.localizer_contrast_gamma,
        flip_native_y=args.prior_flip_native_y,
    )

    view_fn, inv_fn, axes_label = build_view_transform(
        info, selected[0]["record"], args.prior_flip_native_y)

    init_mask = None
    if args.init:
        init_mask = np.load(args.init)
        print(f"[roi] loaded initial mask from {args.init}  in-ROI={int(init_mask.sum())}")

    single_slice = None
    if args.single_slice is not None:
        single_slice = (nz // 2) if args.single_slice == -1 else int(args.single_slice)
        single_slice = max(0, min(nz - 1, single_slice))
        print(f"[roi] single-slice mode: drawing z={single_slice + 1}/{nz}, "
              f"will be tiled across all {nz} slices on save")

    drawer = RoiDrawer(prior, view_fn, inv_fn, axes_label,
                       out_path=args.out, init_mask=init_mask,
                       single_slice=single_slice)
    print("[roi] Interactive window open. Press 's' to save, 'q' to quit.")
    plt.show()

    # Autosave on exit (uses the single-slice tiling if enabled).
    drawer._save()


if __name__ == "__main__":
    main()

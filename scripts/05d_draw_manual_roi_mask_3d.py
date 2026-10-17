#!/usr/bin/env python3
"""
Interactive 3D manual ROI mask drawer for 05d SPICE (interleaved 31P CRT).

Draw the ROI slice-by-slice on the *real, high-resolution* localizer anatomy
(sampled onto the exact MRSI FOV/orientation) and save a boolean mask of shape
(Nx, Ny, Nz) in the convention 05d expects (--roi-mask). Internally you draw on
a fine super-sampled image; on save the fine ROI is binned down to the MRSI
voxel grid (a voxel is kept if its in-ROI fraction >= --voxel-frac).

The anatomy is displayed upright (voxel (xi, yi) at display cell (xi, yi)). The
saved voxel mask is orientation-independent (indexed in the recon voxel frame,
same as localizer_spatial_prior_native_3d.npy), so it feeds --roi-mask directly
regardless of how it is displayed here.

RUN THIS ON A MACHINE WITH A DISPLAY (needs interactive matplotlib; e.g. WSLg,
an X server, or native Linux/Mac). It cannot run headless.

Background source:
  --dat + --localizer   (REQUIRED for high-res real-anatomy drawing) resample
                        the DICOM localizer onto a fine MRSI-FOV grid here.
  --prior  path to a saved localizer_spatial_prior_native_3d.npy  -> only the
           coarse 10x10 grid is available (no --dat), draw per-voxel (S=1).

Usage (high-res, recommended):
  python scripts/05d_draw_manual_roi_mask_3d.py \
      --dat "/mnt/f/.../meas_MID00047_...dat" --rep 2 --cache-dir _tmp_recon/cache \
      --localizer "/mnt/f/.../0626_invivo/locolizer" \
      --supersample 16 --voxel-frac 0.5 \
      --out-mask roi_mask_cardiac_manual.npy

Controls:
  BRUSH mode (default): left click/drag = add, right click/drag = erase
  [ / ]  : shrink / grow brush          l : lasso-add    k : lasso-erase
  n / ->  : next slice                   p / <-  : previous slice
  c : clear this slice    a : clear all    f : fill this slice
  s : save mask (.npy + preview .png)      q : quit (auto-saves)

Overlays: cyan = MRSI voxel grid, red = your fine ROI, lime outline = voxels
that will be KEPT at the current --voxel-frac threshold.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import_05d():
    import importlib.util as ilu, pathlib as pl, sys, types
    # The drawer only samples the localizer; it never runs a NUFFT. Importing 05d
    # (and 07b) pulls in `import mrinufft` at module top, so stub it when it is not
    # importable — this keeps the drawer usable in any env that merely has
    # numpy / scipy / matplotlib / pydicom.
    if "mrinufft" not in sys.modules:
        try:
            import mrinufft  # noqa: F401
        except Exception:
            stub = types.ModuleType("mrinufft")
            def _no_nufft(*a, **k):
                raise RuntimeError("mrinufft is stubbed in the ROI drawer (not needed here)")
            stub.get_operator = _no_nufft
            sys.modules["mrinufft"] = stub
            print("[mask] note: mrinufft not found — stubbed (drawer does not need it).")
    spec = ilu.spec_from_file_location(
        "_05d", pl.Path(__file__).parent / "05d_spice_recon_31p_3d.py")
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _load_info(m, args):
    ld = dict(rep_override=args.rep)
    if args.cache_dir and m._cached_load is not None:
        kx = dict(ld) if args.rep is not None else {}
        return m._cached_load(m.load_interleaved_31p, args.dat, cache_dir=args.cache_dir,
                              key_extras=kx, refresh=False, tag="31p3d", **ld)
    return m.load_interleaved_31p(args.dat, rep_override=args.rep)


def _sample_highres(m, args, info, shape_xyz, S):
    """Fine super-sampled localizer over the exact MRSI FOV.

    Reuses 05d.load_localizer_prior_3d for identical per-slice plane selection,
    then re-samples the chosen DICOM plane on a (Nx*S, Ny*S) sub-voxel grid using
    the same voxel->world geometry. Returns fine (Nx*S, Ny*S, Nz) in [0,1].
    """
    nx, ny, nz = shape_xyz
    _, selected = m.load_localizer_prior_3d(
        args.localizer, info, shape_xyz,
        percentiles=m.LOCALIZER_PERCENTILES, gamma=m.LOCALIZER_GAMMA,
        flip_native_y=m.PRIOR_FLIP_NATIVE_Y)

    normal = np.array([info["snv_x"], info["snv_y"], info["snv_z"]], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-20
    phase, read = m._fGSLCalcPRS(normal, info["in_plane_rotation"])
    phase /= np.linalg.norm(phase) + 1e-20
    read /= np.linalg.norm(read) + 1e-20
    voxel_xy = float(info["fov_mm"]) / nx
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    p_low, p_high = m.LOCALIZER_PERCENTILES
    gamma = m.LOCALIZER_GAMMA

    # continuous MRSI voxel coordinates for the fine grid (S sub-pixels per voxel)
    xf = (np.arange(nx * S) + 0.5) / S - 0.5
    yf = (np.arange(ny * S) + 0.5) / S - 0.5
    XF, YF = np.meshgrid(xf, yf, indexing="ij")           # (nx*S, ny*S)

    fine = np.zeros((nx * S, ny * S, nz), dtype=np.float32)
    for z in range(nz):
        rec = selected[z]["record"]
        center = selected[z]["mrsi_center"]
        img = rec["img"]
        lo, hi = np.percentile(img, [p_low, p_high])
        imn = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0) ** gamma
        base = center - rec["ipp"]
        rows = (np.dot(base, rec["row_dir"])
                + (XF - cx) * voxel_xy * np.dot(phase, rec["row_dir"])
                + (YF - cy) * voxel_xy * np.dot(read, rec["row_dir"])) / rec["sp_row"]
        cols = (np.dot(base, rec["col_dir"])
                + (XF - cx) * voxel_xy * np.dot(phase, rec["col_dir"])
                + (YF - cy) * voxel_xy * np.dot(read, rec["col_dir"])) / rec["sp_col"]
        fine[:, :, z] = map_coordinates(
            imn, [rows.ravel(), cols.ravel()], order=1, mode="constant",
            cval=0.0).reshape(nx * S, ny * S)

    if m.PRIOR_FLIP_NATIVE_Y:                              # match native-prior frame
        fine = np.flip(fine, axis=1).copy()
    return fine


def _load_background(args):
    """Return (fine (H,W,Nz) float, S, (Nx,Ny,Nz))."""
    if args.dat and args.localizer:
        m = _import_05d()
        info = _load_info(m, args)
        nx = ny = int(info["matrix_size"]); nz = int(info["Nz"])
        S = max(1, int(args.supersample))
        fine = _sample_highres(m, args, info, (nx, ny, nz), S)
        print(f"[mask] high-res anatomy {fine.shape} (S={S}) over MRSI grid {nx}x{ny}x{nz}")
        return fine, S, (nx, ny, nz)
    if args.prior:
        prior = np.load(args.prior).astype(np.float32)
        if prior.ndim != 3:
            sys.exit(f"--prior must be a 3D (Nx,Ny,Nz) array, got {prior.shape}")
        print(f"[mask] coarse prior {prior.shape} (S=1, per-voxel drawing). "
              "Pass --dat/--localizer for high-res real anatomy.")
        return prior, 1, prior.shape
    sys.exit("Provide --dat and --localizer (high-res), or --prior (coarse).")


class Mask3DEditor:
    """Draw a ROI on fine anatomy, bin down to a (Nx,Ny,Nz) voxel mask."""

    def __init__(self, fine, S, shape_xyz, out_mask, voxel_frac):
        self.fine = fine
        self.H, self.W, self.Nz = fine.shape
        self.S = S
        self.Nx, self.Ny, _ = shape_xyz
        self.out_mask = out_mask
        self.voxel_frac = voxel_frac
        self.fmask = np.zeros((self.H, self.W, self.Nz), dtype=bool)
        self.z = self.Nz // 2
        self.button = None
        self.lasso = None
        self.lasso_value = True
        self.brush = max(1, S // 3)
        self.vmax = max(1e-6, float(np.percentile(self.fine, 99.5)))

        # fine-pixel centers in display coords (upright: x=col=wf, y=row=hf)
        hf, wf = np.mgrid[0:self.H, 0:self.W]
        self._fcenters = np.column_stack(
            [wf.ravel().astype(float), hf.ravel().astype(float)])

        self.fig, self.ax = plt.subplots(figsize=(8.5, 8.5))
        self._raise_window()
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    def _raise_window(self):
        """Force the Tk window to appear in front (WSLg often buries it)."""
        try:
            mgr = self.fig.canvas.manager
            mgr.set_window_title("05d ROI drawer — draw the ROI, press s to save")
            win = getattr(mgr, "window", None)
            if win is not None and hasattr(win, "wm_geometry"):
                win.wm_geometry("+80+40")
                win.lift()
                win.attributes("-topmost", True)
                win.after(700, lambda: win.attributes("-topmost", False))
                win.focus_force()
        except Exception:
            pass

    # ---- display convention: anatomy upright, display (row=hf, col=wf) --------
    def _disp(self, arr2d):
        return arr2d

    def _hw_from_click(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        wf = int(round(event.xdata)); row = int(round(event.ydata))
        if not (0 <= wf < self.W and 0 <= row < self.H):
            return None
        return row, wf                                   # (hf, wf)

    def _paint(self, event, value):
        hw = self._hw_from_click(event)
        if hw is None:
            return
        hf, wf = hw
        r = self.brush
        h0, h1 = max(0, hf - r), min(self.H, hf + r + 1)
        w0, w1 = max(0, wf - r), min(self.W, wf + r + 1)
        yy, xx = np.mgrid[h0:h1, w0:w1]
        disk = (yy - hf) ** 2 + (xx - wf) ** 2 <= r * r
        self.fmask[h0:h1, w0:w1, self.z][disk] = value
        self._draw()

    def _voxel_mask_slice(self, z):
        blk = self.fmask[:, :, z].reshape(self.Nx, self.S, self.Ny, self.S)
        return blk.mean(axis=(1, 3)) >= self.voxel_frac      # (Nx, Ny) bool

    def _voxel_mask(self):
        vox = np.zeros((self.Nx, self.Ny, self.Nz), dtype=bool)
        for z in range(self.Nz):
            vox[:, :, z] = self._voxel_mask_slice(z)
        return vox

    # ---- events ---------------------------------------------------------------
    def _on_press(self, event):
        if self.lasso is not None or event.inaxes != self.ax:
            return
        self.button = event.button
        self._paint(event, value=(event.button == 1))

    def _on_release(self, event):
        self.button = None

    def _on_motion(self, event):
        if self.lasso is not None or self.button is None or event.inaxes != self.ax:
            return
        self._paint(event, value=(self.button == 1))

    def _set_lasso(self, active, value=True):
        if self.lasso is not None:
            try:
                self.lasso.disconnect_events()
            except Exception:
                pass
            self.lasso = None
        if active:
            self.lasso_value = value
            self.lasso = LassoSelector(self.ax, onselect=self._on_lasso, button=1)
        self._draw()

    def _on_lasso(self, verts):
        if len(verts) >= 3:
            inside = MplPath(verts).contains_points(self._fcenters)
            self.fmask[:, :, self.z].flat[inside] = self.lasso_value
        self._set_lasso(False)                           # one-shot, back to brush
        self._draw()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k in ("n", "right"):
            self._set_lasso(False); self.z = min(self.Nz - 1, self.z + 1); self._draw()
        elif k in ("p", "left"):
            self._set_lasso(False); self.z = max(0, self.z - 1); self._draw()
        elif k == "l":
            self._set_lasso(self.lasso is None, value=True)
        elif k == "k":
            self._set_lasso(self.lasso is None, value=False)
        elif k == "[":
            self.brush = max(1, self.brush - 1); self._draw()
        elif k == "]":
            self.brush = min(max(self.H, self.W), self.brush + 1); self._draw()
        elif k == "c":
            self.fmask[:, :, self.z] = False; self._draw()
        elif k == "a":
            self.fmask[:] = False; self._draw()
        elif k == "f":
            self.fmask[:, :, self.z] = True; self._draw()
        elif k == "s":
            self._save()
        elif k == "q":
            self._save(); plt.close(self.fig)

    # ---- rendering ------------------------------------------------------------
    def _draw(self):
        self.ax.clear()
        self.ax.imshow(self._disp(self.fine[:, :, self.z]), cmap="gray",
                       origin="upper", vmax=self.vmax)
        fm = self._disp(self.fmask[:, :, self.z]).astype(float)
        self.ax.imshow(np.ma.masked_where(fm < 0.5, fm), cmap="autumn",
                       origin="upper", alpha=0.35, vmin=0, vmax=1)
        # MRSI voxel grid every S fine pixels
        for i in range(self.Nx + 1):
            self.ax.axhline(i * self.S - 0.5, color="cyan", lw=0.4, alpha=0.5)
        for j in range(self.Ny + 1):
            self.ax.axvline(j * self.S - 0.5, color="cyan", lw=0.4, alpha=0.5)
        # voxels that will be KEPT -> lime outline (upsampled to fine display)
        vox = self._voxel_mask_slice(self.z)
        if vox.any():
            up = np.repeat(np.repeat(vox.astype(float), self.S, 0), self.S, 1)
            self.ax.contour(self._disp(up), levels=[0.5], colors="lime", linewidths=1.2)
        nkeep = int(vox.sum())
        mode = ("LASSO-add" if (self.lasso is not None and self.lasso_value)
                else "LASSO-erase" if self.lasso is not None
                else f"BRUSH r={self.brush}")
        self.ax.set_title(
            f"slice z={self.z + 1}/{self.Nz}  |  voxels kept this slice: {nkeep}"
            f"  total: {int(self._voxel_mask().sum())}  |  frac>={self.voxel_frac:g}"
            f"  |  mode: {mode}\n"
            "L/R drag paint | [ ] brush | l/k lasso | n/p slice | c clear | "
            "a clear-all | f fill | s save | q quit", fontsize=8)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.canvas.draw_idle()

    def _save(self):
        vox = self._voxel_mask()
        np.save(self.out_mask, vox)
        cols = min(self.Nz, 5); rows = (self.Nz + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), squeeze=False)
        for z in range(self.Nz):
            ax = axes[z // cols][z % cols]
            ax.imshow(self._disp(self.fine[:, :, z]), cmap="gray", origin="upper", vmax=self.vmax)
            v = self._voxel_mask_slice(z)
            up = np.repeat(np.repeat(v.astype(float), self.S, 0), self.S, 1)
            ax.contour(self._disp(up), levels=[0.5], colors="lime", linewidths=1.1)
            ax.set_title(f"z={z+1} ({int(v.sum())})", fontsize=7)
            ax.axis("off")
        for j in range(self.Nz, rows * cols):
            axes[j // cols][j % cols].axis("off")
        png = os.path.splitext(self.out_mask)[0] + "_preview.png"
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print(f"[mask] saved {self.out_mask}  shape={vox.shape}  "
              f"({int(vox.sum())} voxels)  + {png}")

    def run(self):
        # map the window first (non-blocking), force it to the front, THEN drive an
        # explicit blocking event loop. This surfaces reliably under WSLg where a
        # plain plt.show() often buries or never maps the window.
        plt.show(block=False)
        self._raise_window()
        try:
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        except Exception:
            pass
        try:
            while plt.fignum_exists(self.fig.number):
                self.fig.canvas.start_event_loop(0.2)
        except Exception:
            pass
        # only persist if there is something to save (never clobber with an empty
        # mask from a spurious immediate exit)
        if self.fmask.any():
            self._save()
        else:
            print("[mask] window closed with an empty ROI — nothing saved.")


def main():
    p = argparse.ArgumentParser(description="Interactive high-res 3D ROI mask drawer for 05d")
    p.add_argument("--prior", default=None, help="saved localizer_spatial_prior_native_3d.npy (coarse fallback)")
    p.add_argument("--dat", default=None)
    p.add_argument("--localizer", default=None)
    p.add_argument("--rep", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--supersample", type=int, default=16, help="fine sub-pixels per MRSI voxel")
    p.add_argument("--voxel-frac", type=float, default=0.5, help="keep voxel if in-ROI fraction >= this")
    p.add_argument("--out-mask", required=True)
    args = p.parse_args()

    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        pass
    if matplotlib.get_backend().lower() == "agg":
        sys.exit("No interactive matplotlib backend available — run on a machine "
                 "with a display (WSLg/X server) or `pip install PyQt5`/tk.")

    fine, S, shape_xyz = _load_background(args)
    print("[mask] draw the ROI, then press 's' to save.")
    Mask3DEditor(fine, S, shape_xyz, args.out_mask, args.voxel_frac).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Interactive manual ROI mask drawer for 05c SPICE.

This tool registers the localizer to the MRSI voxel grid (same geometry convention
as 05c), lets you toggle voxels, and saves a boolean mask .npy in the exact shape
expected by 05c (--roi-mask): mask[xi, yi] with shape (N, N).

Usage:
  python scripts/05c_draw_manual_roi_mask.py \
    --dat "/mnt/f/.../meas_MID00276_...dat" \
    --localizer "/mnt/f/.../Series0015...IMA" \
    --out-mask /home/zsun/SPICE_Margarita/roi_mask_mid00276_manual.npy

Controls:
  - Left click: toggle voxel
  - Right click: erase voxel
  - Key s: save mask and preview PNG
  - Key c: clear all
  - Key i: invert mask
  - Key q: quit
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

import importlib.util as _ilu
import pathlib as _pl

plt = None


class MaskEditor:
    def __init__(self, anatomy, out_mask, title="Manual ROI Mask"):
        self.anatomy = anatomy.astype(np.float32)
        self.ny, self.nx = self.anatomy.shape
        if self.nx != self.ny:
            raise ValueError("This tool expects square MRSI grids (N x N).")
        self.N = self.nx
        self.out_mask = out_mask

        # 05c ROI convention: roi[xi, yi]
        self.roi_xy = np.zeros((self.N, self.N), dtype=bool)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title(title)
        self.ax.imshow(
            self.anatomy,
            origin="lower",
            cmap="gray",
            extent=[0, self.N, 0, self.N],
            vmin=0,
            vmax=max(1e-6, float(self.anatomy.max())),
        )

        # Display overlay in row/col view: transpose roi[xi,yi] -> [yi,xi]
        self.overlay = self.ax.imshow(
            self.roi_xy.T.astype(float),
            origin="lower",
            cmap="autumn",
            alpha=0.35,
            extent=[0, self.N, 0, self.N],
            vmin=0,
            vmax=1,
        )

        # Grid lines for voxel boundaries
        for k in range(self.N + 1):
            self.ax.plot([k, k], [0, self.N], color="cyan", lw=0.6, alpha=0.5)
            self.ax.plot([0, self.N], [k, k], color="cyan", lw=0.6, alpha=0.5)

        self.ax.set_xlim(0, self.N)
        self.ax.set_ylim(0, self.N)
        self.ax.set_xlabel("x voxel index")
        self.ax.set_ylabel("y voxel index")

        self.status = self.fig.text(
            0.02,
            0.02,
            self._status_text(),
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 6},
            color="white",
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _status_text(self):
        n_on = int(self.roi_xy.sum())
        return (
            f"Selected voxels: {n_on}/{self.N*self.N}   "
            "[L-click toggle, R-click erase, s save, c clear, i invert, q quit]"
        )

    def _refresh(self):
        self.overlay.set_data(self.roi_xy.T.astype(float))
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _xy_from_event(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        xi = int(np.floor(event.xdata))
        yi = int(np.floor(event.ydata))
        if xi < 0 or xi >= self.N or yi < 0 or yi >= self.N:
            return None
        return xi, yi

    def on_click(self, event):
        xy = self._xy_from_event(event)
        if xy is None:
            return
        xi, yi = xy

        if event.button == 1:
            self.roi_xy[xi, yi] = ~self.roi_xy[xi, yi]
        elif event.button == 3:
            self.roi_xy[xi, yi] = False
        self._refresh()

    def on_key(self, event):
        if event.key == "s":
            self.save()
        elif event.key == "c":
            self.roi_xy[:, :] = False
            self._refresh()
        elif event.key == "i":
            self.roi_xy = ~self.roi_xy
            self._refresh()
        elif event.key == "q":
            plt.close(self.fig)

    def save(self):
        out_dir = os.path.dirname(os.path.abspath(self.out_mask))
        os.makedirs(out_dir, exist_ok=True)

        np.save(self.out_mask, self.roi_xy.astype(bool))

        preview_png = os.path.splitext(self.out_mask)[0] + "_preview.png"
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(
            self.anatomy,
            origin="lower",
            cmap="gray",
            extent=[0, self.N, 0, self.N],
            vmin=0,
            vmax=max(1e-6, float(self.anatomy.max())),
        )
        ax2.imshow(
            self.roi_xy.T.astype(float),
            origin="lower",
            cmap="autumn",
            alpha=0.35,
            extent=[0, self.N, 0, self.N],
            vmin=0,
            vmax=1,
        )
        for k in range(self.N + 1):
            ax2.plot([k, k], [0, self.N], color="cyan", lw=0.5, alpha=0.45)
            ax2.plot([0, self.N], [k, k], color="cyan", lw=0.5, alpha=0.45)
        ax2.set_xlim(0, self.N)
        ax2.set_ylim(0, self.N)
        ax2.set_title(f"Manual ROI ({int(self.roi_xy.sum())} voxels)")
        ax2.set_xlabel("x voxel index")
        ax2.set_ylabel("y voxel index")
        fig2.tight_layout()
        fig2.savefig(preview_png, dpi=160)
        plt.close(fig2)

        print(f"[mask] Saved mask: {self.out_mask}")
        print(f"[mask] Saved preview: {preview_png}")


def parse_args():
    p = argparse.ArgumentParser(description="Draw manual ROI mask for 05c SPICE")
    p.add_argument("--dat", required=True, help="Twix .dat file used by 05c")
    p.add_argument("--localizer", required=True, help="Localizer DICOM path (.IMA or dir)")
    p.add_argument("--out-mask", required=True, help="Output .npy path for ROI mask")
    p.add_argument("--backend", default="qtagg",
                   help="Matplotlib GUI backend for drawing (e.g., qtagg, tkagg)")
    p.add_argument("--rep", type=int, default=None,
                   help="Rep index for loader metadata (default: all reps merged metadata)")
    return p.parse_args()


def main():
    args = parse_args()

    # Force interactive backend before importing pyplot. This avoids
    # non-interactive backend side effects from imported reconstruction modules.
    try:
        matplotlib.use(args.backend, force=True)
    except Exception as e:
        sys.exit(f"Failed to set matplotlib backend '{args.backend}': {e}")

    global plt
    import matplotlib.pyplot as plt

    # Reuse the same load + localizer registration logic as 05c.
    _spec_05c = _ilu.spec_from_file_location(
        "_05c", _pl.Path(__file__).parent / "05c_spice_recon_31p.py"
    )
    _05c = _ilu.module_from_spec(_spec_05c)
    _spec_05c.loader.exec_module(_05c)

    # Re-assert requested backend after importing 05c.
    if matplotlib.get_backend().lower() != args.backend.lower():
        try:
            matplotlib.use(args.backend, force=True)
        except Exception:
            pass

    if not args.out_mask.lower().endswith(".npy"):
        sys.exit("--out-mask must end with .npy")

    print(f"[mask] matplotlib backend: {matplotlib.get_backend()}")

    print("[mask] Loading scan metadata...")
    info = _05c.load_concept_twix_3d(args.dat, rep_sel=args.rep)
    N = int(info["matrix_size"])

    print("[mask] Registering localizer to MRSI grid...")
    anatomy = _05c.load_localizer_registered(
        args.localizer,
        N,
        fov_mm=info["fov_mm"],
        pos_sag=info["pos_sag"],
        pos_cor=info["pos_cor"],
        pos_tra=info["pos_tra"],
    )

    editor = MaskEditor(anatomy, args.out_mask, title=f"Manual ROI Mask ({N}x{N})")
    print("[mask] Interactive editor ready.")
    print("[mask] Left click toggle, right click erase, key s save, q quit.")
    plt.show()


if __name__ == "__main__":
    main()

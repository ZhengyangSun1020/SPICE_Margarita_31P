#!/usr/bin/env python3
"""
Interactive MRSI spectrum viewer — click a voxel to see its spectrum.

Usage:
    python 05c_interactive_viewer.py --data-dir ./output/data
    python 05c_interactive_viewer.py --data-dir ./output/data --ppmlim -20 20
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recon_concept_twix_31p import FIDToSpec


def parse_args():
    p = argparse.ArgumentParser(description="Interactive MRSI viewer")
    p.add_argument("--data-dir", required=True, help="Directory with recon_spec.npy / recon_fid.npy")
    p.add_argument("--ppm-center", type=float, default=0.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--dwell", type=float, default=None, help="Spectral dwell time (s). Auto-detected from NIfTI if available.")
    p.add_argument("--larmor", type=float, default=None, help="Larmor freq (MHz). Auto-detected from NIfTI if available.")
    return p.parse_args()


def main():
    args = parse_args()

    spec_path = os.path.join(args.data_dir, "recon_spec.npy")
    fid_path = os.path.join(args.data_dir, "recon_fid.npy")
    nii_path = os.path.join(args.data_dir, "adj_recon.nii.gz")

    if os.path.exists(spec_path):
        image_spec = np.load(spec_path)
    elif os.path.exists(fid_path):
        image_spec = FIDToSpec(np.load(fid_path), axis=-1)
    else:
        print(f"No recon_spec.npy or recon_fid.npy found in {args.data_dir}")
        sys.exit(1)

    Ny, Nx, n_spec = image_spec.shape
    print(f"Loaded: ({Ny}, {Nx}, {n_spec})")

    dwell = args.dwell
    larmor = args.larmor
    if dwell is None or larmor is None:
        if os.path.exists(nii_path):
            try:
                from nifti_mrs.nifti_mrs import NIFTI_MRS
                nii = NIFTI_MRS(nii_path)
                if dwell is None:
                    dwell = nii.dwelltime
                if larmor is None:
                    larmor = float(nii.spectrometer_frequency[0])
            except Exception:
                pass
        if dwell is None:
            dwell = 360e-6
            print(f"Warning: using default dwell={dwell}")
        if larmor is None:
            larmor = 49.895
            print(f"Warning: using default larmor={larmor} MHz")

    FREQ = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell))
    PPM = FREQ / larmor + args.ppm_center

    mag_map = np.mean(np.abs(image_spec), axis=-1)

    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1, 2]})
    ax_map.imshow(mag_map, origin="lower", cmap="viridis")
    ax_map.set_title("Click a voxel")
    marker, = ax_map.plot([], [], 'r+', markersize=15, markeredgewidth=2)

    vy, vx = Ny // 2, Nx // 2
    sp = np.abs(image_spec[vy, vx, :])
    line, = ax_spec.plot(PPM, sp, 'k-', lw=0.8)
    fill = ax_spec.fill_between(PPM, 0, sp, alpha=0.15, color='blue')
    ax_spec.set_xlim(args.ppmlim[1], args.ppmlim[0])
    ax_spec.set_ylim(bottom=0)
    ax_spec.set_xlabel("ppm")
    ax_spec.set_ylabel("|spectrum|")
    ax_spec.set_title(f"Voxel [{vy},{vx}]")

    for pos, name in [(0, 'PCr'), (5, 'Pi'), (-2.5, 'g-ATP'),
                      (-7.5, 'a-ATP'), (-16, 'b-ATP')]:
        ax_spec.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.4)
        ax_spec.text(pos, 0.95, name, fontsize=8, ha='center', color='red',
                     transform=ax_spec.get_xaxis_transform())

    def onclick(event):
        nonlocal fill
        if event.inaxes != ax_map:
            return
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if 0 <= x < Nx and 0 <= y < Ny:
            sp = np.abs(image_spec[y, x, :])
            line.set_ydata(sp)
            fill.remove()
            fill = ax_spec.fill_between(PPM, 0, sp, alpha=0.15, color='blue')
            ax_spec.set_ylim(0, sp.max() * 1.1 + 1e-20)
            ax_spec.set_title(f"Voxel [{y},{x}]")
            marker.set_data([x], [y])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

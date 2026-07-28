#!/usr/bin/env python3
"""
Prepare a small, self-contained review package for the VULCAN-vs-repeat-scan
uncertainty discussion.

The notebook intentionally depends only on numpy/matplotlib. This preparation
script extracts the heavier project outputs into compact .npy/.npz files:

  * repeated SPICE_f stacks from the five repeat scans
  * repeated-scan MC std computed exactly as Uncert_07 does
  * VULCAN posterior std from Uncert_02
  * lightweight diag(V @ mHm @ V^H) diagnostics from the saved Hessian inverse

Run from the repository root:

    python review_vulcan_mc_notebook/prepare_review_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "review_vulcan_mc_notebook"
DATA_OUT = PACK_DIR / "data"

SUBJECT_BASE = "invivo_260623"
REFERENCE_SUBJECT = "invivo_260623_01"
SUBJECTS = [f"{SUBJECT_BASE}_{i:02d}" for i in range(1, 6)]
RUN_TAGS = ["w5000_l0.0001", "w5000_l1e-05", "w5000_l1e-06"]

NY, NX, NSEQ = 64, 64, 300
RANK = 20
PPM_CENTER = 3.027
BRAIN_THRESHOLD = 0.16
BRAIN_EROSION = 1


def tag_key(run_tag: str) -> str:
    return run_tag.replace(".", "p").replace("-", "m")


def fid_to_spec(fid: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm="ortho"), axes=-1)


def normalize_wref(wref: np.ndarray) -> np.ndarray:
    arr = np.abs(np.asarray(wref).squeeze())
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-12)


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small numpy-only 8-neighbor binary erosion for the notebook mask."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        pad = np.pad(out, 1, mode="constant", constant_values=False)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbors.append(pad[1 + dy : 1 + dy + out.shape[0],
                                     1 + dx : 1 + dx + out.shape[1]])
        out = np.logical_and.reduce(neighbors)
    return out


def find_group_std(run_tag: str) -> Path:
    target = f"group_{SUBJECT_BASE}_{run_tag}".lower()
    candidates = [
        p for p in (ROOT / "output").glob(f"group_{SUBJECT_BASE}_*")
        if p.name.lower() == target
    ]
    if not candidates:
        raise FileNotFoundError(f"No group directory found for {run_tag}")
    path = candidates[0] / "prefitting_std.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_scan_params() -> dict:
    path = ROOT / "data" / "processed" / REFERENCE_SUBJECT / "scan_params.json"
    with path.open("r") as f:
        return json.load(f)


def save_common_data() -> dict:
    data_dir = ROOT / "data" / "processed" / REFERENCE_SUBJECT
    wref_norm = normalize_wref(np.load(data_dir / "wref_o.npy", mmap_mode="r"))
    brain_mask = erode_mask(wref_norm > BRAIN_THRESHOLD, BRAIN_EROSION)

    scan_params = load_scan_params()
    ts = (scan_params["k_mrsi"] / NSEQ) * scan_params["dwelltime"]
    sweepwidth = 1.0 / ts
    freq_axis = np.linspace(-sweepwidth / 2, sweepwidth / 2, NSEQ)
    ppm_axis = freq_axis / scan_params["center_freq"] + PPM_CENTER

    np.save(DATA_OUT / "wref_norm.npy", wref_norm.astype(np.float32))
    np.save(DATA_OUT / "brain_mask.npy", brain_mask)
    np.save(DATA_OUT / "ppm_axis.npy", ppm_axis.astype(np.float32))

    return {
        "subject_base": SUBJECT_BASE,
        "reference_subject": REFERENCE_SUBJECT,
        "subjects": SUBJECTS,
        "run_tags": RUN_TAGS,
        "dim": [NY, NX],
        "n_seq": NSEQ,
        "rank": RANK,
        "brain_threshold": BRAIN_THRESHOLD,
        "brain_erosion": BRAIN_EROSION,
        "scan_params": scan_params,
        "notes": [
            "spice_fid_stack_<tag>.npy has shape (5, 64, 64, 300).",
            "mc_std_<tag>.npy is np.std(fid_to_spec(spice_fid_stack), axis=0).",
            "vulcan_std_<tag>.npy is copied from Uncert_02 posterior_std.npy.",
            "vmhmv_diag_fid_<tag>.npy stores diag(2*sigma_noise^2 * V @ mHm @ V^H) in FID domain.",
            "Full per-voxel 300x300 V @ mHm @ V^H covariance is not stored because it is too large.",
        ],
    }


def extract_repeated_scan_data(run_tag: str) -> None:
    key = tag_key(run_tag)
    stack = []
    for subject in SUBJECTS:
        path = ROOT / "output" / subject / f"spice_{run_tag}" / "SPICE_f.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        fid = np.load(path, mmap_mode="r").reshape(NY, NX, NSEQ)
        stack.append(np.asarray(fid, dtype=np.complex64))

    fid_stack = np.stack(stack, axis=0)
    np.save(DATA_OUT / f"spice_fid_stack_{key}.npy", fid_stack)

    spec_stack = fid_to_spec(fid_stack)
    mc_std = np.std(spec_stack, axis=0).astype(np.float32)
    np.save(DATA_OUT / f"mc_std_{key}.npy", mc_std)

    group_std = np.asarray(np.load(find_group_std(run_tag)), dtype=np.float32)
    np.save(DATA_OUT / f"mc_std_group_reference_{key}.npy", group_std)

    finite = np.isfinite(mc_std) & np.isfinite(group_std)
    rel = (
        np.linalg.norm(mc_std[finite] - group_std[finite])
        / (np.linalg.norm(group_std[finite]) + 1e-30)
    )
    print(f"[prepare] {run_tag}: repeated stack {fid_stack.shape}, MC/group relerr={rel:.3e}")


def extract_vulcan_data(run_tag: str) -> None:
    key = tag_key(run_tag)
    out_dir = ROOT / "output" / REFERENCE_SUBJECT
    data_dir = ROOT / "data" / "processed" / REFERENCE_SUBJECT
    spice_dir = out_dir / f"spice_{run_tag}"
    hess_dir = out_dir / f"hessian_{run_tag}"
    uncert_dir = out_dir / f"uncertainty_{run_tag}"

    posterior_std_path = uncert_dir / "posterior_std.npy"
    if not posterior_std_path.exists():
        raise FileNotFoundError(posterior_std_path)
    vulcan_std = np.asarray(np.load(posterior_std_path), dtype=np.float32)
    np.save(DATA_OUT / f"vulcan_std_{key}.npy", vulcan_std)

    V = np.asarray(np.load(spice_dir / "V_subspace.npy")[:, :RANK], dtype=np.complex64)
    sigma_noise = float(np.load(data_dir / "sigma_noise.npy"))
    cov_scale = 2.0 * sigma_noise * sigma_noise

    vmhmv_diag = np.zeros((NY * NX, NSEQ), dtype=np.float32)
    vmhmv_trace = np.zeros(NY * NX, dtype=np.float32)
    files = sorted(hess_dir.glob("mHm_*.npy"))
    if not files:
        raise FileNotFoundError(f"No mHm_*.npy files in {hess_dir}")

    for path in files:
        vox = int(path.stem.split("_")[1])
        mHm = np.asarray(np.load(path), dtype=np.complex64)
        Vm = V @ mHm
        diag = cov_scale * np.real(np.sum(Vm * np.conj(V), axis=1))
        vmhmv_diag[vox, :] = np.maximum(diag, 0.0).astype(np.float32)
        vmhmv_trace[vox] = float(np.sum(vmhmv_diag[vox, :]))

    np.save(DATA_OUT / f"vmhmv_diag_fid_{key}.npy", vmhmv_diag.reshape(NY, NX, NSEQ))
    np.save(DATA_OUT / f"vmhmv_trace_fid_{key}.npy", vmhmv_trace.reshape(NY, NX))

    covered = int(np.count_nonzero(vmhmv_trace))
    print(
        f"[prepare] {run_tag}: VULCAN std {vulcan_std.shape}, "
        f"vmhmv diag covered voxels={covered}"
    )


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    metadata = save_common_data()

    for run_tag in RUN_TAGS:
        extract_repeated_scan_data(run_tag)
        extract_vulcan_data(run_tag)

    with (DATA_OUT / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[prepare] Wrote {DATA_OUT / 'metadata.json'}")


if __name__ == "__main__":
    main()

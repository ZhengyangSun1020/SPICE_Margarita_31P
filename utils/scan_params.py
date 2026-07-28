"""Scan parameter I/O: save/load scan_params.json from processed data folders."""

import json
import os


def load_scan_params(args, data_dir, k_key="k_mrsi"):
    """Fill in None CLI args from scan_params.json in data_dir.

    k_key : "k_mrsi" for MRSI-based scripts, "k_wref" for wref-based (script 02).
    Falls back to hardcoded defaults if JSON is missing.
    """
    path = os.path.join(data_dir.rstrip("/"), "scan_params.json")
    params = {}
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
    else:
        print(f"[scan_params] WARNING: {path} not found — using CLI defaults only.")

    if hasattr(args, "dwelltime")   and args.dwelltime   is None:
        args.dwelltime   = params.get("dwelltime",   5e-6)
    if hasattr(args, "k_points")    and args.k_points    is None:
        args.k_points    = params.get(k_key,          39762)
    if hasattr(args, "n_coils")     and args.n_coils     is None:
        args.n_coils     = params.get("n_coils",      32)
    if hasattr(args, "center_freq") and args.center_freq is None:
        args.center_freq = params.get("center_freq",  297.219338)
    return args


def save_scan_params(out_dir, dwelltime, center_freq, n_coils, res, k_mrsi, k_wref):
    """Write scan_params.json to out_dir."""
    params = {
        "dwelltime":   dwelltime,
        "center_freq": center_freq,
        "n_coils":     n_coils,
        "res":         res,
        "k_mrsi":      k_mrsi,
        "k_wref":      k_wref,
    }
    path = os.path.join(out_dir, "scan_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path

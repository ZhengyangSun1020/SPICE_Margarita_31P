#!/usr/bin/env python3
"""Generate an FSL-MRS 31P basis set (PK_3T_Skeletal_2Pi prior) for fsl_mrsi.

One basis FID per metabolite (PCr, Pi, Pi2, PDE, and the J-coupled γ/α/β-ATP
multiplets). Peaks are placed at their chemical shift in Hz = ppm * centralFreq,
with PCr at 0 (DC). Saved in FSL-MRS basis format via fsl_mrs.core.basis.Basis.

Usage:
    python scripts/specfitting/make_31p_basis.py \
        --larmor 49.895 --out basis_31p_3T          # 3T
    python scripts/specfitting/make_31p_basis.py \
        --larmor 120.6  --out basis_31p_7T          # 7T
"""
import argparse
import numpy as np
from fsl_mrs.core.basis import Basis

# PK_3T_Skeletal_2Pi J-couplings (ppm)
_DB = 15.0/51.0; _DA = 16.0/51.0; _DG = 15.0/51.0
# (fsl name, center_ppm, [(delta_ppm, weight), ...])
METABS = [
    ("PCr",       0.0,    [(0.0, 1.0)]),
    ("Pi",        4.85,   [(0.0, 1.0)]),
    ("Pi2",       4.25,   [(0.0, 1.0)]),
    ("PDE",       3.07,   [(0.0, 1.0)]),
    ("ATP_gamma", -2.53,  [(-_DG/2, 1.0), (_DG/2, 1.0)]),
    ("ATP_alpha", -7.58,  [(-_DA/2, 1.0), (_DA/2, 1.0)]),
    ("ATP_beta",  -15.86, [(-_DB, 0.5), (0.0, 1.0), (_DB, 0.5)]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larmor", type=float, required=True, help="central (spectrometer) freq MHz")
    ap.add_argument("--bw", type=float, default=2778.0, help="bandwidth Hz (fixed)")
    ap.add_argument("--npoints", type=int, default=2048, help="basis FID length")
    ap.add_argument("--linewidth", type=float, default=2.0, help="intrinsic basis LW Hz (small; fsl_mrsi broadens)")
    ap.add_argument("--sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="phase-evolution sign (+1 default; flip to -1 if fit comes out mirrored)")
    ap.add_argument("--single-atp", action="store_true",
                    help="combine γ/α/β-ATP into ONE 'ATP' basis metabolite (1:1:1, fixed) — "
                         "far more stable to fit (β-ATP won't blow up) and still shows all 3 peaks")
    ap.add_argument("--out", required=True, help="output basis directory")
    args = ap.parse_args()

    metabs = list(METABS)
    if args.single_atp:
        atp_comps = []
        for name, c_ppm, comps in METABS:
            if name.startswith("ATP"):
                atp_comps += [(c_ppm + d, w) for d, w in comps]
        metabs = [(n, c, cm) for n, c, cm in METABS if not n.startswith("ATP")]
        metabs.append(("ATP", 0.0, atp_comps))        # absolute-ppm components

    t = np.arange(args.npoints) / args.bw
    fids = []; names = []
    for name, c_ppm, comps in metabs:
        fid = np.zeros(args.npoints, dtype=np.complex128)
        for dppm, w in comps:
            f_hz = (c_ppm + dppm) * args.larmor
            fid += w * np.exp(args.sign * 2j*np.pi*f_hz*t)
        fid *= np.exp(-np.pi*args.linewidth*t)
        fids.append(fid); names.append(name)
    fid_array = np.array(fids).T                       # (time, metab)

    hdr = {"dwelltime": 1.0/args.bw, "bandwidth": args.bw,
           "centralFrequency": args.larmor, "fwhm": args.linewidth}
    headers = [dict(hdr) for _ in names]
    basis = Basis(fid_array, names, headers)
    basis.save(args.out, overwrite=True,
               info_str=f"31P PK_3T_Skeletal_2Pi larmor={args.larmor}MHz bw={args.bw}Hz")
    print(f"[basis] saved {len(names)} metabolites -> {args.out}")
    print(f"[basis] {names}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Robust 31P CRT recon + localizer overlay pipeline.

Given a Siemens Twix .dat (31P CONCEPT CRT, interleaved or pure) and a folder of
localizer DICOMs, this:

  1. Reconstructs the 31P volume with 07b (DCF-adjoint or CG), reusing a cache.
  2. Reads the slab geometry from the cache and computes each MRSI voxel's true
     3D LPS centre from the GSL slice-orientation axes (phase/read/normal).
  3. Loads every DICOM in the localizer folder, groups by SeriesInstanceUID, and
     picks the series to use (auto: the one whose slice normal is most parallel
     to the 31P slab, tie-broken by slice count; or --loc-series N).
  4. Overlays the voxel spectra on each localizer slice by *projecting* the 3D
     voxel centres onto the slice plane and keeping voxels within a half-slab of
     it. This is correct for any geometry: when the localizer is parallel to the
     slab it reduces to the clean per-slice grid; when oblique it shows the true
     intersecting band.

Orientation: the recon is left in its NATIVE orientation by default. 07b's figure
convention (xi up, yi right) already matches the overlay's LPS-based placement, so
native gives correct spectra/spatial-map figures AND a correct overlay. The MATLAB
rot90(fid,1)+flip (from reconUsingNUFFT_3D_splitCoils_VE.m) is available via
--out-rot90 1 --out-flip-lr, but note it CANCELS in the overlay (applied to both
data and grid) and only re-orients the standalone figures — enable it only if you
need .npy data in MATLAB orientation. --pad-z-dim (= MATLAB padZDim: reconstruct on
the Nz+1 grid) is ON by default. The even-N NUFFT centering offset (finufft vs the
Fessler/MATLAB n_shift convention) is corrected inside 07b by a one-voxel in-plane
roll for even matrices, so no display-side registration shift is needed here.
Manual knobs remain if ever needed:
  --flip-xi/--flip-yi/--flip-z   reverse an array axis (manual override only)
  --shift-xi/--shift-yi          optional manual roll of MRSI data (voxels); the
                                 orientation and position come from the header
                                 geometry, so these are normally left at 0
  --out-rot90 1 --out-flip-lr    bake MATLAB data-orientation into the .npy/figures
  --no-pad-z-dim                 disable the Nz+1 recon grid

--out-dir is optional: if omitted it is auto-named output_<dat-basename>[_repN].

Usage — cardiac interleaved (run from ~/SPICE_Margarita):

  ~/miniconda3/envs/zs_SPICE_MARGARITA/bin/python scripts/08_recon_overlay_31p.py \
      --dat "/mnt/f/AliceConceptB1/in_vivo_data_7T/Susuan_0624/meas_MID00047_FID92224_flash_CRT_3d_101010_500meas.dat" \
      --localizer-dir "/mnt/f/AliceConceptB1/in_vivo_data_7T/Susuan_0624/0626_invivo/locolizer" \
      --cache-dir _tmp_recon/cache \
      --loc-series 2
  # -> output_meas_MID00047_FID92224_flash_CRT_3d_101010_500meas/

  # Dynamic data has multiple reps (the count is printed during the recon).
  # Add --rep N (1-indexed) to reconstruct one repetition, e.g. --rep 2
  # -> output_meas_MID00047_..._500meas_rep2/

Usage — phantom (oblique slab, TRA localizer):

  ~/miniconda3/envs/zs_SPICE_MARGARITA/bin/python scripts/08_recon_overlay_31p.py \
      --dat "/mnt/f/AliceConceptB1/phantom_data_7T/0427_PHANTOM_16ch/meas_MID00052_FID88395_flash_3dcrt_12x12x12_20rings_cor_iso_moco.dat" \
      --localizer-dir "/mnt/f/AliceConceptB1/phantom_data_7T/0427_PHANTOM_16ch/F7T_2026_PH_089/localizer_COR_6_None" \
      --cache-dir _tmp_recon/cache
"""
import argparse, os, sys, glob, pickle, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("_r07b", os.path.join(HERE, "07b_interleaved_recon_31p_3d.py"))
_r07b = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_r07b)
_fGSLCalcPRS = _r07b._fGSLCalcPRS


# ── geometry ──────────────────────────────────────────────────────────────────

def voxel_lps_grid(info, flip_xi=False, flip_yi=False, flip_z=False):
    """Return (Nx,Ny,Nz,3) LPS centre of every MRSI voxel.

    Array axes map to physical directions via the CRT ICE convention
    (kx=-dGp, ky=-dGr, kz=+snv), so increasing xi/yi/zi move along
    -dGp / -dGr / +snv respectively (LPS = Sag,Cor,Tra).
    """
    N   = int(info["matrix_size"]); Nz = int(info["Nz"])
    fov = float(info["fov_mm"]); fov_z = float(info["fov_m_z"]) * 1e3
    C   = np.array([info["pos_sag"], info["pos_cor"], info["pos_tra"]], float)
    snv = np.array([info["snv_x"], info["snv_y"], info["snv_z"]], float)
    snv = snv / (np.linalg.norm(snv) + 1e-12)
    ipr = float(info["in_plane_rotation"])
    dGp, dGr = _fGSLCalcPRS(snv, ipr)

    a_xi = -np.asarray(dGp, float)       # increasing xi direction (LPS)
    a_yi = -np.asarray(dGr, float)       # increasing yi direction
    a_zi =  snv                          # increasing zi direction
    if flip_xi: a_xi = -a_xi
    if flip_yi: a_yi = -a_yi
    if flip_z:  a_zi = -a_zi

    vox = fov / N; dz = fov_z / Nz
    # Voxel-centre convention: the N-voxel grid is symmetric about the VoI centre
    # C, i.e. voxel i sits at C + (i - (N-1)/2)*vox. This matches the MATLAB grid
    # (n_shift = ceil(N/2) - 0.5). For even N this differs from N//2 by half a
    # voxel (the "grid a bit up" offset).
    cx = (N - 1) / 2.0; cy = (N - 1) / 2.0; cz = (Nz - 1) / 2.0
    grid = np.zeros((N, N, Nz, 3), float)
    for xi in range(N):
        for yi in range(N):
            for zi in range(Nz):
                grid[xi, yi, zi] = (C + (xi - cx) * vox * a_xi
                                      + (yi - cy) * vox * a_yi
                                      + (zi - cz) * dz  * a_zi)
    return grid, dict(vox=vox, dz=dz, snv=snv, a_xi=a_xi, a_yi=a_yi, a_zi=a_zi, C=C)


# ── localizer loading ─────────────────────────────────────────────────────────

def load_localizer_series(loc_dir, loc_series=None):
    """Group all DICOMs in loc_dir by SeriesInstanceUID; return chosen series
    as a list of dicts sorted by position along the slice normal."""
    files = sorted(glob.glob(os.path.join(loc_dir, "*")))
    series = {}
    for f in files:
        if os.path.isdir(f):
            continue
        try:
            ds = pydicom.dcmread(f)
        except Exception:
            continue
        if not hasattr(ds, "ImageOrientationPatient"):
            continue
        uid = getattr(ds, "SeriesInstanceUID", f)
        series.setdefault(uid, []).append(ds)
    if not series:
        raise RuntimeError(f"No readable localizer DICOMs in {loc_dir}")
    return series


def pick_series(series, slab_normal, loc_series=None):
    def sinfo(items):
        ds0 = items[0]
        iop = np.array([float(x) for x in ds0.ImageOrientationPatient])
        n = np.cross(iop[:3], iop[3:]); n /= (np.linalg.norm(n) + 1e-12)
        return n, int(getattr(ds0, "SeriesNumber", -1))
    cands = []
    for uid, items in series.items():
        n, sn = sinfo(items)
        cands.append((uid, items, n, sn))
    if loc_series is not None:
        for uid, items, n, sn in cands:
            if sn == loc_series:
                return uid, items, n
        raise RuntimeError(f"--loc-series {loc_series} not found "
                           f"(available: {sorted(c[3] for c in cands)})")
    # auto: most parallel to slab normal, tie-broken by slice count
    cands.sort(key=lambda c: (-abs(float(np.dot(c[2], slab_normal))), -len(c[1])))
    uid, items, n, sn = cands[0]
    return uid, items, n


def slice_plane(ds):
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    r = iop[:3]; c = iop[3:]                       # r=along columns, c=along rows
    n = np.cross(r, c); n /= (np.linalg.norm(n) + 1e-12)
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    ps = [float(x) for x in ds.PixelSpacing]       # [row_spacing, col_spacing]
    return dict(r=r, c=c, n=n, ipp=ipp, ps_row=ps[0], ps_col=ps[1],
                rows=int(ds.Rows), cols=int(ds.Columns))


def lps_to_pixel(pt, pl):
    d = pt - pl["ipp"]
    col = float(np.dot(d, pl["r"])) / pl["ps_col"]
    row = float(np.dot(d, pl["c"])) / pl["ps_row"]
    dist = float(np.dot(d, pl["n"]))
    return col, row, dist


# ── overlay ───────────────────────────────────────────────────────────────────

def make_overlay(info, grid, geo, spec, ppm, ppm_win, series_items, slab_n,
                 out_dir, band_mm=None, tag="", y_top=None, noise_norm=True):
    Nx, Ny, Nz = grid.shape[:3]
    vox, dz = geo["vox"], geo["dz"]
    band = band_mm if band_mm is not None else dz          # half-slab each side
    m = (ppm >= min(ppm_win)) & (ppm <= max(ppm_win))
    if m.sum() < 8:
        m = np.ones_like(ppm, bool)
    lo, hi = np.where(m)[0][[0, -1]]; sl = slice(lo, hi + 1); nppm = hi - lo + 1

    # Spectral display amplitude. noise_norm=True (default) scales each voxel by
    # its own FID-tail noise (SNR display). noise_norm=False uses the absolute
    # magnitude, which — together with an externally supplied y_top — gives a
    # single GLOBAL y-scale shared across every slice (and across recons when the
    # same y_top is passed to multiple make_overlay calls).
    tail = slice(int(0.8 * spec.shape[-1]), spec.shape[-1])
    if noise_norm:
        noise = np.abs(spec[..., tail]).std(axis=-1) + 1e-20
        amp = np.abs(spec[..., sl]) / noise[..., None]
    else:
        amp = np.abs(spec[..., sl])
    if y_top is None:
        y_top = float(np.percentile(amp, 99.9)) or 1.0
    # per-voxel peak signal (global-normalised) for intensity tinting
    peak = np.abs(spec[..., sl]).max(axis=-1)
    peak_norm = peak / (peak.max() + 1e-20)

    C = geo["C"]; a_zi = geo["a_zi"]; cz = (Nz - 1) / 2.0
    slice_centers = np.array([C + (zi - cz) * dz * a_zi for zi in range(Nz)])

    items = sorted(series_items, key=lambda d: float(np.dot(
        [float(x) for x in d.ImagePositionPatient],
        np.cross(np.array([float(x) for x in d.ImageOrientationPatient])[:3],
                 np.array([float(x) for x in d.ImageOrientationPatient])[3:]))))
    parallel = abs(float(np.dot(slice_plane(items[0])["n"], slab_n))) > 0.966

    os.makedirs(out_dir, exist_ok=True)
    maplines = ["loc_inst | loc_pos_mm | mrsi_slice_zi | mrsi_slice_dist_mm"]
    for k, ds in enumerate(items):
        pl = slice_plane(ds)
        img = ds.pixel_array.astype(float)
        cell = vox / pl["ps_col"]
        # nearest MRSI slice (by centre distance to this localizer plane)
        dists = np.array([abs(np.dot(sc - pl["ipp"], pl["n"])) for sc in slice_centers])
        zi = int(np.argmin(dists))

        fig, ax = plt.subplots(figsize=(10, 10), facecolor="k")
        ax.set_facecolor("k")
        ax.imshow(img, cmap="gray", origin="upper",
                  vmin=0, vmax=np.percentile(img, 99.5))

        cols, rows = [], []
        for xi in range(Nx):
            for yi in range(Ny):
                col, row, _ = lps_to_pixel(grid[xi, yi, zi], pl)
                cols.append(col); rows.append(row)
                # intensity tint (compact source glows red)
                a = float(peak_norm[xi, yi, zi])
                if a > 0.06:
                    ax.add_patch(plt.Rectangle((col - cell/2, row - cell/2), cell, cell,
                                 fill=True, color="red", alpha=min(0.6, a * 0.6), lw=0))
                s = np.clip(amp[xi, yi, zi] / y_top, 0, 1.05)
                xarr = col + (np.arange(nppm) / max(nppm - 1, 1) - 0.5) * cell * 0.9
                ybase = row + cell * 0.45
                ax.plot(xarr, ybase - s * cell * 0.9, color="#33ff66", lw=0.5)
                ax.add_patch(plt.Rectangle((col - cell/2, row - cell/2), cell, cell,
                             fill=False, ec="cyan", lw=0.3, alpha=0.35))
        maplines.append(f"{int(getattr(ds,'InstanceNumber',k)):d} | "
                        f"{float(np.dot(pl['ipp'], pl['n'])):.1f} | {zi} | {dists[zi]:.1f}")
        # Show the FULL localizer image (grid appears as a box inside), matching
        # the MATLAB overlay — do not crop to the grid extent.
        H, W = img.shape
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)     # origin upper (row 0 at top)
        ax.set_title(f"31P grid on localizer inst {int(getattr(ds,'InstanceNumber',k))} "
                     f"pos={float(np.dot(pl['ipp'],pl['n'])):.0f}mm -> MRSI slice zi={zi} "
                     f"(d={dists[zi]:.0f}mm) | {'PARALLEL' if parallel else 'OBLIQUE'} | "
                     f"ppm[{ppm_win[0]:.0f},{ppm_win[1]:.0f}]",
                     color="w", fontsize=8)
        ax.axis("off")
        fig.savefig(os.path.join(out_dir, f"overlay_loc{k:02d}{tag}.png"),
                    dpi=150, facecolor="k", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved overlay_loc{k:02d} pos={float(np.dot(pl['ipp'],pl['n'])):.0f}mm -> zi={zi}")
    open(os.path.join(out_dir, "slice_mapping.txt"), "w").write("\n".join(maplines) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def find_cache(cache_dir, dat):
    base = os.path.splitext(os.path.basename(dat))[0]
    hits = sorted(glob.glob(os.path.join(cache_dir, f"{base}*.pkl")))
    if not hits:
        raise RuntimeError(f"cache not found for {base} in {cache_dir}")
    return hits[0]


def main():
    p = argparse.ArgumentParser(description="31P CRT recon + localizer overlay")
    p.add_argument("--dat", required=True)
    p.add_argument("--localizer-dir", required=True)
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: output_<dat-basename>[_repN] "
                        "(auto-named from the .dat filename and --rep).")
    p.add_argument("--cache-dir", default="_tmp_recon/cache")
    p.add_argument("--maxiter", type=int, default=0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--apodization", type=float, default=20.0)
    p.add_argument("--ppmlim", type=float, nargs=2, default=[-20, 20])
    p.add_argument("--loc-series", type=int, default=None)
    p.add_argument("--band-mm", type=float, default=None,
                   help="half-thickness (mm) of the voxel band kept per localizer slice "
                        "(default = MRSI slice thickness)")
    # Recon orientation. Default OFF: 07b's figure convention (xi up, yi right)
    # already matches the overlay's LPS placement, so native recon gives correct
    # figures AND overlay. The MATLAB rot90+flip cancels in the overlay (applied
    # to both data and grid) and only mis-orients the standalone figures, so it
    # is left off by default; enable it if you need MATLAB data-orientation .npy.
    p.add_argument("--out-rot90", type=int, default=0,
                   help="07b XY rot90 CCW k baked into recon+figures (default 0)")
    p.add_argument("--out-flip-lr", action=argparse.BooleanOptionalAction, default=False,
                   help="07b XY in-plane flip baked into recon+figures (default off)")
    p.add_argument("--pad-z-dim", action=argparse.BooleanOptionalAction, default=True,
                   help="07b MATLAB padZDim: reconstruct on Nz+1 grid (default on)")
    p.add_argument("--rep", type=int, default=None,
                   help="reconstruct a single repetition (1-indexed); omit for first "
                        "complete k-space. Rep count is printed during the recon.")
    p.add_argument("--rep-average", action="store_true",
                   help="average all reps' k-space before recon (√N SNR); overrides --rep")
    p.add_argument("--shift-xi", type=int, default=0,
                   help="optional manual roll of MRSI data along axis0 (voxels); "
                        "orientation/position come from the header, so normally 0")
    p.add_argument("--shift-yi", type=int, default=0, help="roll MRSI data along axis1")
    p.add_argument("--flip-xi", action="store_true")
    p.add_argument("--flip-yi", action="store_true")
    p.add_argument("--flip-z", action="store_true")
    p.add_argument("--skip-recon", action="store_true",
                   help="reuse existing recon_spec.npy (skip 07b)")
    args = p.parse_args()

    # Auto-name the output dir from the .dat basename (+ rep) when not given.
    if args.out_dir is None:
        base = os.path.splitext(os.path.basename(args.dat))[0]
        rep_suffix = f"_rep{args.rep}" if args.rep is not None else ""
        args.out_dir = f"output_{base}{rep_suffix}"
        print(f"[pipeline] auto out-dir: {args.out_dir}")

    spec_npy = os.path.join(args.out_dir, "data", "recon_spec.npy")
    if not (args.skip_recon and os.path.exists(spec_npy)):
        cmd = [sys.executable, os.path.join(HERE, "07b_interleaved_recon_31p_3d.py"),
               "--dat", args.dat, "--out-dir", args.out_dir,
               "--cache-dir", args.cache_dir,
               "--maxiter", str(args.maxiter), "--lam", str(args.lam),
               "--apodization", str(args.apodization),
               "--ppmlim", str(args.ppmlim[0]), str(args.ppmlim[1]),
               "--out-rot90", str(args.out_rot90)]
        if args.out_flip_lr:
            cmd.append("--out-flip-lr")
        if args.pad_z_dim:
            cmd.append("--pad-z-dim")
        if args.rep is not None:
            cmd += ["--rep", str(args.rep)]
        if args.rep_average:
            cmd.append("--rep-average")
        print("[pipeline] running 07b recon ...")
        subprocess.run(cmd, check=True)

    info = pickle.load(open(find_cache(args.cache_dir, args.dat), "rb"))
    spec = np.abs(np.load(spec_npy))                 # (Nx,Ny,Nz_out,n_spec)
    Nz_out = spec.shape[2]

    grid, geo = voxel_lps_grid(info, args.flip_xi, args.flip_yi, args.flip_z)
    # 07b bakes the MATLAB orientation (rot90 + flip) into recon_spec; apply the
    # SAME transform to the LPS grid so grid[a,b,z] stays aligned with the
    # oriented recon_spec[a,b,z] for the overlay/reslice.
    if int(args.out_rot90) % 4:
        grid = np.rot90(grid, k=int(args.out_rot90) % 4, axes=(0, 1))
    if args.out_flip_lr:
        grid = np.flip(grid, axis=1)
    Nz_raw = grid.shape[2]
    # match recon output Nz (07b may interpolate Nz_raw -> Nz_raw+1); resample grid z
    if Nz_out != Nz_raw:
        zt = np.linspace(0, Nz_raw - 1, Nz_out)
        grid = np.stack([
            np.stack([[np.interp(zt, np.arange(Nz_raw), grid[xi, yi, :, d])
                       for d in range(3)] for yi in range(grid.shape[1])], 0)
            for xi in range(grid.shape[0])], 0).transpose(0, 1, 3, 2)

    # (MATLAB orientation is baked into the recon via --out-rot90/--out-flip-lr,
    #  and the same transform was applied to `grid` above.)
    # optional registration shift (array roll) — normally not needed
    if args.shift_xi or args.shift_yi:
        spec = np.roll(spec, (args.shift_xi, args.shift_yi), axis=(0, 1))
        print(f"[pipeline] applied MRSI shift xi={args.shift_xi} yi={args.shift_yi}")

    ns = spec.shape[-1]
    freq = np.fft.fftshift(np.fft.fftfreq(ns, d=float(info["spectral_dwell"])))
    ppm = -freq / float(info["larmor_freq_mhz"])

    series = load_localizer_series(args.localizer_dir)
    slab_n = geo["snv"]
    uid, items, loc_n = pick_series(series, slab_n, args.loc_series)
    print(f"[pipeline] localizer series: SeriesNumber="
          f"{int(getattr(items[0],'SeriesNumber',-1))} "
          f"desc={getattr(items[0],'SeriesDescription','?')} n={len(items)} "
          f"normal={loc_n.round(2).tolist()}  slab_normal={slab_n.round(2).tolist()}  "
          f"|dot|={abs(float(np.dot(loc_n,slab_n))):.3f}")

    out_ov = os.path.join(args.out_dir, "overlay")
    make_overlay(info, grid, geo, spec, ppm, tuple(args.ppmlim),
                 items, slab_n, out_ov, band_mm=args.band_mm)
    print(f"[pipeline] overlays -> {out_ov}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Register MID00073 spectra into MID00070 space and plot voxel-wise overlays.

Produces MATLAB-like spaced grids per slice with black=reference MID00070 and
red=registered MID00073 spectra.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

try:
    import mapvbvd
except ImportError as exc:
    raise SystemExit("pymapvbvd required: pip install pymapvbvd") from exc


def _hdr_float(hdr, section, key_tuple, default=None):
    val = hdr.get(section, {}).get(tuple(key_tuple), None)
    if val is None:
        return default
    return float(val)


def _fGSLClassOri(dGs):
    s, c, t = abs(dGs[0]), abs(dGs[1]), abs(dGs[2])
    eq_sc, eq_st, eq_ct = abs(s-c) <= 1e-6, abs(s-t) <= 1e-6, abs(c-t) <= 1e-6
    if ((eq_sc and eq_st) or (eq_sc and s < t) or (eq_st and s > c) or
            (eq_ct and c > s) or (s > c and s < t) or (s < c and c < t) or
            (s < t and t > c) or (c < t and t > s)):
        return 2
    if ((eq_sc and s > t) or (eq_st and s < c) or (s < c and c > t) or
            (s > t and s < c) or (s < t and t < c)):
        return 1
    if ((eq_ct and c < s) or (s > c and s > t) or (c > t and c < s) or
            (c < t and t < s)):
        return 0
    raise ValueError("Invalid slice orientation")


def _fGSLCalcPRS(dGs, dPhi):
    ori = _fGSLClassOri(dGs)
    if ori == 2:
        d = np.sqrt(dGs[1] ** 2 + dGs[2] ** 2)
        dGp = np.array([0.0, dGs[2] / d, -dGs[1] / d])
    elif ori == 1:
        d = np.sqrt(dGs[0] ** 2 + dGs[1] ** 2)
        dGp = np.array([dGs[1] / d, -dGs[0] / d, 0.0])
    else:
        d = np.sqrt(dGs[0] ** 2 + dGs[1] ** 2)
        dGp = np.array([-dGs[1] / d, dGs[0] / d, 0.0])
    dGr = np.cross(dGs, dGp)
    if abs(dPhi) > 1e-10:
        co, si = np.cos(dPhi), np.sin(dPhi)
        dGp = co * dGp - si * dGr
        dGr = np.cross(dGs, dGp)
    return dGp, dGr


def _load_geom(dat_path, mode):
    twix = mapvbvd.mapVBVD(dat_path)
    if isinstance(twix, list):
        twix = twix[-1]
    hdr = twix["hdr"]

    if mode == "06c":
        fov_mm = _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "dReadoutFOV"), default=300.0)
        # For 3D slab geometry, prefer SliceArray thickness (acquisition slab).
        # sSpecPara.sVoI.dThickness may reflect VOI/shim settings and can be smaller.
        thick_mm = _hdr_float(hdr, "MeasYaps", ("sSliceArray", "asSlice", "0", "dThickness"), default=None)
        if thick_mm is None:
            thick_mm = _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "dThickness"), default=170.0)
        pos = np.array([
            _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "sPosition", "dSag"), 0.0),
            _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "sPosition", "dCor"), 0.0),
            _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "sPosition", "dTra"), 0.0),
        ], dtype=np.float64)
        snv = np.array([
            _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "sNormal", "dSag"), 0.0),
            _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "sNormal", "dCor"), 0.0),
            _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "sNormal", "dTra"), 1.0),
        ], dtype=np.float64)
        ipr = _hdr_float(hdr, "MeasYaps", ("sSpecPara", "sVoI", "dInPlaneRot"), 0.0)
    else:
        nav = ("sNavigatorArray", "asElm", "0", "sCuboid")
        fov_mm = _hdr_float(hdr, "MeasYaps", nav + ("dReadoutFOV",), default=300.0)
        thick_mm = _hdr_float(hdr, "MeasYaps", nav + ("dThickness",), default=300.0)
        pos = np.array([
            _hdr_float(hdr, "MeasYaps", nav + ("sPosition", "dSag"), 0.0),
            _hdr_float(hdr, "MeasYaps", nav + ("sPosition", "dCor"), 0.0),
            _hdr_float(hdr, "MeasYaps", nav + ("sPosition", "dTra"), 0.0),
        ], dtype=np.float64)
        snv = np.array([
            _hdr_float(hdr, "MeasYaps", nav + ("sNormal", "dSag"), 0.0),
            _hdr_float(hdr, "MeasYaps", nav + ("sNormal", "dCor"), 0.0),
            _hdr_float(hdr, "MeasYaps", nav + ("sNormal", "dTra"), 1.0),
        ], dtype=np.float64)
        ipr = _hdr_float(hdr, "MeasYaps", nav + ("dInPlaneRot",), 0.0)

    snv = snv / (np.linalg.norm(snv) + 1e-20)
    dGp, dGr = _fGSLCalcPRS(snv, ipr)
    dGp = dGp / (np.linalg.norm(dGp) + 1e-20)
    dGr = dGr / (np.linalg.norm(dGr) + 1e-20)

    return {
        "fov_mm": float(fov_mm),
        "thick_mm": float(thick_mm),
        "pos_mm": pos,
        "snv": snv,
        "phase": dGp,
        "read": dGr,
        "ipr": float(ipr),
    }


def _voxel_center_world(ix, iy, iz, geom, shape):
    nx, ny, nz = shape
    vx = geom["fov_mm"] / float(nx)
    vy = geom["fov_mm"] / float(ny)
    vz = geom["thick_mm"] / float(nz)
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    cz = (nz - 1) * 0.5

    return (geom["pos_mm"]
            + (ix - cx) * vx * geom["phase"]
            + (iy - cy) * vy * geom["read"]
            + (iz - cz) * vz * geom["snv"])


def _world_to_voxel(world_mm, geom, shape):
    nx, ny, nz = shape
    vx = geom["fov_mm"] / float(nx)
    vy = geom["fov_mm"] / float(ny)
    vz = geom["thick_mm"] / float(nz)
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    cz = (nz - 1) * 0.5

    d = world_mm - geom["pos_mm"]
    ix = np.dot(d, geom["phase"]) / vx + cx
    iy = np.dot(d, geom["read"]) / vy + cy
    iz = np.dot(d, geom["snv"]) / vz + cz
    return ix, iy, iz


def _trilinear_sample(vol, x, y, z):
    nx, ny, nz, ns = vol.shape
    if x < 0 or y < 0 or z < 0 or x > nx - 1 or y > ny - 1 or z > nz - 1:
        return np.zeros(ns, dtype=vol.dtype)

    x0 = int(np.floor(x)); x1 = min(x0 + 1, nx - 1)
    y0 = int(np.floor(y)); y1 = min(y0 + 1, ny - 1)
    z0 = int(np.floor(z)); z1 = min(z0 + 1, nz - 1)
    dx = x - x0; dy = y - y0; dz = z - z0

    c000 = vol[x0, y0, z0]
    c001 = vol[x0, y0, z1]
    c010 = vol[x0, y1, z0]
    c011 = vol[x0, y1, z1]
    c100 = vol[x1, y0, z0]
    c101 = vol[x1, y0, z1]
    c110 = vol[x1, y1, z0]
    c111 = vol[x1, y1, z1]

    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy

    return c0 * (1 - dz) + c1 * dz


def _map_xy_with_rot90(x, y, nx, ny, rot90_k):
    """Map XY coordinates from canonical orientation into stored array orientation.

    rot90_k follows numpy.rot90 semantics on axes (0, 1):
      0=no rotation, 1=90deg CCW, 2=180deg, 3=270deg CCW.
    """
    k = int(rot90_k) % 4
    if k == 0:
        return x, y
    if k == 1:
        return (nx - 1) - y, x
    if k == 2:
        return (nx - 1) - x, (ny - 1) - y
    return y, (ny - 1) - x


def register_volume_to_reference(spec_ref, geom_ref, spec_mov, geom_mov,
                                 mov_rot90_k=0, mov_flip_lr=False):
    nx, ny, nz, ns = spec_ref.shape
    out = np.zeros_like(spec_ref)
    z_map_center = np.zeros(nz, dtype=np.float64)

    sx, sy, sz, _ = spec_mov.shape
    for iz in range(nz):
        world_center = _voxel_center_world((nx - 1) * 0.5, (ny - 1) * 0.5, iz, geom_ref, (nx, ny, nz))
        _, _, zf_center = _world_to_voxel(world_center, geom_mov, (sx, sy, sz))
        z_map_center[iz] = zf_center
        for iy in range(ny):
            for ix in range(nx):
                world = _voxel_center_world(ix, iy, iz, geom_ref, (nx, ny, nz))
                xf, yf, zf = _world_to_voxel(world, geom_mov, (sx, sy, sz))
                xf, yf = _map_xy_with_rot90(xf, yf, sx, sy, mov_rot90_k)
                if mov_flip_lr:
                    yf = (sy - 1) - yf
                out[ix, iy, iz] = _trilinear_sample(spec_mov, xf, yf, zf)
    return out, z_map_center


def _plot_overlay_slice(spec_a, spec_b_reg, ppm, z, out_png, title):
    nx, ny, nz, ns = spec_a.shape
    tail = slice(int(0.8 * ns), ns)
    xlim = (float(np.max(ppm)), float(np.min(ppm)))

    fig, axes = plt.subplots(nx, ny, figsize=(ny * 1.6, nx * 1.25), squeeze=False)
    axes = np.array(axes).reshape(nx, ny)
    y_top = 0.0

    for yi in range(ny):
        for xi in range(nx):
            ax = axes[nx - 1 - xi, yi]
            sa = np.nan_to_num(np.abs(spec_a[xi, yi, z]))
            sb = np.nan_to_num(np.abs(spec_b_reg[xi, yi, z]))
            na = sa / (np.std(sa[tail]) + 1e-20)
            nb = sb / (np.std(sb[tail]) + 1e-20)
            y_top = max(y_top, float(np.percentile(na, 99.5)), float(np.percentile(nb, 99.5)))
            ax.plot(ppm, na, color="black", lw=0.45)
            ax.plot(ppm, nb, color="red", lw=0.45, alpha=0.9)
            ax.set_xlim(xlim)
            ax.axis("off")

    if y_top <= 0:
        y_top = 1.0
    for yi in range(ny):
        for xi in range(nx):
            axes[nx - 1 - xi, yi].set_ylim(0.0, y_top * 1.05)

    fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.05, top=0.92,
                        wspace=0.22, hspace=0.22)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _load_localizer_slices(localizer_dir):
    records = []
    for root, _, files in os.walk(localizer_dir):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(path, force=True)
            except Exception:
                continue
            if not hasattr(ds, "PixelData"):
                continue
            if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
                continue
            if not hasattr(ds, "PixelSpacing"):
                continue

            img = ds.pixel_array.astype(np.float64)
            p2, p99 = np.percentile(img, [2.0, 99.5])
            img = np.clip((img - p2) / (p99 - p2 + 1e-12), 0.0, 1.0)

            ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
            row_dir = iop[:3]
            col_dir = iop[3:]
            normal = np.cross(row_dir, col_dir)
            normal = normal / (np.linalg.norm(normal) + 1e-20)
            row_dir = row_dir / (np.linalg.norm(row_dir) + 1e-20)
            col_dir = col_dir / (np.linalg.norm(col_dir) + 1e-20)
            sp_row = float(ds.PixelSpacing[0])
            sp_col = float(ds.PixelSpacing[1])

            records.append({
                "path": path,
                "img": img,
                "ipp": ipp,
                "row_dir": row_dir,
                "col_dir": col_dir,
                "normal": normal,
                "sp_row": sp_row,
                "sp_col": sp_col,
                "rows": int(img.shape[0]),
                "cols": int(img.shape[1]),
            })

    if not records:
        raise RuntimeError(f"No readable DICOM slices found in {localizer_dir}")

    n_ref = records[0]["normal"]
    for r in records:
        if np.dot(r["normal"], n_ref) < 0:
            r["normal"] *= -1.0
            r["row_dir"] *= -1.0

    records.sort(key=lambda r: float(np.dot(r["ipp"], n_ref)))
    return records


def _world_to_localizer_rc(world_mm, rec):
    d = world_mm - rec["ipp"]
    row = np.dot(d, rec["row_dir"]) / rec["sp_row"]
    col = np.dot(d, rec["col_dir"]) / rec["sp_col"]
    return float(row), float(col)


def _find_nearest_localizer_slice(world_center_mm, records):
    dists = [abs(float(np.dot(world_center_mm - r["ipp"], r["normal"]))) for r in records]
    return int(np.argmin(dists)), float(np.min(dists))


def _plot_localizer_background_overlay(spec_a, spec_b_reg, ppm, z, geom_ref, rec, out_png, title):
    nx, ny, nz, ns = spec_a.shape
    vx = geom_ref["fov_mm"] / float(nx)
    vy = geom_ref["fov_mm"] / float(ny)
    half_p = 0.5 * vx * geom_ref["phase"]
    half_r = 0.5 * vy * geom_ref["read"]

    tail = slice(int(0.8 * ns), ns)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(rec["img"], cmap="gray", origin="upper")

    n_drawn = 0
    for ix in range(nx):
        for iy in range(ny):
            c = _voxel_center_world(ix, iy, z, geom_ref, (nx, ny, nz))
            corners = [
                c - half_p - half_r,
                c - half_p + half_r,
                c + half_p + half_r,
                c + half_p - half_r,
            ]
            rc = np.array([_world_to_localizer_rc(w, rec) for w in corners], dtype=np.float64)
            rows = rc[:, 0]
            cols = rc[:, 1]

            r0, r1 = float(np.min(rows)), float(np.max(rows))
            c0, c1 = float(np.min(cols)), float(np.max(cols))
            h = r1 - r0
            w = c1 - c0

            # Keep all voxels visible even when projected box is tiny/off-edge.
            if h < 2.0:
                rc_mid = float(np.mean(rows))
                r0, r1 = rc_mid - 1.0, rc_mid + 1.0
                h = 2.0
            if w < 2.0:
                cc_mid = float(np.mean(cols))
                c0, c1 = cc_mid - 1.0, cc_mid + 1.0
                w = 2.0

            poly_c = np.r_[cols, cols[0]]
            poly_r = np.r_[rows, rows[0]]
            ax.plot(poly_c, poly_r, color="yellow", lw=0.35, alpha=0.35)

            sa = np.nan_to_num(np.abs(spec_a[ix, iy, z]))
            sb = np.nan_to_num(np.abs(spec_b_reg[ix, iy, z]))
            na = sa / (np.std(sa[tail]) + 1e-20)
            nb = sb / (np.std(sb[tail]) + 1e-20)
            smax = max(float(np.percentile(na, 99.5)), float(np.percentile(nb, 99.5)), 1e-8)
            na = np.clip(na / smax, 0.0, 1.5)
            nb = np.clip(nb / smax, 0.0, 1.5)

            pad_x = 0.10 * w
            pad_y = 0.14 * h
            x0 = c0 + pad_x
            x1 = c1 - pad_x
            yb = r1 - pad_y
            yh = max(h - 2 * pad_y, 1.0)
            if x1 <= x0:
                cx = 0.5 * (c0 + c1)
                x0, x1 = cx - 1.0, cx + 1.0

            xs = np.linspace(x0, x1, ns)
            ya = yb - na * yh
            yb2 = yb - nb * yh
            ax.plot(xs, ya, color="black", lw=0.30, alpha=0.95)
            ax.plot(xs, yb2, color="red", lw=0.30, alpha=0.90)
            n_drawn += 1

    ax.set_xlim(0, rec["cols"])
    ax.set_ylim(rec["rows"], 0)
    ax.axis("off")
    ax.set_title(f"{title} | voxels drawn={n_drawn}/{nx*ny}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Register MID00073 to MID00070 and compare spectra")
    p.add_argument("--dat70", required=True)
    p.add_argument("--dat73", required=True)
    p.add_argument("--spec70", required=True)
    p.add_argument("--spec73", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lb-hz", type=float, default=20.0)
    p.add_argument("--dwell-s", type=float, default=1.0 / 2778.0)
    p.add_argument("--larmor-mhz", type=float, default=120.315736)
    p.add_argument("--localizer-dir", type=str, default=None,
                   help="Optional DICOM localizer folder; overlays spectra on localizer background")
    p.add_argument("--mov-rot90", type=int, default=0,
                   help="Rotate moving spectra map in XY by k*90deg before registration sampling "
                        "(numpy.rot90 convention on axes (0,1); use 1 for +90deg CCW)")
    p.add_argument("--mov-flip-lr", action="store_true",
                   help="Flip moving spectra map left-right in display space after --mov-rot90")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    fig_loc_dir = os.path.join(args.out_dir, "figures_localizer_bg")
    if args.localizer_dir:
        os.makedirs(fig_loc_dir, exist_ok=True)

    spec70 = np.load(args.spec70)
    spec73 = np.load(args.spec73)
    if spec70.shape != spec73.shape:
        raise ValueError(f"Shape mismatch: {spec70.shape} vs {spec73.shape}")

    geom70 = _load_geom(args.dat70, mode="06c")
    geom73 = _load_geom(args.dat73, mode="07b")

    spec73_reg, zmap = register_volume_to_reference(
        spec70, geom70, spec73, geom73,
        mov_rot90_k=args.mov_rot90,
        mov_flip_lr=args.mov_flip_lr)
    np.save(os.path.join(data_dir, "mid73_registered_to_mid70_spec.npy"), spec73_reg)

    ns = spec70.shape[-1]
    freq = np.fft.fftshift(np.fft.fftfreq(ns, d=args.dwell_s))
    ppm = -freq / args.larmor_mhz

    for z in range(spec70.shape[2]):
        title = (
            f"Slice {z + 1}/{spec70.shape[2]} | 12x12 | LB={args.lb_hz:.0f} Hz "
            f"| MID00070 (blk) vs MID00073 registered (red) | norm=noise "
            f"| mapped z73={zmap[z]:.2f}"
        )
        out_png = os.path.join(fig_dir, f"registered_overlay_z{z:02d}.png")
        _plot_overlay_slice(spec70, spec73_reg, ppm, z, out_png, title)

    loc_slice_index = None
    loc_slice_dist_mm = None
    if args.localizer_dir:
        loc_records = _load_localizer_slices(args.localizer_dir)
        loc_slice_index = []
        loc_slice_dist_mm = []
        nx, ny, nz, _ = spec70.shape
        for z in range(nz):
            w_center = _voxel_center_world((nx - 1) * 0.5, (ny - 1) * 0.5, z, geom70, (nx, ny, nz))
            idx, dist_mm = _find_nearest_localizer_slice(w_center, loc_records)
            loc_slice_index.append(idx)
            loc_slice_dist_mm.append(dist_mm)

            title = (
                f"Localizer + spectra | slice {z + 1}/{nz} | MID00070 blk vs MID00073 reg red "
                f"| localizer idx={idx} | center-plane dist={dist_mm:.2f} mm"
            )
            out_png = os.path.join(fig_loc_dir, f"registered_overlay_localizer_z{z:02d}.png")
            _plot_localizer_background_overlay(spec70, spec73_reg, ppm, z, geom70,
                                               loc_records[idx], out_png, title)

    with open(os.path.join(args.out_dir, "registration_report.txt"), "w", encoding="ascii") as f:
        f.write("MID00070 geometry\n")
        f.write(f"  fov_xy_mm={geom70['fov_mm']:.3f}  thick_mm={geom70['thick_mm']:.3f}\n")
        f.write(f"  pos_mm={geom70['pos_mm'].tolist()}\n")
        f.write(f"  snv={geom70['snv'].tolist()}  phase={geom70['phase'].tolist()}  read={geom70['read'].tolist()}\n")
        f.write("MID00073 geometry\n")
        f.write(f"  fov_xy_mm={geom73['fov_mm']:.3f}  thick_mm={geom73['thick_mm']:.3f}\n")
        f.write(f"  pos_mm={geom73['pos_mm'].tolist()}\n")
        f.write(f"  snv={geom73['snv'].tolist()}  phase={geom73['phase'].tolist()}  read={geom73['read'].tolist()}\n")
        f.write(f"moving_xy_rot90_k={int(args.mov_rot90) % 4}\n")
        f.write(f"moving_flip_lr={bool(args.mov_flip_lr)}\n")
        f.write("z mapping (center voxel) from MID00070 slices into MID00073 index\n")
        f.write("  " + ", ".join([f"{v:.3f}" for v in zmap.tolist()]) + "\n")
        if loc_slice_index is not None:
            f.write("localizer slice indices chosen per MID00070 z\n")
            f.write("  " + ", ".join([str(v) for v in loc_slice_index]) + "\n")
            f.write("localizer center-plane distance (mm) per MID00070 z\n")
            f.write("  " + ", ".join([f"{v:.3f}" for v in loc_slice_dist_mm]) + "\n")

    print(f"[reg] saved registered spectra: {os.path.join(data_dir, 'mid73_registered_to_mid70_spec.npy')}")
    print(f"[reg] figures: {fig_dir}")
    if args.localizer_dir:
        print(f"[reg] localizer overlays: {fig_loc_dir}")
    print(f"[reg] report: {os.path.join(args.out_dir, 'registration_report.txt')}")


if __name__ == "__main__":
    main()

"""
Reconstruct 31P CRT MRSI using mri-nufft (finufft backend).

Key improvement over torchkbnufft version: correct physics-based trajectory
normalization matching the MATLAB CRT convention.

Trajectory normalization:
    k_norm = k_raw / (DeltaGM * N)
    DeltaGM = 1e9 / (FoV_m * GyroMagnRatioOverTwoPi)

This maps the trajectory to [-0.5, 0.5], which is the native mri-nufft convention.

Usage:
    python recon_crt_mrinufft.py --mat path/to/rearranged.mat
    python recon_crt_mrinufft.py --mat path/to/rearranged.mat --coil 12 --method cg --lambd 0.01
"""

from __future__ import annotations
import argparse, sys, time
import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.io
except ImportError:
    sys.exit("scipy is required: pip install scipy")

try:
    import mrinufft
except ImportError:
    sys.exit("mri-nufft is required: pip install mri-nufft")


# =====================================================================
# 1. Load rearranged .mat
# =====================================================================
def load_rearranged_mat(mat_path: str) -> dict:
    """Load the rearranged CRT .mat (v5 format) via scipy.io.loadmat."""
    raw = scipy.io.loadmat(mat_path, squeeze_me=False)
    csi = raw["csi"]

    def _get(struct, name):
        v = struct[name]
        if v.shape == (1, 1) and v.dtype == object:
            return np.squeeze(v[0, 0])
        return np.squeeze(v)

    # ---- csi.Par ----
    par = csi["Par"][0, 0]

    dwell_ns        = float(_get(par, "Dwelltimes"))
    larmor_freq_hz  = float(_get(par, "LarmorFreq"))
    gamma_2pi       = float(_get(par, "GyroMagnRatioOverTwoPi"))  # Hz/T
    n_coils_reco    = int(_get(par, "total_channel_no_reco"))
    n_freq_enc      = int(_get(par, "nFreqEnc"))
    n_phas_enc      = int(_get(par, "nPhasEnc"))
    fov_read_mm     = float(_get(par, "FoV_Read"))

    dwell_s         = dwell_ns * 1e-9
    spectral_bw_hz  = 1.0 / dwell_s
    larmor_freq_mhz = larmor_freq_hz * 1e-6
    fov_m           = fov_read_mm * 1e-3

    # ---- csi.RecoPar ----
    rp = csi["RecoPar"][0, 0]

    n_rings   = int(_get(rp, "nAngInts"))
    traj_pts  = np.atleast_1d(_get(rp, "TrajPts").astype(int))
    if traj_pts.size == 1:
        traj_pts = np.full(n_rings, int(traj_pts))
    n_spec    = int(_get(rp, "vecSize"))
    fov_rp_mm = float(_get(rp, "FoV_Read"))
    matrix_size = n_freq_enc

    # Slice position (for FOV shift correction)
    pos_cor = float(_get(rp, "Pos_Cor"))   # mm
    pos_sag = float(_get(rp, "Pos_Sag"))   # mm
    pos_tra = float(_get(rp, "Pos_Tra"))   # mm
    snv_x   = float(_get(rp, "SliceNormalVector_x"))
    snv_y   = float(_get(rp, "SliceNormalVector_y"))
    snv_z   = float(_get(rp, "SliceNormalVector_z"))
    ipr     = float(_get(rp, "InPlaneRotation"))  # rad

    # FirstCirclekSpacePoint (2, n_rings)
    fcp_raw = rp["FirstCirclekSpacePoint"]
    if fcp_raw.shape == (1, 1) and fcp_raw.dtype == object:
        first_circle = np.array(fcp_raw[0, 0], dtype=float)
    else:
        first_circle = np.squeeze(np.array(fcp_raw, dtype=float))

    # ---- csi.Data ----
    csi_data = csi["Data"][0, 0]
    cells = csi_data.ravel()
    kdata_list = []
    for i in range(n_rings):
        d = np.squeeze(cells[i])  # (kpts, vecSize, coils)
        kdata_list.append(d)
    kdata = np.stack(kdata_list, axis=-1)  # (kpts, vecSize, coils, n_rings)

    # ---- Spectral axes ----
    freq_hz  = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
    ppm_axis = freq_hz / larmor_freq_mhz

    # ---- Derived trajectory parameters ----
    delta_gm = 1e9 / (fov_m * gamma_2pi)   # mT/m·µs per k-step

    info = dict(
        kdata=kdata.astype(np.complex64),
        n_rings=n_rings, traj_pts=traj_pts, n_spec=n_spec,
        n_coils=n_coils_reco, matrix_size=matrix_size,
        n_freq_enc=n_freq_enc, n_phas_enc=n_phas_enc,
        dwell_ns=dwell_ns, dwell_s=dwell_s,
        spectral_bw_hz=spectral_bw_hz,
        larmor_freq_hz=larmor_freq_hz,
        larmor_freq_mhz=larmor_freq_mhz,
        fov_mm=fov_read_mm, fov_m=fov_m,
        gamma_2pi=gamma_2pi, delta_gm=delta_gm,
        first_circle=first_circle,
        freq_hz=freq_hz, ppm_axis=ppm_axis,
        # Slice position & orientation (for FOV shift)
        pos_cor=pos_cor, pos_sag=pos_sag, pos_tra=pos_tra,
        snv_x=snv_x, snv_y=snv_y, snv_z=snv_z,
        in_plane_rotation=ipr,
    )

    print(f"Loaded: {mat_path}")
    print(f"  Dwelltimes     = {dwell_ns:.0f} ns  ->  BW = {spectral_bw_hz:.1f} Hz")
    print(f"  LarmorFreq     = {larmor_freq_hz:.0f} Hz  ({larmor_freq_mhz:.3f} MHz)")
    print(f"  GyroMagnR/2pi  = {gamma_2pi:.0f} Hz/T  (31P)")
    print(f"  FOV            = {fov_read_mm:.0f} mm,  matrix = {matrix_size}x{matrix_size}")
    print(f"  DeltaGM        = {delta_gm:.2f} mT/m·µs")
    print(f"  nAngInts={n_rings}, TrajPts={traj_pts[0]}, vecSize={n_spec}, coils={n_coils_reco}")
    print(f"  Pos_Sag={pos_sag:.2f}, Pos_Cor={pos_cor:.2f}, Pos_Tra={pos_tra:.2f} mm")
    print(f"  SliceNormalVec=[{snv_x},{snv_y},{snv_z}], InPlaneRot={ipr:.4f}")
    print(f"  kdata shape    = {kdata.shape}  (kpts, spectral, coils, rings)")
    return info


# =====================================================================
# 2. Build CRT trajectory in [-0.5, 0.5] for mri-nufft
# =====================================================================
def build_crt_trajectory(info: dict):
    """Compute CRT trajectory and normalize to [-0.5, 0.5] using physics.

    Follows io_ReadCRTTraj.m for ring geometry:
        Radii = sqrt(fcp(1,:)^2 + fcp(2,:)^2)
        phi0  = -atan2(fcp(2,:), fcp(1,:)) - pi/2
        Phi   = (0:-1:-TrajPts+1) * 2*pi/TrajPts  (clockwise)

    Then normalizes:
        k_norm = k_raw / (DeltaGM * N)

    Returns
    -------
    traj_norm : ndarray (total_kpts, 2) in [-0.5, 0.5]
    radii     : ndarray (n_rings,) raw radii in gradient-moment units
    """
    first_circle = info["first_circle"]
    traj_pts     = info["traj_pts"]
    delta_gm     = info["delta_gm"]
    N            = info["matrix_size"]
    n_rings      = info["n_rings"]

    radii = np.sqrt(first_circle[0] ** 2 + first_circle[1] ** 2)
    phi0  = -np.arctan2(first_circle[1], first_circle[0]) - np.pi / 2

    all_kx, all_ky = [], []
    for r in range(n_rings):
        npts = int(traj_pts[r])
        dphi = 2 * np.pi / npts
        phi  = np.arange(0, -npts, -1) * dphi
        all_kx.append(radii[r] * np.cos(phi + phi0[r]))
        all_ky.append(radii[r] * np.sin(phi + phi0[r]))

    kx_raw = np.concatenate(all_kx)
    ky_raw = np.concatenate(all_ky)

    # Physics-based normalization: k_norm = k_raw / (DeltaGM * N)
    scale = 1.0 / (delta_gm * N)
    kx_norm = kx_raw * scale
    ky_norm = ky_raw * scale

    # Stack as (total_kpts, 2) — mri-nufft convention
    traj_norm = np.stack([kx_norm, ky_norm], axis=-1).astype(np.float32)

    r_max = np.max(radii)
    k_norm_max = r_max * scale
    print(f"\n  Trajectory: {traj_norm.shape[0]} pts, {n_rings} rings")
    print(f"    r_min_raw  = {radii.min():.2f},  r_max_raw = {r_max:.2f}")
    print(f"    DeltaGM    = {delta_gm:.2f},  N = {N}")
    print(f"    scale      = 1/(DeltaGM*N) = {scale:.6e}")
    print(f"    k_norm_max = {k_norm_max:.4f}  (should be < 0.5)")
    return traj_norm, radii


# =====================================================================
# 3. Density compensation (analytical CRT — matches MATLAB)
# =====================================================================
def compute_crt_dcf(radii: np.ndarray, traj_pts: np.ndarray,
                    delta_gm: float, N: int) -> np.ndarray:
    """Analytical DCF for CRT: annular_area / n_samples_on_ring.

    Uses normalized k-space radii (matching MATLAB's ConcentricRingTrajectory_Theoretical).
    """
    n_rings = len(radii)
    scale = 1.0 / (delta_gm * N)
    norm_radii = radii * scale  # in [-0.5, 0.5] units

    sorted_idx = np.argsort(norm_radii)
    sorted_r   = norm_radii[sorted_idx]

    weights_per_ring = np.zeros(n_rings)
    max_pts = np.max(traj_pts)

    for i in range(n_rings):
        if i == 0:
            r_inner = 0.0
        else:
            r_inner = 0.5 * (sorted_r[i - 1] + sorted_r[i])
        if i == n_rings - 1:
            r_outer = sorted_r[i] + 0.5 * (sorted_r[i] - sorted_r[i - 1])
        else:
            r_outer = 0.5 * (sorted_r[i] + sorted_r[i + 1])

        annular_area = np.pi * (r_outer ** 2 - r_inner ** 2)
        orig_ring = sorted_idx[i]
        npts = int(traj_pts[orig_ring])
        # MATLAB also compensates for different #samples per ring:
        #   DCF * max(TrajPts) / TrajPts(ii)
        weights_per_ring[orig_ring] = annular_area / npts * (max_pts / npts)

    # Expand to per-sample
    dcf_list = []
    for r in range(n_rings):
        npts = int(traj_pts[r])
        dcf_list.append(np.full(npts, weights_per_ring[r]))
    dcf = np.concatenate(dcf_list)

    # Normalize so sum = total samples
    dcf *= len(dcf) / dcf.sum()
    return dcf.astype(np.float32)


# =====================================================================
# 4. FOV shift correction (phase ramp for off-isocenter slices)
# =====================================================================
def check_fov_already_corrected(info: dict, traj_norm: np.ndarray,
                                threshold: float = 0.5) -> bool:
    """Check whether k-space data already has FOV shift correction applied.

    Fits linear phase gradients to the k-space data across all coils.
    If the median offset is < threshold pixels, the data is already centered.

    Parameters
    ----------
    info       : dict with kdata, traj_pts, n_rings, n_coils, etc.
    traj_norm  : (M, 2) normalized trajectory
    threshold  : max offset (pixels) to consider already corrected

    Returns
    -------
    True if the data appears already FOV-shift corrected.
    """
    kdata    = info["kdata"]
    n_rings  = info["n_rings"]
    traj_pts = info["traj_pts"]
    kx = traj_norm[:, 0]
    ky = traj_norm[:, 1]

    all_dx, all_dy = [], []
    for coil in range(info["n_coils"]):
        parts = [kdata[:int(traj_pts[r]), 0, coil, r] for r in range(n_rings)]
        ks = np.concatenate(parts)
        mag = np.abs(ks)
        mask = mag > np.percentile(mag, 50)
        phase = np.angle(ks[mask])
        A = np.column_stack([kx[mask], ky[mask], np.ones(mask.sum())])
        W = np.diag(mag[mask])
        result = np.linalg.lstsq(W @ A, W @ phase, rcond=None)[0]
        all_dx.append(result[0] / (2 * np.pi))
        all_dy.append(result[1] / (2 * np.pi))

    dx_med = float(np.median(all_dx))
    dy_med = float(np.median(all_dy))
    already = abs(dx_med) < threshold and abs(dy_med) < threshold
    return already, dx_med, dy_med


def compute_fov_shift(info: dict, traj_norm: np.ndarray) -> np.ndarray:
    """Compute phase ramp to correct for off-isocenter FOV position.

    Follows MATLAB op_ReconstructNonCartMRData.m lines 196-228:
    1. Build rotation matrix from SliceNormalVector and InPlaneRotation
    2. Rotate LPH = [Pos_Cor, Pos_Sag, Pos_Tra] to PRS coordinates
    3. Apply phase ramp: exp(2πi * k_norm * shift_in_pixels)

    For axial slices (normal=[0,0,1], InPlaneRot=0), Rot=I so PRS=LPH.

    Parameters
    ----------
    info      : dict with pos_sag, pos_cor, fov_mm, matrix_size, etc.
    traj_norm : (total_kpts, 2) normalized trajectory in [-0.5, 0.5]

    Returns
    -------
    phase_ramp : (total_kpts,) complex phase correction
    """
    pos_cor = info["pos_cor"]   # mm
    pos_sag = info["pos_sag"]   # mm
    pos_tra = info["pos_tra"]   # mm
    snv = np.array([info["snv_y"], info["snv_x"], info["snv_z"]])
    ipr = info["in_plane_rotation"]
    fov_read  = info["fov_mm"]  # mm
    fov_phase = info["fov_mm"]  # assume square FOV
    N = info["matrix_size"]

    # LPH = [Pos_Cor, Pos_Sag, Pos_Tra]  (MATLAB convention)
    LPH = np.array([pos_cor, pos_sag, pos_tra])

    # Rotation: Normal1 -> [0,0,1]
    # Use Rodrigues' rotation formula
    normal1 = snv / (np.linalg.norm(snv) + 1e-30)
    normal2 = np.array([0.0, 0.0, 1.0])

    cross_n = np.cross(normal1, normal2)
    dot_n   = np.dot(normal1, normal2)
    if np.linalg.norm(cross_n) < 1e-10:
        Rot = np.eye(3)
    else:
        K = np.array([[0, -cross_n[2], cross_n[1]],
                      [cross_n[2], 0, -cross_n[0]],
                      [-cross_n[1], cross_n[0], 0]])
        Rot = np.eye(3) + K + K @ K / (1 + dot_n)

    # In-plane rotation
    if abs(ipr) > 1e-10:
        c, s = np.cos(ipr), np.sin(ipr)
        cross_i = np.cross(np.array([c, s, 0]), np.array([1, 0, 0]))
        dot_i   = c
        K2 = np.array([[0, -cross_i[2], cross_i[1]],
                        [cross_i[2], 0, -cross_i[0]],
                        [-cross_i[1], cross_i[0], 0]])
        Rot2 = np.eye(3) + K2 + K2 @ K2 / (1 + dot_i + 1e-30)
        Rot = Rot2 @ Rot

    PRS = Rot @ LPH
    P_shift = PRS[0]  # Phase direction (mm)
    R_shift = PRS[1]  # Read direction (mm)

    print(f"\n  FOV shift correction:")
    print(f"    PRS = [{PRS[0]:.2f}, {PRS[1]:.2f}, {PRS[2]:.2f}] mm")
    print(f"    Read  shift = {R_shift:.2f} mm = {R_shift*N/fov_read:.2f} pixels")
    print(f"    Phase shift = {P_shift:.2f} mm = {P_shift*N/fov_phase:.2f} pixels")

    kx_norm = traj_norm[:, 0]
    ky_norm = traj_norm[:, 1]

    # Shift in pixels
    R_shift_pix = R_shift * N / fov_read
    P_shift_pix = P_shift * N / fov_phase

    # MATLAB formula (op_ReconstructNonCartMRData.m, ConjSign=+1):
    #   FOVShift  = exp(-2πi * kx * R_pix)
    #   FOVShift2 = exp(+2πi * ky * P_pix)
    phase_x = -2.0 * np.pi * kx_norm * R_shift_pix
    phase_y =  2.0 * np.pi * ky_norm * P_shift_pix

    phase_ramp = np.exp(1j * (phase_x + phase_y)).astype(np.complex64)
    return phase_ramp


# =====================================================================
# 5. Reconstruct with mri-nufft
# =====================================================================
def reconstruct_adjoint(info: dict, traj_norm: np.ndarray, dcf: np.ndarray,
                        coil_idx: int = 11, recon_size: int | None = None,
                        phase_ramp: np.ndarray | None = None):
    """DCF-weighted adjoint NUFFT for each spectral point.

    Parameters
    ----------
    info       : dict from load_rearranged_mat
    traj_norm  : (total_kpts, 2) in [-0.5, 0.5]
    dcf        : (total_kpts,) density compensation
    coil_idx   : which coil to reconstruct (0-based)
    recon_size : override matrix size
    phase_ramp : (total_kpts,) FOV shift phase correction

    Returns
    -------
    recon : (n_spec, H, W) complex
    """
    kdata   = info["kdata"]       # (kpts, n_spec, n_coils, n_rings)
    n_spec  = info["n_spec"]
    n_rings = info["n_rings"]
    N       = recon_size or info["matrix_size"]

    # Flatten: (n_rings*kpts, n_spec, n_coils)
    kdata_flat = kdata.transpose(3, 0, 1, 2).reshape(-1, n_spec, kdata.shape[2])
    total_kpts = kdata_flat.shape[0]

    print(f"\n  Creating mri-nufft operator: finufft, {total_kpts} samples -> {N}x{N}")
    print(f"  Reconstructing coil {coil_idx + 1} (0-based {coil_idx})")

    # Create single-coil operator with DCF
    nufft_op = mrinufft.get_operator(
        "finufft",
        samples=traj_norm,
        shape=(N, N),
        n_coils=1,
        density=dcf,
    )

    recon = np.zeros((n_spec, N, N), dtype=np.complex64)
    t0 = time.time()

    for s in range(n_spec):
        if s % 20 == 0:
            print(f"    spectral {s}/{n_spec}")
        ks = kdata_flat[:, s, coil_idx]  # (M,) complex
        if phase_ramp is not None:
            ks = ks * phase_ramp
        img = nufft_op.adj_op(ks)        # (N, N) complex
        recon[s] = img.astype(np.complex64)

    elapsed = time.time() - t0
    print(f"  Adjoint recon done in {elapsed:.1f}s")
    return recon


def reconstruct_cg(info: dict, traj_norm: np.ndarray, dcf: np.ndarray,
                   coil_idx: int = 11, recon_size: int | None = None,
                   lambd: float = 0.01, max_iter: int = 20,
                   phase_ramp: np.ndarray | None = None):
    """CG-based Tikhonov NUFFT recon: (A^H W A + λI)x = A^H W y.

    Uses mri-nufft forward/adjoint inside a CG loop.
    """
    kdata   = info["kdata"]
    n_spec  = info["n_spec"]
    N       = recon_size or info["matrix_size"]

    kdata_flat = kdata.transpose(3, 0, 1, 2).reshape(-1, n_spec, kdata.shape[2])
    total_kpts = kdata_flat.shape[0]

    print(f"\n  CG-Tikhonov: finufft, {total_kpts} samples -> {N}x{N}")
    print(f"  lambda={lambd}, max_iter={max_iter}, coil {coil_idx + 1}")

    # Operator WITHOUT built-in DCF (we apply manually)
    nufft_op = mrinufft.get_operator(
        "finufft",
        samples=traj_norm,
        shape=(N, N),
        n_coils=1,
        density=False,
    )

    # Also create DCF-weighted adjoint operator for the RHS
    nufft_op_dcf = mrinufft.get_operator(
        "finufft",
        samples=traj_norm,
        shape=(N, N),
        n_coils=1,
        density=dcf,
    )

    recon = np.zeros((n_spec, N, N), dtype=np.complex64)
    t0 = time.time()

    for s in range(n_spec):
        if s % 20 == 0:
            print(f"    spectral {s}/{n_spec}")
        ks = kdata_flat[:, s, coil_idx]  # (M,) complex

        if phase_ramp is not None:
            ks = ks * phase_ramp
        # RHS: b = A^H W y
        b = nufft_op_dcf.adj_op(ks).ravel()

        # CG to solve (A^H W A + λI) x = b
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rsold = np.real(np.vdot(r, r))

        for it in range(max_iter):
            # A^H W A p
            Ap_k = nufft_op.op(p.reshape(N, N))    # forward: image -> kspace
            Ap_k_w = Ap_k * dcf                     # apply DCF
            AhWAp = nufft_op.adj_op(Ap_k_w).ravel() # adjoint
            AhWAp += lambd * p                       # + λI

            pAp = np.real(np.vdot(p, AhWAp))
            if pAp < 1e-30:
                break
            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * AhWAp
            rsnew = np.real(np.vdot(r, r))
            if rsnew < 1e-20:
                break
            p = r + (rsnew / (rsold + 1e-30)) * p
            rsold = rsnew

        recon[s] = x.reshape(N, N).astype(np.complex64)

    elapsed = time.time() - t0
    print(f"  CG recon done in {elapsed:.1f}s")
    return recon


# =====================================================================
# 5. All-coils adjoint (for later combination)
# =====================================================================
def reconstruct_all_coils_adjoint(info: dict, traj_norm: np.ndarray, dcf: np.ndarray,
                                  recon_size: int | None = None,
                                  phase_ramp: np.ndarray | None = None):
    """Adjoint NUFFT for ALL coils at once using multi-coil operator."""
    kdata   = info["kdata"]
    n_spec  = info["n_spec"]
    n_coils = info["n_coils"]
    N       = recon_size or info["matrix_size"]

    kdata_flat = kdata.transpose(3, 0, 1, 2).reshape(-1, n_spec, n_coils)
    total_kpts = kdata_flat.shape[0]

    print(f"\n  All-coil adjoint: finufft, {total_kpts} samples -> {N}x{N}, {n_coils} coils")

    nufft_op = mrinufft.get_operator(
        "finufft",
        samples=traj_norm,
        shape=(N, N),
        n_coils=n_coils,
        density=dcf,
    )

    recon = np.zeros((n_spec, n_coils, N, N), dtype=np.complex64)
    t0 = time.time()

    for s in range(n_spec):
        if s % 20 == 0:
            print(f"    spectral {s}/{n_spec}")
        ks = kdata_flat[:, s, :].T  # (n_coils, M) complex
        if phase_ramp is not None:
            ks = ks * phase_ramp[np.newaxis, :]  # broadcast (n_coils, M)
        img = nufft_op.adj_op(ks)   # (n_coils, N, N) complex
        recon[s] = img.astype(np.complex64)

    elapsed = time.time() - t0
    print(f"  All-coil adjoint done in {elapsed:.1f}s")
    return recon


# =====================================================================
# 6. WSVD coil combination
# =====================================================================
def coil_combine_wsvd(recon_all: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """SVD-based coil combination.

    Parameters
    ----------
    recon_all : (n_spec, n_coils, H, W) complex
        Reconstructed images for all coils.

    Returns
    -------
    combined : (n_spec, H, W) complex
        Combined FID images.
    """
    T, C, H, W = recon_all.shape
    out = np.zeros((T, H, W), dtype=np.complex64)
    for ix in range(H):
        for iy in range(W):
            a = recon_all[:, :, ix, iy].T   # (C, T)
            if np.sum(np.abs(a) ** 2) < eps:
                continue
            u, s, vh = np.linalg.svd(a, full_matrices=False)
            fid = np.conj(u[:, 0]) @ a      # (T,)
            fid *= np.exp(-1j * np.angle(fid[0]))  # zero-order phase
            out[:, ix, iy] = fid
    return out


# =====================================================================
# 7. Plotting
# =====================================================================
def plot_results(recon, info, traj_norm, coil_idx, method, out_prefix, label=None):
    """4-panel figure: trajectory, spatial map, spectrum (Hz), spectrum (ppm)."""
    n_spec   = info["n_spec"]
    freq_hz  = info["freq_hz"]
    ppm_axis = info["ppm_axis"]
    N        = recon.shape[1]
    if label is None:
        label = f"Coil {coil_idx+1}"

    # Spatial projection: use first N_FID points where signal dominates.
    # Using all spectral points is noise-dominated and gives misleading maps.
    n_fid = min(20, n_spec)   # first 20 FID points
    proj = np.mean(np.abs(recon[:n_fid]), axis=0)
    max_idx = np.unravel_index(np.argmax(proj), proj.shape)
    max_x, max_y = max_idx
    print(f"  Max-signal voxel: [{max_x}, {max_y}]  (intensity={proj[max_x, max_y]:.2e})")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Trajectory scatter
    axes[0, 0].scatter(traj_norm[:, 0], traj_norm[:, 1], s=0.5, alpha=0.5)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title(f"CRT trajectory ({traj_norm.shape[0]} pts, {info['n_rings']} rings)")
    axes[0, 0].set_xlabel("kx (normalized)"); axes[0, 0].set_ylabel("ky (normalized)")
    axes[0, 0].axhline(0, color='gray', lw=0.5); axes[0, 0].axvline(0, color='gray', lw=0.5)
    # Show ±0.5 bounds
    for v in [-0.5, 0.5]:
        axes[0, 0].axhline(v, color='r', lw=0.5, ls='--', alpha=0.5)
        axes[0, 0].axvline(v, color='r', lw=0.5, ls='--', alpha=0.5)

    # (b) Spatial projection map
    im = axes[0, 1].imshow(proj.T, cmap="hot", origin="lower",
                           extent=[0, N, 0, N])
    axes[0, 1].plot(max_x + 0.5, max_y + 0.5, "c+", ms=14, mew=2)
    axes[0, 1].set_title(f"{label} — {method} — projection map")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
    plt.colorbar(im, ax=axes[0, 1])

    # (c) Max-voxel spectrum (Hz)
    spec_max = np.fft.fftshift(np.fft.fft(recon[:, max_x, max_y]))
    axes[1, 0].plot(freq_hz, np.abs(spec_max), "b-", lw=0.8)
    axes[1, 0].set_title(f"Max voxel [{max_x},{max_y}] — Hz")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("|Spectrum|")
    axes[1, 0].set_xlim(freq_hz[-1], freq_hz[0])

    # (d) Max-voxel spectrum (ppm)
    axes[1, 1].plot(ppm_axis, np.abs(spec_max), "r-", lw=0.8)
    axes[1, 1].set_title(f"Max voxel [{max_x},{max_y}] — ppm  (f0={info['larmor_freq_mhz']:.3f} MHz)")
    axes[1, 1].set_xlabel("Chemical shift (ppm)")
    axes[1, 1].set_ylabel("|Spectrum|")
    axes[1, 1].set_xlim(ppm_axis[-1], ppm_axis[0])

    plt.suptitle(
        f"31P CRT mri-nufft — {label} — {method} | BW={info['spectral_bw_hz']:.0f} Hz | "
        f"FOV={info['fov_mm']:.0f} mm | {N}x{N} | {info['n_rings']} rings\n"
        f"γ/2π={info['gamma_2pi']:.0f} Hz/T | ΔGM={info['delta_gm']:.2f} | "
        f"k_norm ∈ [{traj_norm.min():.4f}, {traj_norm.max():.4f}]",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=200)
    print(f"  Saved {out_prefix}.png")
    return proj, max_idx


# =====================================================================
# 8. Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="31P CRT recon with mri-nufft")
    parser.add_argument("--mat", required=True, help="Path to rearranged .mat file")
    parser.add_argument("--out", default="crt_mrinufft_recon", help="Output prefix")
    parser.add_argument("--method", default="adjoint", choices=["adjoint", "cg"])
    parser.add_argument("--lambd", type=float, default=0.01, help="Tikhonov lambda (for cg)")
    parser.add_argument("--max-iter", type=int, default=20, help="CG iterations")
    parser.add_argument("--coil", type=int, default=12, help="Coil number (1-based)")
    parser.add_argument("--recon-size", type=int, default=0, help="Override matrix size")
    parser.add_argument("--all-coils", action="store_true", help="Also recon all coils")
    parser.add_argument("--wsvd", action="store_true", help="WSVD coil combination (recons all coils)")
    parser.add_argument("--dcf-method", default="analytical",
                        choices=["analytical", "voronoi", "pipe"],
                        help="DCF computation method")
    parser.add_argument("--no-fov-shift", action="store_true",
                        help="Skip FOV shift correction (auto-detected by default)")
    parser.add_argument("--force-fov-shift", action="store_true",
                        help="Force FOV shift correction even if data appears pre-corrected")
    args = parser.parse_args()

    coil_idx = args.coil - 1  # to 0-based
    recon_size = args.recon_size if args.recon_size > 0 else None

    # ------------------------------------------------------------------
    # 1) Load data
    # ------------------------------------------------------------------
    info = load_rearranged_mat(args.mat)

    # ------------------------------------------------------------------
    # 2) Build trajectory in [-0.5, 0.5]
    # ------------------------------------------------------------------
    traj_norm, radii = build_crt_trajectory(info)

    # ------------------------------------------------------------------
    # 3) Density compensation
    # ------------------------------------------------------------------
    N = recon_size or info["matrix_size"]

    if args.dcf_method == "analytical":
        dcf = compute_crt_dcf(radii, info["traj_pts"], info["delta_gm"], N)
        print(f"  DCF (analytical): range [{dcf.min():.4f}, {dcf.max():.4f}], mean={dcf.mean():.4f}")
    elif args.dcf_method == "voronoi":
        dcf = mrinufft.voronoi(traj_norm)
        dcf = dcf.astype(np.float32)
        print(f"  DCF (voronoi): range [{dcf.min():.4f}, {dcf.max():.4f}], mean={dcf.mean():.4f}")
    elif args.dcf_method == "pipe":
        dcf = mrinufft.pipe(traj_norm, shape=(N, N))
        dcf = dcf.astype(np.float32)
        print(f"  DCF (pipe): range [{dcf.min():.4f}, {dcf.max():.4f}], mean={dcf.mean():.4f}")

    # ------------------------------------------------------------------
    # 3b) FOV shift correction (phase ramp for off-isocenter slices)
    # ------------------------------------------------------------------
    if args.no_fov_shift:
        phase_ramp = None
        print("\n  FOV shift: SKIPPED (--no-fov-shift)")
    else:
        # Auto-detect if data is already FOV-corrected
        already, dx_meas, dy_meas = check_fov_already_corrected(info, traj_norm)
        if already and not args.force_fov_shift:
            phase_ramp = None
            print(f"\n  FOV shift: SKIPPED (data already corrected, "
                  f"measured offset = [{dx_meas:+.3f}, {dy_meas:+.3f}] pix)")
        else:
            phase_ramp = compute_fov_shift(info, traj_norm)
            if already:
                print("    WARNING: data appears pre-corrected but --force-fov-shift is set")

    # ------------------------------------------------------------------
    # 4) Reconstruct
    # ------------------------------------------------------------------
    if args.wsvd:
        # WSVD needs all coils
        print("\n  === WSVD coil combination ===")
        recon_all = reconstruct_all_coils_adjoint(info, traj_norm, dcf,
                                                  recon_size=recon_size,
                                                  phase_ramp=phase_ramp)
        print("  Running WSVD coil combination...")
        t0 = time.time()
        recon = coil_combine_wsvd(recon_all)  # (n_spec, N, N)
        print(f"  WSVD done in {time.time()-t0:.1f}s")
        coil_label = "WSVD"
    else:
        if args.method == "adjoint":
            recon = reconstruct_adjoint(info, traj_norm, dcf,
                                        coil_idx=coil_idx, recon_size=recon_size,
                                        phase_ramp=phase_ramp)
        elif args.method == "cg":
            recon = reconstruct_cg(info, traj_norm, dcf,
                                   coil_idx=coil_idx, recon_size=recon_size,
                                   lambd=args.lambd, max_iter=args.max_iter,
                                   phase_ramp=phase_ramp)
        coil_label = None  # uses default "Coil X"

    # ------------------------------------------------------------------
    # 5) Plot and save
    # ------------------------------------------------------------------
    proj, max_idx = plot_results(recon, info, traj_norm, coil_idx, args.method,
                                 args.out, label=coil_label)

    save_dict = dict(
        recon=recon,
        traj_norm=traj_norm,
        dcf=dcf,
        freq_hz=info["freq_hz"],
        ppm_axis=info["ppm_axis"],
        proj=proj,
        max_voxel=np.array(max_idx),
        matrix_size=N,
        coil_index=coil_idx,
        method=args.method,
        lambd=args.lambd,
        delta_gm=info["delta_gm"],
        gamma_2pi=info["gamma_2pi"],
        fov_mm=info["fov_mm"],
        larmor_freq_hz=info["larmor_freq_hz"],
        spectral_bw_hz=info["spectral_bw_hz"],
        wsvd=args.wsvd,
    )
    if args.wsvd:
        save_dict["recon_all_coils"] = recon_all
    np.savez(f"{args.out}.npz", **save_dict)
    print(f"  Saved {args.out}.npz")

    # ------------------------------------------------------------------
    # 6) Optional: all-coil recon (without WSVD)
    # ------------------------------------------------------------------
    if args.all_coils and not args.wsvd:
        recon_all = reconstruct_all_coils_adjoint(info, traj_norm, dcf,
                                                  recon_size=recon_size,
                                                  phase_ramp=phase_ramp)
        np.savez(f"{args.out}_allcoils.npz", recon_all=recon_all)
        print(f"  Saved {args.out}_allcoils.npz")

    print("\nDone!")


if __name__ == "__main__":
    main()

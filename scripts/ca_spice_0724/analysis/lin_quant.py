#!/usr/bin/env python3
"""Noise-unbiased quantification and noise-corrected display for NUFFT and CA-SPICE (24-07 bouts).
Both reconstructions are linear in k-space (lin_ops.npz: NUFFT X = PN K, CA-SPICE X = (PS K V) V^H), so for any voxel / ROI weighting w the noise
variance of its spectrum at every frequency is exact:  var(f) = sig2 * ||w P||^2 * sum_j |FFT(apo * b_j)(f)|^2 / n_avg   (b_j: spectral basis, NUFFT = identity).
Quantification: first-order phase (delay tau, per bout + method, from the GM ROI) + zero order per voxel (rest PCr), then REAL-part band integrals
(PCr |ppm|<1, Pi + Pi2 3.8-5.4, gamma/alpha/beta-ATP) -> linear, zero-mean noise, no magnitude noise floor.
Display: power-subtracted magnitude sqrt(max(|S|^2 - var(f), 0)).
Run as a script: precomputes per-voxel band integrals for all bouts -> output_study_mid<ds>/spice_final/linq.npz, and prints rest Pi/PCr checks."""
import os, json, numpy as np
ROOT = "/home/zsun/SPICE_Margarita"
BANDS = {"PCr": (-1.0, 1.0), "Pi": (3.8, 5.4), "gATP": (-3.6, -1.4), "aATP": (-8.7, -6.4), "bATP": (-17.4, -14.6)}
APO_HZ = 8.0                                                                                       # exponential line broadening used for quantification and display

def load(B):
    O = f"{ROOT}/output_study_mid{B}"; L = dict(np.load(f"{O}/spice_final/lin_ops.npz"))
    X = {"NUFFT": np.load(f"{O}/adj_fid.npy"), "CA-SPICE": np.load(f"{O}/spice_final/fid_spice.npy")}
    n = X["NUFFT"].shape[-1]; L["n"] = n; L["t"] = np.arange(n) * float(L["dw"]); L["apo"] = np.exp(-np.pi * APO_HZ * L["t"])
    L["ppm_raw"] = -np.fft.fftshift(np.fft.fftfreq(n, float(L["dw"]))) / float(L["f0"])
    return L, X

def spec(X, apo):
    return np.fft.fftshift(np.fft.fft(np.conj(X * apo), axis=-1), axes=-1)

def basis_spectra(L, meth, apo=None):
    """FFT of apo * basis vectors (rows): CA-SPICE = the rank-12 subspace V, NUFFT = identity (all time points)."""
    apo = L["apo"] if apo is None else apo
    if meth == "CA-SPICE": return spec(np.conj(L["V"].T).astype(np.complex128), apo)                # X = z V^H -> spec(X) = sum_j conj(z_j) FFT(apo V_j): spec(conj V_j)
    return np.fft.fftshift(np.fft.fft(np.diag(apo).astype(np.complex128), axis=-1), axes=-1)

def op(L, meth): return L["PS"] if meth == "CA-SPICE" else L["PN"]

def noise_var_spec(L, meth, w, n_avg=1, BS=None, apo=None):
    """Variance of sum_v w_v S_v(f) per frequency (complex variance E|n|^2). w: (N^2,) complex weights; apo: apodisation used for the spectrum."""
    BS = basis_spectra(L, meth, apo) if BS is None else BS; a = np.asarray(w) @ op(L, meth)
    return float(L["sig2"]) * np.sum(np.abs(a)**2) * np.sum(np.abs(BS)**2, 0) / n_avg

def phase_model(L, S_gm_rest, S_gm_all, Q):
    """Delay tau (s) maximising absorption of PCr, Pi, gamma/alpha/beta-ATP in the GM ROI (rest + all reps)."""
    f_hz = Q * float(L["f0"]); pk = [(0.0, 0.35), (4.9, 0.5), (-2.5, 0.4), (-7.6, 0.4), (-16.2, 0.5)]; best = None
    for tau in np.arange(-2.5e-3, -0.4999e-3, 0.02e-3):                                              # around the known ~1.5 ms acquisition delay (wide grids alias on alpha/beta-ATP)
        sc = 0.0
        for Sx in (S_gm_rest, S_gm_all):
            Sc = Sx * np.exp(-2j * np.pi * f_hz * tau); Sc = Sc * np.exp(-1j * np.angle(Sc[np.abs(Q) < 0.3].sum()))
            sc += sum(np.real(Sc[np.abs(Q - p) < h]).sum() / (np.abs(Sc[np.abs(Q - p) < h]).sum() + 1e-30) for p, h in pk)
        if best is None or sc > best[0]: best = (sc, tau)
    return best[1]

def quantify(B, meth, L=None, X=None):
    """Per-voxel phased spectra -> real band integrals per rep. Returns dict with tau, Q, phase factors, band integrals (NR, N^2, bands)."""
    if L is None: L, X = load(B)
    R0, R1 = map(int, L["rest"]); legv = L["legv"].astype(bool); vGM = L["vGM"].astype(bool)
    S = spec(X[meth], L["apo"])                                                                      # (NR, N^2, n)
    Sr = S[R0:R1].mean(0); k0 = np.flatnonzero(np.abs(L["ppm_raw"]) < 1.5)[np.argmax(np.abs(Sr[vGM]).sum(0)[np.abs(L["ppm_raw"]) < 1.5])]
    Q = L["ppm_raw"] - L["ppm_raw"][k0]
    c = Sr[:, k0]; wg = np.where(vGM, np.exp(-1j * np.angle(c)), 0)
    tau = phase_model(L, (Sr * wg[:, None]).sum(0), (S.mean(0) * wg[:, None]).sum(0), Q)
    first = np.exp(-2j * np.pi * Q * float(L["f0"]) * tau)                                            # first-order correction (frequency axis)
    Srf = Sr * first; zero = np.exp(-1j * np.angle(Srf[:, np.abs(Q) < 0.3].sum(1)))                  # zero order per voxel at PCr
    Sp = S * first[None, None, :] * zero[None, :, None]
    bands = {k: (Q >= lo) & (Q <= hi) for k, (lo, hi) in BANDS.items()}
    I = np.stack([np.real(Sp[..., b]).sum(-1) for b in bands.values()], -1)                         # (NR, N^2, nb)
    return dict(tau=tau, Q=Q, first=first, zero=zero, I=I, bands=list(BANDS), Sp=Sp)

def roi_curves(q, vox, R0, R1, EX=None):
    """ROI = rest-PCr-weighted sum of the phased voxel spectra (as before: weights |c_v|). Returns PCr, Pi (Pi+Pi2), ATPs per rep, rel. to rest PCr."""
    I = q["I"][:, vox]; w = np.maximum(I[R0:R1, :, 0].mean(0), 0); w = w / max(w.sum(), 1e-30)
    tot = (I * w[None, :, None]).sum(1); r0 = tot[R0:R1, 0].mean()
    return {k: tot[:, i] / r0 for i, k in enumerate(q["bands"])}, w

def load_ops(B):
    """lin_ops only (no FIDs): for figure scripts that already hold the spectra."""
    L = dict(np.load(f"{ROOT}/output_study_mid{B}/spice_final/lin_ops.npz")); return L

def denoised_mag(Sp_avg, var_f):
    """Power-subtracted magnitude spectrum: sqrt(max(|S|^2 - var, 0))."""
    return np.sqrt(np.maximum(np.abs(Sp_avg)**2 - var_f, 0))

if __name__ == "__main__":
    for B in ("283", "279", "281", "277"):
        L, X = load(B); R0, R1 = map(int, L["rest"]); E0, E1 = map(int, L["ex"]); REP = float(L["REP"]); NR = int(L["NR"]); EXw = slice(E1 - int(round(96 / REP)), E1)
        out = {}
        for meth in ("NUFFT", "CA-SPICE"):
            q = quantify(B, meth, L, X); out[meth] = q
            np.savez(f"{ROOT}/output_study_mid{B}/spice_final/linq_{meth}.npz", tau=q["tau"], Q=q["Q"], zero=q["zero"], first=q["first"], I=q["I"].astype(np.float32), bands=np.array(q["bands"]))
        msg = []
        for mus, msk in (("GM", L["vGM"]), ("soleus", L["vSOL"])):
            vox = np.flatnonzero(msk.astype(bool) & L["legv"].astype(bool))
            for meth in ("NUFFT", "CA-SPICE"):
                cv, _ = roi_curves(out[meth], vox, R0, R1)
                msg.append(f"{mus} {meth}: rest Pi/PCr {cv['Pi'][R0:R1].mean():.2f}, depl {100 * (1 - cv['PCr'][EXw].mean()):.0f}%, Pi rise {100 * (cv['Pi'][EXw].mean() - cv['Pi'][R0:R1].mean()):+.0f}, rest bATP/PCr {cv['bATP'][R0:R1].mean():.2f}")
        print(f"MID00{B}: tau NUFFT {1e3 * out['NUFFT']['tau']:+.2f} ms, CA-SPICE {1e3 * out['CA-SPICE']['tau']:+.2f} ms\n   " + "\n   ".join(msg))

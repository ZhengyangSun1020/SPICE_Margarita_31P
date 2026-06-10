"""
Spectral fitting utilities: voxel-by-voxel nonlinear fitting, Monte Carlo basis fitting.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional

from fsl_mrs.utils.misc import FIDToSpec, SpecToFID

FLOAT_D_TYPE = np.float32


def fit_mrs_spectrum_lstsq_batch_vbv(bm_FIDs, time_axis, spectra, initial_guesses):
    """
    Batched nonlinear fitting for multiple voxels (2-metabolite model).

    Inputs
    ------
    bm_FIDs : list of basis FIDs
    time_axis : 1D array of length T
    spectra : 2D array (N_voxel, T)
    initial_guesses : 1D array [lw0, c0, lw1, c1]

    Returns
    -------
    popt : flattened [lw1_all, cm1_all, lw2_all, cm2_all]
    pcov : None
    """
    N_voxel = spectra.shape[0]
    N_SEQ_POINTS = len(time_axis)

    lw1_list = []
    cm1_list = []
    lw2_list = []
    cm2_list = []
    failed_voxels = 0

    def fit_single_voxel(data, basis_fid, timeaxis, initial_guesses):

        def fwd_model(x, bases):
            fid = np.zeros((N_SEQ_POINTS), dtype=complex)
            concs = x[1::2]
            linewidths = x[0::2]
            for cm, lw, basis in zip(concs, linewidths, bases):
                fid += cm * basis * np.exp(-lw * 2 * np.pi * timeaxis)
            return fid

        def jac(x, bases):
            grad = []
            concs = x[1::2]
            linewidths = x[0::2]
            for cm, lw, basis in zip(concs, linewidths, bases):
                grad.append(-cm * 2 * np.pi * timeaxis * basis * np.exp(-lw * 2 * np.pi * timeaxis))
                grad.append(basis * np.exp(-lw * 2 * np.pi * timeaxis))
            return np.asarray(grad)

        def loss(x):
            return np.linalg.norm(data - fwd_model(x, basis_fid))

        def grad(x):
            S = fwd_model(x, basis_fid)
            dS = jac(x, basis_fid)
            out = np.real(
                np.sum(
                    S * np.conj(dS) + np.conj(S) * dS - np.conj(data) * dS - data * np.conj(dS),
                    axis=1))
            return out

        x0 = initial_guesses
        bounds = ((0, None), (0, None), (0, None), (0, None))
        xout = minimize(loss, x0, bounds=bounds, jac=grad)
        print(xout)
        return xout.x

    for v in range(N_voxel):
        try:
            spectrum = spectra[v, :]
            popt = fit_single_voxel(spectrum, bm_FIDs, time_axis, initial_guesses)
            lw1_list.append(popt[0])
            cm1_list.append(popt[1])
            lw2_list.append(popt[2])
            cm2_list.append(popt[3])
        except Exception as e:
            print(f"⚠️ Voxel {v} fitting failed:", e)
            failed_voxels += 1

    if failed_voxels > 0:
        print(f"⚠️ {failed_voxels} voxels failed during fitting")

    popt = np.concatenate([lw1_list, cm1_list, lw2_list, cm2_list])
    return popt, None


def fit_mrs_spectrum_lstsq_batch_vbv_arbmat(bm_FIDs, time_axis, spectra, initial_guesses):
    """
    Batched fitting for multiple voxels using nonlinear least-squares (arbitrary metabolites).

    Inputs
    ------
    bm_FIDs : list of 1D arrays (len T), one per metabolite
    time_axis : 1D array of length T
    spectra : 2D array (N_voxel, T)
    initial_guesses : 1D array of length 2*num_metab [lw0, c0, lw1, c1, ...]

    Returns
    -------
    popt_all : np.ndarray, shape (N_voxel, 2*num_metab)
    pcov : None
    """
    N_voxel = spectra.shape[0]
    num_metab = len(bm_FIDs)
    n_params = 2 * num_metab

    popt_all = np.zeros((N_voxel, n_params))
    failed_voxels = 0

    def fit_single_voxel(data, bases, timeaxis, x0):
        def fwd_model(x, bases):
            fid = np.zeros(len(timeaxis), dtype=complex)
            concs = x[1::2]
            linewidths = x[0::2]
            for cm, lw, basis in zip(concs, linewidths, bases):
                fid += cm * basis * np.exp(-lw * 2 * np.pi * timeaxis)
            return fid

        def jac(x, bases):
            grad = []
            concs = x[1::2]
            linewidths = x[0::2]
            for cm, lw, basis in zip(concs, linewidths, bases):
                grad.append(-cm * 2 * np.pi * timeaxis * basis * np.exp(-lw * 2 * np.pi * timeaxis))
                grad.append(basis * np.exp(-lw * 2 * np.pi * timeaxis))
            return np.vstack(grad)

        def loss(x):
            return np.linalg.norm(data - fwd_model(x, bases))

        def grad(x):
            S = fwd_model(x, bases)
            dS = jac(x, bases)
            return np.real(np.sum((S - data) * np.conj(dS), axis=1) * 2)

        bounds = [(0, None)] * n_params
        xout = minimize(loss, x0, bounds=bounds, jac=grad)
        return xout.x

    for v in range(N_voxel):
        try:
            spectrum = spectra[v, :]
            popt = fit_single_voxel(spectrum, bm_FIDs, time_axis, initial_guesses)
            popt_all[v, :] = popt
        except Exception as e:
            print(f"⚠️ Voxel {v} fitting failed:", e)
            failed_voxels += 1

    if failed_voxels > 0:
        print(f"⚠️ {failed_voxels} voxels failed during fitting")

    return popt_all, None


def mc_basis(spice_mc_U, bm_FIDs, taxis, N_VOXEL):
    """Monte Carlo basis fitting for 2-metabolite model.

    N_VOXEL must be passed explicitly.
    """
    iterations = len(spice_mc_U) if len(spice_mc_U) <= 5000 else 5000
    output_cm1 = []
    output_cm2 = []
    output_LW1 = []
    output_LW2 = []
    n_skipped = 0

    for i in range(iterations):
        est_U = spice_mc_U[i, :, :]
        spectrum = est_U
        initial_guesses = np.array([8, 2, 8, 2], dtype=FLOAT_D_TYPE)

        popt, pcov = fit_mrs_spectrum_lstsq_batch_vbv(bm_FIDs, taxis, spectrum, initial_guesses)
        if popt is None:
            print("⚠️ 拟合失败，当前 spectrum 被跳过")
            n_skipped += 1
            continue
        lw1 = popt[0:N_VOXEL]
        fit1 = popt[N_VOXEL:2 * N_VOXEL]
        lw2 = popt[2 * N_VOXEL:3 * N_VOXEL]
        fit2 = popt[3 * N_VOXEL:4 * N_VOXEL]
        output_cm1.append(fit1)
        output_cm2.append(fit2)
        output_LW1.append(lw1)
        output_LW2.append(lw2)

        if i % 50 == 0:
            print(f"Iteration {i} completed.")

    print(f"✅ 共跳过 {n_skipped} 个 spectrum")
    return np.asarray(output_cm1), np.asarray(output_cm2), np.asarray(output_LW1), np.asarray(output_LW2)


def mc_basis_arbmat(spice_mc_U, bm_FIDs, taxis, initial_guesses=None):
    """
    Monte Carlo basis fitting for arbitrary number of metabolites.

    Parameters
    ----------
    spice_mc_U : np.ndarray shape (N_iter, N_voxel, T)
    bm_FIDs : list of arrays, one per metabolite
    taxis : time axis
    initial_guesses : [lw0, c0, lw1, c1, ...] or None (defaults to [8,2]*num_metab)

    Returns
    -------
    output_concs : np.ndarray shape (N_iter_eff, N_voxel, num_metab)
    output_lws : np.ndarray shape (N_iter_eff, N_voxel, num_metab)
    """
    iterations = len(spice_mc_U) if len(spice_mc_U) <= 5000 else 5000
    num_metab = len(bm_FIDs)

    if initial_guesses is None:
        initial_guesses = [val for _ in range(num_metab) for val in (8, 2)]
    initial_guesses = np.array(initial_guesses, dtype=FLOAT_D_TYPE)

    output_concs = []
    output_lws = []
    n_skipped = 0

    for i in range(iterations):
        spectrum = spice_mc_U[i, :, :]

        popt, _ = fit_mrs_spectrum_lstsq_batch_vbv_arbmat(bm_FIDs, taxis, spectrum, initial_guesses)

        if popt is None:
            print("⚠️ 拟合失败，当前 spectrum 被跳过")
            n_skipped += 1
            continue

        lw_all = popt[:, 0::2]
        conc_all = popt[:, 1::2]

        output_lws.append(lw_all)
        output_concs.append(conc_all)

        if i % 50 == 0:
            print(f"Iteration {i} completed.")

    print(f"✅ 共跳过 {n_skipped} 个 spectrum")
    return np.asarray(output_concs), np.asarray(output_lws)

"""
Uncertainty quantification: covariance estimation, Laplacian sampling, MC utilities.
"""

import numpy as np
from typing import Union
from scipy.sparse.linalg import LinearOperator as SciLin

from fsl_mrs.utils.plotting import FID2Spec

D_TYPE = np.complex64


def calc_Covariance_spat(NOISE_SD: float,
                         lamda_1: float,
                         W_edge: np.ndarray,
                         K_POINTS: int,
                         V: np.ndarray,
                         freq: int):
    """Compute spatial covariance at a single frequency index."""
    A = Calc_Uncert_Amat(NOISE_SD, lamda_1, W_edge, K_POINTS)
    w = Select_freq_w(freq, V)
    Cov = w * np.linalg.inv(A)
    return Cov


def calc_Covariance_spat_overall(
        NOISE_SD: float,
        lamda_1: float,
        W_edge: np.ndarray,
        N_VOXEL: int,
        V: np.ndarray,
        F_OP: Union[np.ndarray, SciLin],
        B0_mat: Union[np.ndarray, SciLin],
        N_rank: int,
):
    """Compute overall spatial covariance (summed over all frequencies)."""
    A = Calc_Uncert_Amat(NOISE_SD, lamda_1, W_edge, N_VOXEL, F_OP, B0_mat, V, N_rank)

    def unmasked_w(V):
        Vh = V.conj().T
        Vh_Spec = []
        for i in range(Vh.shape[0]):
            Vh_Spec.append(np.array(FID2Spec(Vh[i, :])))
        Vh_Spec = np.vstack(Vh_Spec)
        square_sum = np.sum(Vh_Spec[:] * np.conjugate(Vh_Spec[:]))
        return square_sum

    w = unmasked_w(V)
    Cov = np.linalg.inv(A.astype(D_TYPE))
    return Cov


def precompute_cholesky(cov):
    """Compute Cholesky factorization of covariance matrix."""
    return np.linalg.cholesky(cov)


def complex_multivariate_normal(mean, L, n_samples):
    """Sample complex multivariate normal using precomputed Cholesky factor L."""
    n = mean.shape[0]
    w = (np.random.normal(0, 1 / np.sqrt(2), size=(n_samples, n))
         + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(n_samples, n)))
    samples = w @ L.T + mean
    return samples


def calc_std_uncert(Cov):
    """Compute standard uncertainty from diagonal of covariance matrix."""
    std_uncert = np.sqrt(np.abs(np.diag(Cov)))
    return std_uncert


def Create_laplacian_samples(spice_est: np.ndarray,
                              cov_overall: np.ndarray,
                              n_samples: int) -> np.ndarray:
    """Generate MC samples using the Laplacian approximation covariance.

    Args:
        spice_est: estimated KI-space from a single run of SPICE
        cov_overall: Laplacian method covariance
        n_samples: number of samples

    Returns:
        np.ndarray: generated data for MC uncertainty
    """
    n_dim, n_channels = spice_est.shape

    samples = np.zeros((n_samples, n_dim, n_channels), dtype=D_TYPE)
    cov_cholesky = precompute_cholesky(cov_overall)

    for i in range(n_channels):
        mean_vec = spice_est[:, i]
        samples[:, :, i] = complex_multivariate_normal(mean_vec, cov_cholesky, n_samples)
    return samples

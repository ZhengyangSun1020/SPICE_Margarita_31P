"""
Signal generation utilities: FID synthesis, phantom construction, noise.
"""

import numpy as np


def gen_voxel_signal(
          lb: list[float],
          concs: list[float],
          shifts: list[float],
          bm: list[np.ndarray],
          b0: list[float],
          taxis: np.ndarray) -> np.ndarray:
    """Generate the FID signal for a voxel as a weighted sum of basis sets.

    :param lb: Line broadening in units of hertz
    :type lb: list[float]
    :param concs: Concentrations of each peak
    :type concs: list[float]
    :param shifts: Chemical shift of each peak
    :type shifts: list[float]
    :return: Summed FID of all peaks
    :rtype: np.ndarray
    """
    try:
        assert len({len(x) for x in (lb, concs, shifts)}) == 1
    except AssertionError as exc:
            print('The length of lb, concs, and shifts must match!')
            print(f'Currently they are {len(lb)}, {len(concs)}, {len(shifts)}')
            raise exc

    fids = []
    for shift, linews, concentr, basis_fid in zip(shifts, lb, concs, bm):
        broadening = np.exp(-linews * 2 * np.pi * taxis)
        shifting = np.exp(-1j * 2 * np.pi * b0 * taxis)
        fid = (basis_fid * broadening * shifting).T
        fids.append(concentr * fid)
    return np.sum(fids, axis=0)


def make_2d_spectral_phantom_arbmet(GT_conc_map: tuple[np.ndarray],
                                    GT_B0_map: np.ndarray,
                                    peak_cs: list[float],
                                    lw: list[float],
                                    bm: list[float],
                                    taxis: np.ndarray) -> np.ndarray:
    """
    Construct the full 2D spatial spectral phantom with arbitrary number of metabolites.

    :param GT_conc_map: tuple of 2D arrays (one per metabolite), shape (num_metab, height, width)
    :param peak_cs: Chemical shift values of peaks
    :param lw: Linewidth values
    :param bm: Baseline model coefficients
    :param taxis: Time axis
    :return: 3D array of FIDs
    :rtype: np.ndarray of shape (height, width, timepoints)
    """
    num_metab = len(GT_conc_map)
    height, width = GT_conc_map[0].shape
    fids = np.zeros((height, width, len(taxis)), dtype=complex)

    for i in range(height):
        for j in range(width):
            amps = tuple(GT_conc_map[m][i, j] for m in range(num_metab))
            b0 = GT_B0_map[i, j]
            fid = gen_voxel_signal(lw, amps, peak_cs, bm, b0, taxis)
            fids[i, j, :] = fid

    return fids


def make_2d_spectral_phantom(amp1: list[list[float]],
                             amp2: list[list[float]],
                             peak_cs: list[float],
                             lw: list[float],
                             bm: list[float],
                             taxis: np.ndarray) -> np.ndarray:
    """
    Construct the full 2D spatial phantom.

    :param amp1: 2D list of amplitudes for metabolite 1
    :param amp2: 2D list of amplitudes for metabolite 2
    :param peak_cs: Chemical shift values of peaks
    :param lw: Linewidth values
    :param bm: Baseline model coefficients
    :param taxis: Time axis
    :return: 2D Stack of FIDs
    :rtype: np.ndarray of shape (height, width, timepoints)
    """
    height = len(amp1)
    width = len(amp1[0])
    fids = []

    for i in range(height):
        row = []
        for j in range(width):
            fid = gen_voxel_signal(lw, (amp1[i][j], amp2[i][j]), peak_cs, bm, taxis)
            row.append(fid)
        fids.append(row)

    return np.array(fids)


def add_noise2kt(
    kspace: np.ndarray,
    rng: np.random.Generator,
    noise_SD: float) -> np.ndarray:
    """Adds complex gaussian noise to kt space model

    :param kspace: Noiseless kt-space
    :type kspace: np.ndarray
    :param rng: rng object
    :type rng: np.random.Generator
    :param noise_SD: Noise standard deviation
    :type noise_SD: float
    :return: Noisy kt-space data
    :rtype: np.ndarray
    """
    return kspace + (
        (noise_SD / np.sqrt(2)) * rng.standard_normal(size=kspace.shape)
        + (noise_SD / np.sqrt(2)) * 1j * rng.standard_normal(size=kspace.shape)
    )

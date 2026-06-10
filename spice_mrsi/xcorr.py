from fsl_mrs.utils.preproc import freqshift, pad, apodize, applyPhase
from scipy.signal import correlate
from fsl_mrs.utils.preproc.shifting import freqshift_array
from fsl_mrs.utils.preproc.filtering import calc_aprox_t2decay
from fsl_mrs.utils.misc import FIDToSpec
from fsl_mrs.core.nifti_mrs import NIFTI_MRS
from fsl_mrs.core.basis import Basis
import numpy as np

from fsl.data.image import Image

def xcorr_align_complex(
        fids_in: np.ndarray,
        dwelltime: float,
        target: np.ndarray | None = None,
        zpad_factor: int = 1,
        apodize_hz: float = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align FIDs using cross correlation of magnitude spectrum

    By default aligns to the mean of all FIDs. Optionally pass a target

    :param fids_in: Array of FIDs, transients x timedomain
    :type fids_in: numpy.ndarray
    :param dwelltime: spectral dwell time (1/bandwidth) in s.
    :type dwelltime: float
    :param target: Alignment target FID, defaults to None. Zero-pad will be applied to target
    :type target: np.ndarray | None, optional
    :param zpad_factor: Zeropadding applied to fid before xcorrelation, defaults to 1, 0 disables
    :type zpad_factor: int
    :param apodize_hz: Apodization to apply to FIDs (not target), defaults to 0
    :type apodize_hz: float, optional
    :return: Returns shifted FIDs and the shift in hertz
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    def zpad(x):
        return pad(x, fids_in.shape[1] * zpad_factor, 'last')

    def prep_spec(x):
        x = zpad(x)
        x = apodize(
            x,
            dwelltime,
            apodize_hz)
        return FIDToSpec(x)

    if target is None:
        target = prep_spec(fids_in.mean(axis=0))
    else:
        if target.size != fids_in.shape[1]:
            raise ValueError(f'Shape of target {target.size} must match input {fids_in.shape[1]}.')
        target = FIDToSpec(zpad(target))

    shifts = []
    phases = []
    for fid in fids_in:
        xc = correlate(prep_spec(fid), target, mode='same')
        max_index = np.argmax(np.abs(xc))
        shifts.append(max_index)
        phases.append(-np.angle(xc[max_index]))

    shifts = np.asarray(shifts)
    shifts -= int(fids_in.shape[1] * 0.5 * (1 + zpad_factor))
    bandwidth = 1 / dwelltime
    shifts_hz = - shifts.astype(float) * bandwidth / (fids_in.shape[1] * (1 + zpad_factor))

    phases = np.asarray(phases)

    return np.stack([applyPhase(freqshift(fid, dwelltime, shi), phs) for fid, shi, phs in zip(fids_in, shifts_hz, phases)]), shifts_hz, phases


def my_mrsi_freq_align(
        data: NIFTI_MRS,
        target: None | NIFTI_MRS | Basis = None,
        basis_ignore: list[str] = [],
        mask: Image = None,
        zpad_factor: int = 1,
        apodize: str | float = "auto",
        higher_dimensions: str | int = "separate") -> tuple[NIFTI_MRS, Image]:
    """Frequency align MRSI data using cross correlation.

    Align either to mean, to a provided target, or to a basis spectrum
    A target FID must be a single FID with no higher dimensions.
    The spectra to use in a Basis target can be reduced using `basis_ignore`.

    :param data: MRSI data
    :type data: NIFTI_MRS
    :param target: Select what the target is.
        None = mean of all voxels within mask.
        If a Basis object is passed, alignment will be done to the unbroadened basis.
        If a NIFTI_MRS object is passed, alignment will be done to the passed spectrum.
        Defaults to None
    :type target: None | NIFTI_MRS | Basis, optional
    :param basis_ignore: List of basis spectra to remove from a Basis object target.
        Defaults to empty list, i.e. uses all the basis spectra
    :type basis_ignore: list[str], optional
    :param mask: If provided only voxels in mask will be aligned, defaults to None
    :type mask: Image, optional
    :param zpad_factor: Multiples of zero padding applied to FID before alignment, defaults to 1, 0 disables
    :type zpad_factor: int, optional
    :param apodize: Amount of apodization to apply in hertz, defaults to "auto" which estimates amount.
    :type apodize: str | float, optional
    :param higher_dimensions: How to handle higher dimensions.
        "separate" runs alignment on each higher index separately.
        "combine" runs alignment on all indices together.
        Passing an index (int) indicates the result of that index should be applied to all others.
        Defaults to "separate"
    :type higher_dimensions: str | int, optional
    :return: Returns shifted MRSI data and Image containing shifts applied in Hz
    :rtype: tuple[NIFTI_MRS, Image]
    """
    # Handle target
    if isinstance(target, Basis):
        target = np.sum(
            target.get_formatted_basis(
                data.bandwidth,
                data.shape[3],
                ignore=basis_ignore
            ), axis=-1)
    elif isinstance(target, NIFTI_MRS):
        if not np.isclose(target.dwelltime, data.dwelltime)\
                or not np.isclose(target.shape[3], data.shape[3]):
            raise ValueError('Target must have the same dwell time and number of points as data.')

        if np.prod(target.shape[4:]) > 1:
            raise ValueError('Target must not have any higher dimensions.')

        if target.shape[:3] != (1, 1, 1):
            raise ValueError('Target must be single voxel.')
        else:
            target = target[0, 0, 0, :]
    elif target is None:
        pass
    else:
        raise TypeError('target must be a NIFTI_MRS or Basis object, or None.')

    if mask is None:
        mask = np.ones(data.shape[:3]).astype(bool)
    else:
        mask = mask[:].astype(bool)

    # Calculate automatic apodization amount
    if apodize == "auto":
        apodize = calc_aprox_t2decay(
            np.moveaxis(data[:][mask, :], 1, -1).reshape(-1, data.shape[3]),
            data.dwelltime
        )
        print(f'Setting apodization filter to {apodize:0.1f} Hz.')
    elif not (isinstance(apodize, (float, int)) and apodize >= 0):
        raise ValueError('Apodize should be a value >= 0.')

    # Define nested function to avoid repeating lots of options for each case
    def xcorr_align_worker(dat):
        return xcorr_align_complex(
            dat,
            data.dwelltime,
            target=target,
            zpad_factor=zpad_factor,
            apodize_hz=apodize)[:2]

    shift_array = np.zeros(data.shape[:3] + data.shape[4:])
    if higher_dimensions == "separate":
        out = data.copy()
        shifts = np.zeros(data.shape[:3])
        for dd, idx in data.iterate_over_dims(iterate_over_space=False):
            dd[mask, :], shifts[mask] = xcorr_align_worker(dd[mask, :])
            out[idx] = dd
            shift_array[idx[:3] + idx[4:]] = shifts
        return out, Image(shift_array, xform=data.voxToWorldMat)
    elif isinstance(higher_dimensions, int):
        if higher_dimensions >= data.shape[4]:
            raise ValueError('higher_dimensions index must be < data.shape[4]')
        out = data.copy()
        shifts = np.zeros(data.shape[:3])
        _, shifts[mask] = xcorr_align_worker(data[:][mask, :, higher_dimensions])
        for dd, idx in data.iterate_over_dims(iterate_over_space=False):
            out[idx] = freqshift_array(
                dd,
                data.dwelltime,
                shifts)
            shift_array[idx[:3] + idx[4:]] = shifts
        return out, Image(shift_array, xform=data.voxToWorldMat)

    elif higher_dimensions == "combine":
        out = data[:].copy()
        tmp, shifts = xcorr_align_worker(
            np.moveaxis(data[:][mask, :], 1, -1).reshape(-1, data.shape[3]))

        out[mask, :] = np.moveaxis(tmp.reshape((np.sum(mask),) + data.shape[4:] + (data.shape[3],)), -1, 1)
        if shift_array.ndim == 3:
            shift_array[mask] = shifts.reshape((np.sum(mask),) + data.shape[4:])
        else:
            shift_array[mask, :] = shifts.reshape((np.sum(mask),) + data.shape[4:])
        return NIFTI_MRS(out, header=data.header), Image(shift_array, xform=data.voxToWorldMat)
    else:
        raise ValueError('higher_dimensions must be "separate", "combine" or an integer index.')
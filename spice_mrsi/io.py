"""
I/O utilities: CSV training data, NIfTI writing, status logging.
"""

import os
import time
import numpy as np
import psutil

from nifti_mrs.create_nmrs import gen_nifti_mrs
from fsl.data.image import Image


def log_status(msg=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | Memory: {mem:.2f} MB")


def save_training_data_as_csv(training_data: np.ndarray,
                              save_dir: str,
                              filename: str,
                              savecondition: bool):
    """
    Save training data with complex numbers to a CSV file.

    :param training_data: The training data array to save
    :param save_dir: save location
    :param filename: Name of the CSV file
    :param savecondition: Condition to decide whether to save the file
    """
    if savecondition:
        real_part = training_data.real
        imag_part = training_data.imag
        combined_data = np.hstack((real_part, imag_part))
        filepath = os.path.join(save_dir, filename + '.csv')
        np.savetxt(filepath, combined_data, delimiter=',')
        print(f"Training data saved to {filepath}")


def read_training_data_from_csv(save_dir: str,
                                filename: str) -> np.ndarray:
    """
    Read training data with complex numbers from a CSV file.

    :param save_dir: save location
    :param filename: Name of the CSV file
    :return: The training data array
    """
    filepath = os.path.join(save_dir, filename + '.csv')
    combined_data = np.loadtxt(filepath, delimiter=',')
    num_columns = combined_data.shape[1]
    real_part = combined_data[:, :num_columns // 2]
    imag_part = combined_data[:, num_columns // 2:]
    training_data = real_part + 1j * imag_part
    print(f"Training data loaded from {filepath}")
    return training_data


def calc_affine_centred(pix_sz, img_sz):
    """Compute a centred affine matrix from pixel and image size."""
    affine = np.diag(list(pix_sz) + [1,])
    offsets = -np.asarray(pix_sz).astype(float) * (np.asarray(img_sz).astype(float) / 2 + 0.5)
    affine[0:3, -1] = offsets
    return affine


def write_to_niftimrs(data, fname, Dim_Voxel, N_SEQ_POINTS,
                      sweepwidth=2800, affine=np.eye(4),
                      spec_freq=42.567 * 6.98):
    """Write data to NIfTI-MRS format.

    Note: Dim_Voxel and N_SEQ_POINTS must be passed explicitly.
    """
    D_TYPE = np.complex64
    mynmrs = gen_nifti_mrs(
        data.reshape(Dim_Voxel[0], Dim_Voxel[1], 1, N_SEQ_POINTS).astype(D_TYPE),
        1 / sweepwidth,
        spec_freq,
        "1H",
        affine=affine,
        no_conj=True
    )
    mynmrs.save(fname)


def write_to_nifti(data, fname, affine=np.eye(4)):
    """Write data to NIfTI format."""
    Image(data, affine=affine).save(fname)

"""
Simulation utilities: synthetic B0 maps, phantom generation.
"""

import numpy as np
import matplotlib.pyplot as plt


def make_b0_map(Dim_Voxel, sigma=5.0, amplitude=100.0):
    """
    Make a synthetic B0 map with a Gaussian blob centered in an ellipse.

    Parameters
    ----------
    Dim_Voxel : tuple (Nx, Ny)
        Size of the map
    sigma : float
        Standard deviation (controls spread of Gaussian), larger sigma smaller the spread
    amplitude : float
        Peak value of Gaussian at center

    Returns
    -------
    b0_map : (Nx,Ny) ndarray
        Synthetic B0 map
    """
    Nx, Ny = Dim_Voxel
    x = np.arange(Ny)
    y = np.arange(Nx)
    X, Y = np.meshgrid(x, y)

    x_min, x_max = Nx/8, 3*Nx/8
    y_min, y_max = Ny/3, 2*Ny/3

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    rx = (x_max - x_min) / 2.0
    ry = (y_max - y_min) / 2.0

    dx = (X - cx) / rx
    dy = (Y - cy) / ry
    r2 = dx**2 + dy**2

    b0_map = amplitude * np.exp(-r2 * sigma)

    plt.figure(figsize=(6, 5))
    plt.imshow(b0_map, cmap='jet', origin='lower')
    plt.colorbar(label='Δf (Hz)')
    plt.title('Synthetic B0 Map (Gaussian in ellipse)')
    plt.xlabel('x (voxels)')
    plt.ylabel('y (voxels)')
    plt.show()

    return b0_map


def make_lowres_b0_map(b0_high, low_shape):
    """
    Downsample high-res B0 map into low-res version by block averaging.

    Parameters
    ----------
    b0_high : (Nx, Ny) ndarray
        High-res ground-truth B0 map
    low_shape : (Nx_low, Ny_low)
        Desired low-res output shape

    Returns
    -------
    b0_low : (Nx_low, Ny_low) ndarray
        Low-res B0 map (block averages)
    """
    Nx, Ny = b0_high.shape
    Nx_low, Ny_low = low_shape

    srf_x = Nx // Nx_low
    srf_y = Ny // Ny_low

    b0_low = b0_high.reshape(Nx_low, srf_x, Ny_low, srf_y).mean(axis=(1, 3))

    plt.figure(figsize=(6, 5))
    plt.imshow(b0_low, cmap='jet', origin='lower')
    plt.colorbar(label='Δf (Hz)')
    plt.title('low resolution B0 Map')
    plt.xlabel('x (voxels)')
    plt.ylabel('y (voxels)')
    plt.show()

    return b0_low

"""
Visualization utilities: spectral plots, spatial maps, uncertainty displays.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, Dict, Any

from fsl_mrs.utils.plotting import FID2Spec
from fsl_mrs.utils.misc import FIDToSpec

from .graph import classify_edge_direction


def plot_anatomical_mask_points_size_directional(
    mask, anatomical_prior, edge_index,
    threshold=1e-6,
    size_scale=20,
    eps=1e-12
):
    anatomical_prior = np.array(anatomical_prior)
    dim_x, dim_y = anatomical_prior.shape
    mask = np.array(mask)

    dir_colors = {
        'vertical': 'red',
        'horizontal': 'blue',
        'main_diag': 'green',
        'anti_diag': 'orange'
    }

    valid_vals = np.abs(mask[np.abs(mask) > threshold])

    if len(valid_vals) == 0:
        print("No valid edges above threshold.")
        return

    vmin = valid_vals.min()
    vmax = valid_vals.max()

    points_by_dir = {k: {'x': [], 'y': [], 'sizes': []} for k in dir_colors.keys()}

    for idx, (v1, v2) in enumerate(edge_index):
        val = abs(mask[idx])
        if val <= threshold:
            continue

        val_norm = (val - vmin) / (vmax - vmin + eps)

        x1, y1 = divmod(v1, dim_y)
        x2, y2 = divmod(v2, dim_y)

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) == 1 and dy == 0:
            direction = 'vertical'
        elif dx == 0 and abs(dy) == 1:
            direction = 'horizontal'
        elif abs(dx) == 1 and abs(dy) == 1:
            if dx == dy:
                direction = 'anti_diag'
            else:
                direction = 'main_diag'
        else:
            continue

        mid_x = (y1 + y2) / 2
        mid_y = (x1 + x2) / 2

        points_by_dir[direction]['x'].append(mid_x)
        points_by_dir[direction]['y'].append(mid_y)
        points_by_dir[direction]['sizes'].append(val_norm * size_scale)

    plt.figure(figsize=(8, 6))
    plt.imshow(anatomical_prior, cmap='viridis', origin='lower')

    for direction, pts in points_by_dir.items():
        if pts['x']:
            plt.scatter(
                pts['x'], pts['y'],
                c=dir_colors[direction],
                s=pts['sizes'],
                alpha=0.6,
                label=direction,
                edgecolors='k',
                linewidths=0.3,
                marker='o'
            )

    plt.title('Edge-based Strength Visualization (0–1 normalized)')
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0)
    plt.axis('on')
    plt.tight_layout()
    plt.show()


def plot_spatial_mc_ana_combined(mc_cg, mc_analyt,
                                 std_uncert1_analyt, std_uncert2_analyt,
                                 water_rou_1D=None,
                                 PEAK_0_ROUGH_IDX=None, PEAK_1_ROUGH_IDX=None):
    res_array_spec1 = FIDToSpec(mc_cg, axis=-1)
    res_array_spec2 = FIDToSpec(mc_analyt, axis=-1)

    peak2est_cg = np.abs(res_array_spec1.std(axis=0))[:, PEAK_0_ROUGH_IDX].reshape(32, 32)
    peak1est_cg = np.abs(res_array_spec1.std(axis=0))[:, PEAK_1_ROUGH_IDX].reshape(32, 32)
    peak2est_analyt = np.abs(res_array_spec2.std(axis=0))[:, PEAK_0_ROUGH_IDX].reshape(32, 32)
    peak1est_analyt = np.abs(res_array_spec2.std(axis=0))[:, PEAK_1_ROUGH_IDX].reshape(32, 32)

    std_uncert1 = np.array(std_uncert1_analyt).reshape(32, 32)
    std_uncert2 = np.array(std_uncert2_analyt).reshape(32, 32)

    scale_factor = 0.02
    if water_rou_1D is not None:
        if water_rou_1D.ndim == 1:
            water_map = water_rou_1D.reshape(32, 32) * scale_factor
        else:
            water_map = water_rou_1D * scale_factor
    else:
        water_map = np.zeros((32, 32))

    water_map = np.clip(water_map, 0, None) + 0.5

    x = np.arange(32)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, water_map, cmap='gray', alpha=0.4, linewidth=0)
    ax1.plot_surface(X, Y, peak1est_cg, cmap='viridis', alpha=0.7, linewidth=0)
    ax1.plot_surface(X, Y, peak1est_analyt, cmap='plasma', alpha=0.7, linewidth=0)
    ax1.plot_surface(X, Y, std_uncert1, cmap='coolwarm', alpha=0.5, linewidth=0)
    ax1.set_title('Peak 1 Uncertainty (CG vs Analytical)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Uncertainty')
    ax1.set_zlim(0, 0.5)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, water_map, cmap='gray', alpha=0.4, linewidth=0)
    ax2.plot_surface(X, Y, peak2est_cg, cmap='viridis', alpha=0.7, linewidth=0)
    ax2.plot_surface(X, Y, peak2est_analyt, cmap='plasma', alpha=0.7, linewidth=0)
    ax2.plot_surface(X, Y, std_uncert2, cmap='coolwarm', alpha=0.5, linewidth=0)
    ax2.set_title('Peak 2 Uncertainty (CG vs Analytical)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Uncertainty')
    ax2.set_zlim(0, 0.5)

    plt.show()

    return peak1est_cg, std_uncert1_analyt


def plot_spec_analyt(ax, rcon, ppm_axis, limits):
    """Plot uncertainty in the spectral domain for specified voxels."""
    for voxel in [2, 8, 16, 24, 30]:
        ax.plot(ppm_axis, [voxel] * len(ppm_axis), rcon[:, voxel].real)
    ax.set_zlim(limits)
    ax.set_xlim([ppm_axis[-1], ppm_axis[0]])
    ax.set_xlabel('$\\delta$ / ppm')
    ax.set_ylabel('Voxel #')
    ax.set_zlabel('Uncertainty')
    ax.view_init(elev=15, azim=-80)
    ax.legend()
    plt.show()


def plot_mc_compare_spec(res_array1: np.ndarray, res_array2: np.ndarray, res_array3: np.ndarray,
                         plot_spec_mc, plot_spec_analyt, ppm_axis, limits):
    """Summarize and plot Monte Carlo spectral results."""
    res_array_spec1 = FIDToSpec(res_array1, axis=-1)
    res_array_spec2 = FIDToSpec(res_array2, axis=-1)
    res_array_spec3 = res_array3

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_title("Spectral Uncert for cg")
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title("Spectral Uncert for analy")
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title("Spectral Uncert for analy Laplace")
    ax1.autoscale()
    ax2.autoscale()
    ax3.autoscale()

    plot_spec_mc(ax1, res_array_spec1.std(axis=0), ppm_axis, limits)
    plot_spec_mc(ax2, res_array_spec2.std(axis=0), ppm_axis, limits)
    plot_spec_analyt(ax3, res_array_spec3, ppm_axis, limits)
    plt.show()


def plot_mc_compare_spat(res_array1: np.ndarray, res_array2: np.ndarray,
                         plot_spatial_mc, water_rou_1D, limits=[0, 5e-1]):
    """Summarize and plot Monte Carlo spatial results."""
    res_array_spec1 = FIDToSpec(res_array1, axis=-1)
    res_array_spec2 = FIDToSpec(res_array2, axis=-1)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Spatial Uncert for cg")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Spatial Uncert for analy")

    plot_spatial_mc(ax1, res_array_spec1.std(axis=0), water_rou_1D=water_rou_1D, limits=limits)
    plot_spatial_mc(ax2, res_array_spec2.std(axis=0), water_rou_1D=water_rou_1D, limits=limits)
    plt.show()


def plot_spec_mc(ax, rcon, ppm_axis, limits):
    """Plot uncertainty in the spectral domain for specified voxels."""
    for voxel in [2, 8, 16, 24, 30]:
        ax.plot(ppm_axis, rcon[voxel, :].real, zs=voxel, zdir='y')
    ax.set_zlim(limits)
    ax.set_xlim([ppm_axis[-1], ppm_axis[0]])
    ax.set_xlabel('$\\delta$ / ppm')
    ax.set_ylabel('Voxel #')
    ax.set_zlabel('Uncertainty')
    ax.view_init(elev=15, azim=-80, roll=0)


def plot_spatial_mc(ax, recon, water_rou_1D=None, limits=[0, 5e-1], scale_factor=0.02,
                    PEAK_0_ROUGH_IDX=None, PEAK_1_ROUGH_IDX=None):
    """Plot Peak1 and Peak2 uncertainty as 3D surfaces overlaid with water density."""
    ax.remove()

    peak1est = np.abs(recon)[:, PEAK_0_ROUGH_IDX].reshape(32, 32)
    peak2est = np.abs(recon)[:, PEAK_1_ROUGH_IDX].reshape(32, 32)

    if water_rou_1D is not None:
        if water_rou_1D.ndim == 1:
            water_map = water_rou_1D.reshape(32, 32) * scale_factor
        else:
            water_map = water_rou_1D * scale_factor
    else:
        water_map = np.zeros((32, 32))

    water_map = np.clip(water_map, 0, None) + 0.5

    x = np.arange(32)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, water_map, cmap='gray', alpha=0.2, linewidth=0, antialiased=False)
    ax1.plot_surface(X, Y, peak1est, cmap='viridis', alpha=0.7, linewidth=0, antialiased=False)
    ax1.set_zlim(limits[0], limits[1])
    ax1.set_title('Peak 1 uncertainty with water overlay')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Uncertainty')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, water_map, cmap='gray', alpha=0.2, linewidth=0, antialiased=False)
    ax2.plot_surface(X, Y, peak2est, cmap='plasma', alpha=0.7, linewidth=0, antialiased=False)
    ax2.set_zlim(limits[0], limits[1])
    ax2.set_title('Peak 2 uncertainty with water overlay')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Uncertainty')

    plt.show()


def condition_label(minpool_param: int,
                    minpooling_Handler: bool,
                    lambda_param: float,
                    Wmax_param: float,
                    Undersample_Handler: bool,
                    Unsersample_param: float) -> str:
    if minpooling_Handler == False:
        minpool_condition = 'Minpooling Off'
    else:
        minpool_condition = 'poolsize = ' + str(minpool_param)

    if lambda_param == 0:
        lambda_condition = 'Spatial Constraint Off'
    else:
        lambda_condition = 'lambda = ' + str(lambda_param) + '\n' + 'Wmax = ' + str(Wmax_param)
    if Undersample_Handler == False:
        Undersample_condition = 'Fully Sampled'
    else:
        Undersample_condition = 'Undersample' + str(Unsersample_param)
    return '\n' + minpool_condition + '\n' + lambda_condition + '\n' + Undersample_condition


def plot_popt_arbmet(popt, META_LIST, Dim_Voxel, col_idx=None, minpool_param=0,
                     minpooling_Handler=False, lambda_param=0, Wmax_param=0,
                     Undersample_Handler=False, Unsersample_param=1.0):
    """Plot optimized parameters for arbitrary number of metabolites."""
    N_row, N_col = Dim_Voxel
    N_voxel = N_row * N_col
    num_metab = len(META_LIST)

    lw_all = popt[:, 0::2].reshape(N_voxel, num_metab)
    conc_all = popt[:, 1::2].reshape(N_voxel, num_metab)

    runtime_condition = condition_label(
        minpool_param=minpool_param, minpooling_Handler=minpooling_Handler,
        lambda_param=lambda_param, Wmax_param=Wmax_param,
        Undersample_Handler=Undersample_Handler, Unsersample_param=Unsersample_param)

    if col_idx is not None:
        plt.figure(figsize=(8, 4))
        for m, meta in enumerate(META_LIST):
            conc_2d = conc_all[:, m].reshape(N_row, N_col, order='C')
            conc_slice = conc_2d[:, col_idx]
            plt.plot(conc_slice, marker='o', linestyle='-', label=f'{meta} - Col {col_idx + 1}')
        plt.xlabel('Row Index')
        plt.ylabel('Value')
        plt.title(f'Optimized Fit Parameters (Column {col_idx + 1})\n{runtime_condition}')
        plt.grid(True)
        plt.legend()
        plt.show()

        vmin, vmax = 0, 1.5
        n_subplots = num_metab
        plt.figure(figsize=(5 * n_subplots, 5))
        for m, meta in enumerate(META_LIST):
            conc_2d = conc_all[:, m].reshape(N_row, N_col, order='C')
            plt.subplot(1, n_subplots, m + 1)
            plt.imshow(conc_2d, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
            plt.title(f'Peak {m+1} Concentration ({meta})')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar(label='Amplitude')
        plt.suptitle(f'Fitted Concentration Maps\n{runtime_condition}')
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(10, 5))
        for m, meta in enumerate(META_LIST):
            plt.plot(conc_all[:, m], marker='o', linestyle='-', label=f'{meta}')
        plt.xlabel('Voxel Index')
        plt.ylabel('Value')
        plt.title(f'Optimized Fit Parameters from MRS Fitting\n{runtime_condition}')
        plt.legend()
        plt.show()


def plot_popt(popt, N_row, N_col, col_idx=None, minpool_param=0, minpooling_Handler=False,
              lambda_param=0, Wmax_param=0, Undersample_Handler=False, Unsersample_param=1.0):
    """Plot optimized parameters for 2-metabolite fit.

    N_row and N_col must be passed explicitly.
    """
    N_VOXEL = N_row * N_col

    lw1 = popt[0:N_VOXEL]
    fit1 = popt[N_VOXEL:2 * N_VOXEL]
    lw2 = popt[2 * N_VOXEL:3 * N_VOXEL]
    fit2 = popt[3 * N_VOXEL:4 * N_VOXEL]

    runtime_condition = condition_label(
        minpool_param=minpool_param, minpooling_Handler=minpooling_Handler,
        lambda_param=lambda_param, Wmax_param=Wmax_param,
        Undersample_Handler=Undersample_Handler, Unsersample_param=Unsersample_param)

    if col_idx is not None:
        fit1_2d = fit1.reshape(N_row, N_col, order='C')
        fit2_2d = fit2.reshape(N_row, N_col, order='C')
        fit1_slice = fit1_2d[:, col_idx]
        fit2_slice = fit2_2d[:, col_idx]

        plt.figure(figsize=(8, 4))
        plt.plot(fit1_slice, marker='o', label=f'Fit 1 - Col {col_idx + 1}')
        plt.plot(fit2_slice, marker='s', label=f'Fit 2 - Col {col_idx + 1}')
        plt.xlabel('Row Index')
        plt.ylabel('Value')
        plt.title(f'Optimized Fit Parameters (Column {col_idx + 1})\n{runtime_condition}')
        plt.grid(True)
        plt.legend()
        plt.show()

        vmin, vmax = 0, 1.0
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(fit1_2d, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
        plt.title('Peak 1 Concentration (Glu)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Amplitude')
        plt.subplot(1, 2, 2)
        plt.imshow(fit2_2d, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
        plt.title('Peak 2 Concentration (Cho)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='Amplitude')
        plt.suptitle(f'Fitted Concentration Maps\n{runtime_condition}')
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(10, 5))
        plt.plot(fit1, marker='o', linestyle='-', label='Fit 1')
        plt.plot(fit2, marker='s', linestyle='-', label='Fit 2')
        plt.xlabel('Voxel Index')
        plt.ylabel('Value')
        plt.title(f'Optimized Fit Parameters from MRS Fitting\n{runtime_condition}')
        plt.legend()
        plt.show()


def plot_bm_and_bmFID(bm_FIDs: list, ppm_axis):
    """Plot the spectral basis bm and its corresponding FID for indices 0 and 1."""
    bm_FID_0 = bm_FIDs[0]
    bm_FID_1 = bm_FIDs[1]

    bm_0 = FID2Spec(bm_FID_0)
    bm_1 = FID2Spec(bm_FID_1)

    plt.figure(figsize=(8, 5))
    plt.plot(ppm_axis, np.real(bm_0), label="Real Part (bm 0)", linestyle="-")
    plt.plot(ppm_axis, np.imag(bm_0), label="Imaginary Part (bm 0)", linestyle="dashed")
    plt.plot(ppm_axis, np.real(bm_1), label="Real Part (bm 1)", linestyle="-")
    plt.plot(ppm_axis, np.imag(bm_1), label="Imaginary Part (bm 1)", linestyle="dashed")
    plt.xlabel("Frequency Index")
    plt.ylabel("Intensity")
    plt.title("Spectral Basis (bm_0 and bm_1)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(np.real(bm_FID_0), label="Real Part (bm_FID 0)", linestyle="-")
    plt.plot(np.real(bm_FID_1), label="Real Part (bm_FID 1)", linestyle="-.")
    plt.xlabel("Time Index")
    plt.ylabel("Intensity")
    plt.title("FID Signal (bm_FID_0 and bm_FID_1)")
    plt.legend()
    plt.show()


def plot_voxel_sum_map(
    mask,
    anatomical_prior,
    edge_index,
    threshold=1e-6,
    use_abs=True,
    eps=1e-12,
    cmap='cividis'
):
    anatomical_prior = np.asarray(anatomical_prior)
    mask = np.asarray(mask)
    edge_index = np.asarray(edge_index)

    dim_x, dim_y = anatomical_prior.shape
    num_voxels = dim_x * dim_y

    voxel_sum = np.zeros(num_voxels)
    voxel_count = np.zeros(num_voxels)

    for idx, (v1, v2) in enumerate(edge_index):
        val = abs(mask[idx]) if use_abs else mask[idx]
        if val <= threshold:
            continue
        voxel_sum[v1] += val
        voxel_sum[v2] += val
        voxel_count[v1] += 1
        voxel_count[v2] += 1

    voxel_sum_map = voxel_sum.reshape(dim_x, dim_y)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='black')
    ax.set_facecolor('black')
    ax.imshow(anatomical_prior, cmap='gray', origin='lower', alpha=1.0)

    cmap_obj = plt.cm.Greens.copy()
    cmap_obj.set_under('black')

    im = ax.imshow(voxel_sum_map, cmap=cmap_obj, origin='lower', vmax=30000, alpha=0.8)

    ax.set_title('Voxel-level sum prior map', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    plt.tight_layout()
    plt.show()

    return voxel_sum_map


def plot_voxel_spectrum_and_maps(
    spice_test: np.ndarray,
    img_shape: tuple,
    voxel_x: int,
    voxel_y: int,
    voxel_z: int = 0,
    phi0: float = 0.0,
    brain_mask_inner: Optional[np.ndarray] = None,
    PPM_AXIS: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (18, 5),
    show: bool = True,
    dark_mode: bool = False,
):
    """
    Plot spectral magnitude maps and selected voxel spectrum.
    voxel_x = column (x-axis), voxel_y = row (y-axis).
    """
    nx, ny, nt = img_shape
    expected_len = nx * ny * nt
    spice_test = np.asarray(spice_test)

    if spice_test.size != expected_len:
        if spice_test.ndim == 3 and spice_test.shape == (nx, ny, nt):
            SPICE_img = spice_test
        else:
            raise ValueError(
                f"spice_test size mismatch. expected {expected_len}, got {spice_test.size}."
            )
    else:
        SPICE_img = spice_test.reshape(nx, ny, nt)

    Spec = SPICE_img[:, :, np.newaxis, :]
    nx, ny, nz, npts = Spec.shape

    if not (0 <= voxel_x < ny and 0 <= voxel_y < nx and 0 <= voxel_z < nz):
        pass

    mag_map = np.mean(np.abs(Spec[..., :]), axis=-1)
    mag_map_2d = mag_map[:, :, voxel_z]

    spec_voxel = Spec[voxel_y, voxel_x, voxel_z, :].copy().astype(np.complex128)
    if phi0 != 0:
        spec_voxel = spec_voxel * np.exp(1j * np.deg2rad(phi0))

    spec_real = np.real(spec_voxel)
    spec_abs = np.abs(spec_voxel)

    if brain_mask_inner is None:
        mask_2d = np.ones((nx, ny), dtype=bool)
    else:
        bm = np.asarray(brain_mask_inner)
        if bm.ndim == 2:
            mask_2d = bm
        elif bm.ndim == 3:
            mask_2d = bm[:, :, voxel_z]
        else:
            raise ValueError("brain_mask_inner must be 2D or 3D boolean array")

    mag_map_masked = mag_map_2d.copy()
    mag_map_masked[~mask_2d] = np.nan

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    if dark_mode:
        fig.patch.set_facecolor('black')
        for ax in axs:
            ax.set_facecolor('black')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

    im = axs[0].imshow(np.abs(mag_map_2d), cmap=cmap, origin='lower')
    axs[0].set_title(f"Average spectral magnitude (slice {voxel_z})")
    axs[0].set_xlabel("x (cols)")
    axs[0].set_ylabel("y (rows)")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    rect = Rectangle(
        (voxel_x - 0.5, voxel_y - 0.5), 1, 1,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    axs[0].add_patch(rect)

    im1 = axs[1].imshow(np.abs(mag_map_masked), cmap=cmap, origin='lower')
    axs[1].set_title("Average spectral magnitude (brain mask)")
    axs[1].set_xlabel("x (cols)")
    axs[1].set_ylabel("y (rows)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    if PPM_AXIS is None:
        x_axis = np.arange(npts)
        axs[2].set_xlabel("index")
    else:
        x_axis = PPM_AXIS
        axs[2].set_xlabel("ppm")

    axs[2].plot(x_axis, spec_real, label="Real")
    axs[2].plot(x_axis, spec_abs, label='Magnitude', alpha=0.7)
    axs[2].set_title(f"Spectrum at voxel (row={voxel_y}, col={voxel_x}, slice={voxel_z})")
    if PPM_AXIS is not None:
        axs[2].invert_xaxis()
    axs[2].set_ylabel("Signal")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    if show:
        plt.show()

    return spec_voxel, fig, axs


def plot_recon_b0_and_voxel_spectrum(
    Spec: np.ndarray,
    PPM_AXIS: np.ndarray,
    FREQ_AXIS: np.ndarray,
    B0_map: np.ndarray,
    voxel_x: int,
    voxel_y: int,
    phi0: float = 0.0,
    target_ppm: float = None,
    window_ppm: float = 0.2,
    cmap_mag: str = "viridis",
    cmap_b0: str = "coolwarm",
    cmap_delta: str = "seismic",
    figsize: Tuple[float, float] = (18, 6),
    show: bool = True
) -> Tuple[np.ndarray, plt.Figure, Dict[str, Any]]:
    """Plot delta map, B0 map, and selected voxel spectrum."""
    Spec = np.asarray(Spec)
    PPM_AXIS = np.asarray(PPM_AXIS)
    FREQ_AXIS = np.asarray(FREQ_AXIS)
    B0_map = np.asarray(B0_map)

    if Spec.ndim != 3:
        raise ValueError("Spec must be 3D: (nx,ny,npts) or (ny,nx,npts).")

    if Spec.shape[0:2] == B0_map.shape:
        ny, nx, npts = Spec.shape
        Spec2 = Spec
    elif (Spec.shape[1], Spec.shape[0]) == B0_map.shape:
        nx, ny, npts = Spec.shape
        Spec2 = np.transpose(Spec, (1, 0, 2))
    else:
        raise ValueError("Spec spatial dims don't match B0_map. "
                         f"Spec.shape[:2]={Spec.shape[:2]}, B0_map.shape={B0_map.shape}")

    if PPM_AXIS.size != npts or FREQ_AXIS.size != npts:
        raise ValueError("PPM_AXIS and FREQ_AXIS must match Spec's spectral length.")

    if not (0 <= voxel_x < nx and 0 <= voxel_y < ny):
        raise IndexError("voxel_x/voxel_y out of bounds relative to B0_map / Spec.")

    if target_ppm is None:
        if np.any(np.isclose(PPM_AXIS, 3.0, atol=1e-6)):
            target_ppm = 3.0
        else:
            target_ppm = float(PPM_AXIS[len(PPM_AXIS) // 2])

    mag_map_2d = np.mean(np.abs(Spec2), axis=-1)

    spec_voxel = Spec2[voxel_y, voxel_x, :].astype(np.complex128)
    if phi0 != 0:
        spec_voxel = spec_voxel * np.exp(1j * np.deg2rad(phi0))
    spec_real = np.real(spec_voxel)
    spec_abs = np.abs(spec_voxel)

    idx_target = int(np.argmin(np.abs(PPM_AXIS - target_ppm)))
    freq_at_target = float(FREQ_AXIS[idx_target])
    window_mask = np.abs(PPM_AXIS - target_ppm) <= window_ppm
    idxs_in_window = np.nonzero(window_mask)[0]
    if idxs_in_window.size == 0:
        raise ValueError("No points in ppm window. Increase window_ppm or check PPM_AXIS.")

    mag_window = spec_abs[idxs_in_window]
    if mag_window.size == 0 or np.all(~np.isfinite(mag_window)) or mag_window.max() == 0:
        peak_idx = idxs_in_window[0]
        peak_freq = np.nan
        delta_hz = np.nan
    else:
        rel_peak_idx = int(np.nanargmax(mag_window))
        peak_idx = int(idxs_in_window[rel_peak_idx])
        peak_freq = float(FREQ_AXIS[peak_idx])
        delta_hz = float(peak_freq - freq_at_target)

    spec_mag_flat = np.abs(Spec2).reshape(-1, npts)
    mag_window_all = spec_mag_flat[:, window_mask]
    valid_mask = np.isfinite(mag_window_all).all(axis=1) & (np.nanmax(mag_window_all, axis=1) > 0)
    nvox = spec_mag_flat.shape[0]
    peak_global_idx = np.full((nvox,), -1, dtype=int)
    if np.any(valid_mask):
        rel_peaks = np.nanargmax(mag_window_all[valid_mask], axis=1)
        window_indices = np.nonzero(window_mask)[0]
        peak_global_idx[valid_mask] = window_indices[rel_peaks]
    peak_freqs = np.full((nvox,), np.nan, dtype=float)
    valid_peak_vox = peak_global_idx >= 0
    if np.any(valid_peak_vox):
        peak_freqs[valid_peak_vox] = FREQ_AXIS[peak_global_idx[valid_peak_vox]]
    delta_flat = peak_freqs - float(freq_at_target)
    delta_map = delta_flat.reshape((ny, nx))

    b0_value = float(B0_map[voxel_y, voxel_x])
    voxel_delta = float(delta_map[voxel_y, voxel_x]) if np.isfinite(delta_map[voxel_y, voxel_x]) else np.nan

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    finite_deltas = delta_map[np.isfinite(delta_map)]
    if finite_deltas.size > 0:
        vmax = np.percentile(np.abs(finite_deltas), 98)
        vmin = -vmax
    else:
        vmax, vmin = 1.0, -1.0
    im0 = axs[0].imshow(delta_map, origin='lower', aspect='auto', cmap=cmap_delta, vmin=vmin, vmax=vmax)
    axs[0].set_title("Delta map (Hz)")
    axs[0].set_xlabel("x (cols)")
    axs[0].set_ylabel("y (rows)")
    rect0 = Rectangle((voxel_x - 0.5, voxel_y - 0.5), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
    axs[0].add_patch(rect0)
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(B0_map, cmap=cmap_b0, origin='lower', aspect='auto')
    axs[1].set_title("B0 map")
    axs[1].set_xlabel("x (cols)")
    axs[1].set_ylabel("y (rows)")
    rect1 = Rectangle((voxel_x - 0.5, voxel_y - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
    axs[1].add_patch(rect1)
    annot_text = f"voxel ({voxel_y},{voxel_x})\nb0 = {b0_value:.4f}\ndelta = {voxel_delta:.2f} Hz"
    axs[1].text(0.02, 0.98, annot_text, transform=axs[1].transAxes,
                fontsize=10, va='top', ha='left', color='white',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.4'))
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].plot(PPM_AXIS, spec_real, label=f"Real (rot {phi0:.0f}°)")
    axs[2].plot(PPM_AXIS, spec_abs, label='|S| (magnitude)', alpha=0.7)
    axs[2].axvline(x=target_ppm if target_ppm is not None else PPM_AXIS[idx_target],
                   color='k', linestyle='--', linewidth=1, label='target ppm')
    if np.isfinite(peak_freq):
        axs[2].axvline(x=PPM_AXIS[peak_idx], color='r', linestyle='-', linewidth=1, label='Detected peak')
        axs[2].annotate(f"{delta_hz:.2f} Hz",
                        xy=(PPM_AXIS[peak_idx], spec_abs[peak_idx]),
                        xytext=(PPM_AXIS[peak_idx], spec_abs[peak_idx] * 1.1),
                        arrowprops=dict(arrowstyle="->", color='red'),
                        fontsize=9, color='red')
    axs[2].set_title(f"Spectrum at voxel (row={voxel_y}, col={voxel_x})")
    axs[2].set_xlabel("ppm")
    axs[2].invert_xaxis()
    axs[2].set_ylabel("Signal")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    if show:
        plt.show()

    meta = {
        "delta_map": delta_map,
        "mag_map": mag_map_2d,
        "peak_freq": peak_freq if 'peak_freq' in locals() else np.nan,
        "peak_idx": int(peak_idx) if 'peak_idx' in locals() else -1,
        "spec_voxel": spec_voxel,
        "peak_global_idx_flat": peak_global_idx
    }
    return spec_voxel, fig, meta


def plot_average_variation(
    spice_test: np.ndarray,
    img_shape: tuple,
    voxel_x: int,
    voxel_y: int,
    voxel_z: int = 0,
    phi0: float = 0.0,
    brain_mask_inner: Optional[np.ndarray] = None,
    brain_prior_map: Optional[np.ndarray] = None,
    prior_alpha: float = 0.35,
    threshold: Optional[float] = None,
    PPM_AXIS: Optional[np.ndarray] = None,
    cmap: str = "Reds",
    figsize: Tuple[float, float] = (18, 5),
    show: bool = True
):
    nx, ny, nt = img_shape
    expected_len = nx * ny * nt
    spice_test = np.asarray(spice_test)

    if spice_test.size != expected_len:
        if spice_test.ndim == 3 and spice_test.shape == (nx, ny, nt):
            SPICE_img = spice_test
        else:
            raise ValueError(
                f"spice_test size mismatch. expected {expected_len}, got {spice_test.size}."
            )
    else:
        SPICE_img = spice_test.reshape(nx, ny, nt)

    Spec = SPICE_img[:, :, np.newaxis, :]
    nx, ny, nz, npts = Spec.shape

    mag_map = np.mean(np.abs(Spec[..., :]), axis=-1)
    mag_map_2d = mag_map[:, :, voxel_z]

    spec_voxel = Spec[voxel_y, voxel_x, voxel_z, :].copy().astype(np.complex128)
    if phi0 != 0:
        spec_voxel = spec_voxel * np.exp(1j * np.deg2rad(phi0))

    spec_real = np.real(spec_voxel)
    spec_abs = np.abs(spec_voxel)

    if brain_mask_inner is None:
        mask_2d = np.ones((nx, ny), dtype=bool)
    else:
        bm = np.asarray(brain_mask_inner)
        if bm.ndim == 2:
            mask_2d = bm
        elif bm.ndim == 3:
            mask_2d = bm[:, :, voxel_z]
        else:
            raise ValueError("brain_mask_inner must be 2D or 3D boolean array")

    mag_map_masked = mag_map_2d.copy()
    mag_map_masked[~mask_2d] = np.nan

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    im = axs[0].imshow(np.abs(mag_map_2d), cmap='viridis', origin='lower')
    axs[0].set_title(f"Average spectral magnitude (slice {voxel_z})")
    axs[0].set_xlabel("x (cols)")
    axs[0].set_ylabel("y (rows)")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    rect = Rectangle(
        (voxel_x - 0.5, voxel_y - 0.5), 1, 1,
        linewidth=2, edgecolor='green', facecolor='none'
    )
    axs[0].add_patch(rect)

    if threshold is None:
        im1 = axs[1].imshow(np.abs(mag_map_masked), cmap=cmap, origin='lower')
    else:
        im1 = axs[1].imshow(np.abs(mag_map_masked), cmap=cmap, origin='lower', vmin=0, vmax=threshold)

    axs[1].set_title("Average spectral magnitude (brain mask)")
    axs[1].set_xlabel("x (cols)")
    axs[1].set_ylabel("y (rows)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    if brain_prior_map is not None:
        prior_2d = np.asarray(brain_prior_map)
        if prior_2d.ndim == 3:
            prior_2d = prior_2d[:, :, voxel_z]
        if prior_2d.shape != (nx, ny):
            raise ValueError(f"brain_prior_map shape must be {(nx, ny)} or {(nx, ny, nz)}")
        axs[1].imshow(prior_2d, cmap='gray', origin='lower', alpha=prior_alpha)

    if PPM_AXIS is None:
        x_axis = np.arange(npts)
        axs[2].set_xlabel("index")
    else:
        x_axis = PPM_AXIS
        axs[2].set_xlabel("ppm")

    axs[2].plot(x_axis, spec_real, label=f"Real (rot {phi0:.0f}°)")
    axs[2].plot(x_axis, spec_abs, label='|S| (magnitude)', alpha=0.7)
    axs[2].set_title(f"Spectrum at voxel (row={voxel_y}, col={voxel_x}, slice={voxel_z})")
    if PPM_AXIS is not None:
        axs[2].invert_xaxis()
    axs[2].set_ylabel("Signal")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    if show:
        plt.show()

    return spec_voxel, fig, axs

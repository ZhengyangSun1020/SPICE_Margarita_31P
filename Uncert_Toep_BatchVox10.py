''' 
This python script is for invivo SPICE uncertainty analysis
'''

############# 1 Load dependencies. ############

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import libs
import matplotlib.pyplot as plt
import numpy as np
from fsl_mrs.core import basis
from fsl_mrs.utils.plotting import FID2Spec
from fsl_mrs.utils.misc import FIDToSpec,SpecToFID
from scipy.sparse.linalg import LinearOperator, aslinearoperator,cg
import scipy.sparse as sp ## help to vectorize matrices
from fsl_mrs.utils import mrs_io 
from SPICE_2D_Ancillary import (
    read_training_data_from_csv,
    save_training_data_as_csv,
    calc_Bmatrix,
    plot_anatomical_mask_points_size_directional,
    NUFFTOp,
)

from warnings import filterwarnings

import torch
import torchkbnufft as tkbn
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from scipy.sparse.linalg import LinearOperator as SciLin
from scipy.sparse.linalg import LinearOperator, cg,bicgstab, lobpcg

import traceback


# import tracemalloc





# from mrisensesim import mrisensesim

device = torch.device("cpu")

filterwarnings("ignore") # ignore floor divide warnings
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


############# Marco definations ############
''' 
AXIS AND BASIS
'''

BASIS_DIR='/home/fs0/fcj757/scratch/ISMRM2026/Hessian/ISMRM2026_BASIS/'


dwelltime = 5e-6#6.1e-4 #5e-6#1 / N_SEQ_BANDWIDTH
K_POINTS = 39762  
N_SEQ_POINTS = 300
N_SHOTs = 360
PPM_CENTER = 3#4.65
N_recorded_points = K_POINTS #* N_SHOTS
TS = (N_recorded_points/ N_SEQ_POINTS)*dwelltime
sweepwidth = 1 / TS

fullbasis = mrs_io.read_basis(BASIS_DIR) 
basis = fullbasis.get_formatted_basis(bandwidth=sweepwidth, points=N_SEQ_POINTS)#ignore=[list of metabolites you don't want to include] 


center_freq = 297.219338
original_points = fullbasis.original_points

FREQ_AXIS = np.linspace(-sweepwidth / 2, sweepwidth / 2, N_SEQ_POINTS)#np.arange(-sweepwidth/2, sweepwidth/2, sweepwidth/N_SEQ_POINTS)

PPM_AXIS = FREQ_AXIS / center_freq + PPM_CENTER

TIME_AXIS = np.linspace(TS,
                           TS * N_SEQ_POINTS,
                           N_SEQ_POINTS)

''' 
Parameters to adjust SPICE
'''
#SPICE performance related
NUM_SPICE_RANK = 15
NOISE_SNR = 20
Lamda_1 = 1e-10#1e-4#0.01#0.05
LAMBDA_WE_max = 500#4.5
MITER = 300


Dim_Voxel = [64,64]
N_VOXEL = Dim_Voxel[0]*Dim_Voxel[1]
# f_loc_vs = './invivo_260115/251217/'
f_loc_jlt = '/home/fs0/fcj757/scratch/ISMRM2026/Hessian/invivo_260305/cr/'
f_loc_hess = '/home/fs0/fcj757/scratch/ISMRM2026/Hessian/invivo_260305/cr/Hess_vox/'
D_TYPE = np.complex64
T_D_TYPE = torch.complex64
Trej_D_TYPE = np.float32


#subspace training related
SAVE_DIR = f_loc_jlt + 'SPICE_result/'
CSV_FILE_NAME = 'SS260227'
SUBSPACE_DATA_RW = False

param_window_ppm=0.1

############# Metabolites basis ############

''' 
Register Basis set
'''
META_LIST = [
    'Cr', 'GABA',  'Glu', 
     'Lac', 'NAA', 'Ins', 'PCh']
NUM_METAB = len(META_LIST)

basis_dict = {}
for i, meta_nam in enumerate(fullbasis.names):
    basis_dict[f"Basis_{meta_nam}_FID"] = basis[:, i]   # shape (points,), dtype=complex128

# Access:
# META_LIST = ['Glu', 'NAA', 'Ala', 'GABA']
for m in META_LIST:
    key = f"Basis_{m}_FID"
    if key not in basis_dict:
        raise KeyError(f"Missing key in basis_dict: {key}")
    globals()[key] = basis_dict[key]   # defines Basis_Glu_FID, Basis_NAA_FID, ...

Basis_FID = {}
Basis_FID_SPEC = {}

for i, meta in enumerate(META_LIST):
    # find where this metabolite appears in fullbasis.names
    try:
        j = fullbasis.names.index(meta)   # index in fullbasis
    except ValueError:
        raise ValueError(f"{meta} not found in fullbasis.names")

    # extract the right FID from basis and shift
    fid = basis[:, j]
    # fid*= np.exp(-1j * BASIS_SHIFT * TIME_AXIS.T * np.pi * 2)

    # store by index
    globals()[f"Basis{i}_FID"] = globals()[f"Basis_{meta}_FID"] = Basis_FID[i] = fid.conj()
    globals()[f"Basis{i}_FID_SPEC"] = globals()[f"Basis_{meta}_FID_SPEC"] =  Basis_FID_SPEC[i] = FID2Spec(fid.conj())


# bm_FIDs = [Basis0_FID,Basis1_FID]
bm_FIDs = [globals()[f"Basis{i}_FID"] for i in range(NUM_METAB)]


''' 
Build the F operator via torchkbnufft
'''

# DTYPE for numpy arrays used by linear operator
LINOP_DTYPE = np.complex64

def make_linop(op: NUFFTOp, Dim_Voxel: list, N_K_POINTS: int, N_K_SHOTS:int,  N_img_SEQ_POINTS: int, NUM_CMAP_CHANNEL:int):
    Ny, Nx = Dim_Voxel
    Nvox = Ny * Nx   # flatten image

    def matvec(x):
        # x: (Nvox,) -> reshape to image
        x_img = x.astype(D_TYPE).reshape(Ny, Nx, N_img_SEQ_POINTS)
        y = op.A_np(x_img)  # (1,1,N_K_POINTS)
        y_arr = np.asarray(y, dtype=D_TYPE)
        return y_arr

    def rmatvec(y):
        # y: (N_K_POINTS,) -> back to image
        if NUM_CMAP_CHANNEL == 1:
            y = y.astype(D_TYPE).reshape(-1)
        else:
            y = y.astype(D_TYPE).reshape(NUM_CMAP_CHANNEL,-1)
        x_img = op.AH_np(y)  # (1,NUM_CMAP_CHANNEL,Ny,Nx)
        x_arr = np.asarray(x_img, dtype=D_TYPE)
        return x_arr
    

    return LinearOperator((N_K_POINTS*NUM_CMAP_CHANNEL*N_K_SHOTS, Nvox*N_img_SEQ_POINTS), matvec=matvec, rmatvec=rmatvec, dtype=D_TYPE)


def make_fft1d_op(n: int, mode="spec2fid", dtype=np.complex64):
    """
    mode:
        "spec2fid": spec -> fid
        "fid2spec": fid -> spec
    """

    n_sum = n[0] * n[1] * n[-1]
    fft_axis = len(n) - 1   # 最后一维做 1D FFT

    def spec2fid(x):
        return np.fft.ifft(
            np.fft.ifftshift(x, axes=fft_axis),
            axis=fft_axis,
            norm='ortho'
        )

    def fid2spec(x):
        return np.fft.fftshift(
            np.fft.fft(x, axis=fft_axis, norm='ortho'),
            axes=fft_axis
        )

    if mode == "spec2fid":
        mat = spec2fid
        rmat = fid2spec
    elif mode == "fid2spec":
        mat = fid2spec
        rmat = spec2fid
    else:
        raise ValueError("mode must be 'spec2fid' or 'fid2spec'")

    def matvec(x):
        x = np.asarray(x).reshape(n)
        return mat(x).ravel().astype(dtype, copy=False)

    def rmatvec(x):
        x = np.asarray(x).reshape(n)
        return rmat(x).ravel().astype(dtype, copy=False)

    def matmat(X):
        X = np.asarray(X)
        if X.ndim == 1:
            return matvec(X)[:, None]
        k = X.shape[1]
        X = X.reshape((*n, k))
        Y = mat(X)
        return Y.reshape(n_sum, k).astype(dtype, copy=False)

    def rmatmat(X):
        X = np.asarray(X)
        if X.ndim == 1:
            return rmatvec(X)[:, None]
        k = X.shape[1]
        X = X.reshape((*n, k))
        Y = rmat(X)
        return Y.reshape(n_sum, k).astype(dtype, copy=False)

    return LinearOperator(
        shape=(n_sum, n_sum),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=dtype,
    )


def spec_to_fid(spec):
    return np.fft.ifft(np.fft.ifftshift(spec, axes=-1), axis=-1, norm='ortho')

def fid_to_spec(fid):
    return np.fft.fftshift(np.fft.fft(fid, axis=-1, norm='ortho'), axes=-1)

# ---------- helpers: numpy <-> torch ----------
def _to_torch(x: np.ndarray, device, dtype=D_TYPE) -> torch.Tensor:
    t = torch.tensor(x, device=device).unsqueeze(0).unsqueeze(0)
    return t

def _k_to_torch(x: np.ndarray, device, dtype=D_TYPE) -> torch.Tensor:
    t = torch.tensor(x, device=device).unsqueeze(0)
    return t

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    # return x.detach().cpu().numpy()
    t = x.detach().cpu()
    # drop up to two leading 1s
    shape = list(t.shape)
    while shape and shape[0] == 1:
        shape.pop(0)
        t = t.squeeze(dim=0)
    while shape and shape and t.dim() > 0 and t.shape[0] == 1:
        # if there were exactly two leading 1s
        t = t.squeeze(dim=0)
    return t.numpy()

class ToepNUFFTOp(nn.Module):
    """
    Toeplitz NUFFT operator (A^H A approx) as NumPy-friendly wrapper.

    I/O (NumPy):
      x_np: (..., C, *im_size) complex
      y_np: (..., C, *im_size) complex
    """
    def __init__(self,
                 im_size,
                 ktraj,
                 grid_size=None,
                 kernel = None,
                 smaps=None,
                 device=None,
                 norm='ortho'):
        super().__init__()

        self.im_size = tuple(im_size)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm = norm

        # ---- trajectory ----
        if isinstance(ktraj, np.ndarray):
            ktraj = torch.tensor(ktraj, device=self.device)
        else:
            ktraj = ktraj.to(self.device)
        self.register_buffer("ktraj", ktraj)

        # ---- smaps ----
        if smaps is not None:
            if isinstance(smaps, np.ndarray):
                smaps = torch.tensor(smaps, device=self.device)
            else:
                smaps = smaps.to(self.device)
        self.smaps = smaps

        # ---- operator ----
        self.toep = tkbn.ToepNufft().to(self.device)

        # ---- kernel (VERY IMPORTANT: fixed!) ----
        self.kernel = kernel
        if self.kernel == None:
            self.kernel = tkbn.calc_toeplitz_kernel(
                self.ktraj,
                im_size=self.im_size,
                grid_size=grid_size,
                norm=self.norm
            ).to(self.device)

    # ---------------- torch core ----------------
    def _T_torch(self, x_t):
        return self.toep(
            x_t,
            self.kernel,
            smaps=self.smaps,
            norm=self.norm
        )

    # ---------------- numpy API ----------------
    def T_np(self, x_np):
        x_t = _to_torch(x_np, self.device)
        y_t = self._T_torch(x_t)
        return _to_numpy(y_t)



def make_toep_linop(toep_op: ToepNUFFTOp):
    im_size = toep_op.im_size
    n = int(np.prod(im_size))

    def matvec(x):
        x_img = x.reshape(im_size)
        y_img = toep_op.T_np(x_img)
        return y_img.ravel()

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        rmatvec=matvec,   # Toeplitz ≈ Hermitian
        dtype=np.complex64
    )

def Calc_B0_matrix_mx(df_high, taxis):
    """
    Vectorized B0 modulation matrix (no explicit loops).
    df_high : (H,W) high-res Δf map [Hz]
    taxis   : (T,) time points [s]

    Returns
    -------
    B : ((H*W), T) complex ndarray
        Each row r = exp(-i 2π Δf[r] * t)
    """
    H, W = df_high.shape
    df_flat = df_high.reshape(-1, 1)     # (H*W, 1)
    t = taxis.reshape(1, -1)             # (1, T)

    # broadcasting: (H*W) * (1,T) -> (H*W,T)
    B = np.exp(-1j * 2* np.pi * df_flat * t)  
    return B.astype(D_TYPE)

def generate_M_matrices_looponr(p, r, dtype=np.float32):
    """
    Generate p selection matrices M[i], each of shape (p*r, r).

    New indexing:
        M[i][k + i*r, k] = 1
    """
    M_list = []

    for i in range(p):
        M = np.zeros((p * r, r), dtype=dtype)
        for k in range(r):
            M[k + i * r, k] = 1
        M_list.append(M)

    return M_list

''' 
Build Hessian-vector Product Operator
'''


def H_action(deltaU_mat, V, B, FHF, WtW, lambda_):
    """
    Compute H[deltaU] as matrix (m,r) without vectorization.
    Steps:
      1) deltaX = deltaU @ V.conj().T      # (m,n)
      2) BX = B * deltaX                    # (m,n) elementwise
      3) y = apply_F(BX)                    # measurement vector
      4) z = apply_FH(y)                    # (m,n)
      5) term = (B.conj() * z) @ V          # (m,r)
      6) reg  = lambda_ * (WtW @ deltaU)    # (m,r)
      7) return term + reg
    """
    # 1
    deltaX = (deltaU_mat @ V.conj().T)           # (m, n)
    # 2
    BX = (B * deltaX).ravel()                               # elementwise
    # 3 -> 4
    y = FHF @ BX                              # vector in measurement space
    z = y.reshape(B.shape)                            # (m, n)
    # 5
    term = (B.conj() * z) @ V                     # (m, r)
    # 6
    reg = lambda_ * (WtW @ deltaU_mat)            # (m, r)
    # 7
    return (term + reg).astype(D_TYPE)

# -------------------------
# 把它包装为 LinearOperator（flat interface，适配 scipy.cg 等）
# -------------------------
def make_H_linop(p, r, V, B, FHF, WtW, lambda_):
    """
    Returns scipy.sparse.linalg.LinearOperator H_linop of shape (d, d) where d = m*r.
    H_linop.matvec accepts flattened vector (length d, Fortran-order) and returns flattened H*v.
    """
    d = p * r
    def matvec(u_flat):
        deltaU = (u_flat.reshape(p, r)).astype(D_TYPE)
        out_mat = H_action(deltaU, V, B, FHF, WtW, lambda_).astype(D_TYPE)
        return (out_mat.ravel()).astype(D_TYPE)
    # dtype not forced here; scipy will infer / you can pass dtype if needed
    return LinearOperator((d, d), matvec=matvec,dtype=D_TYPE)



def calc_H_inv(x_vec: np.ndarray,
               Hess_op: Union[LinearOperator,np.ndarray],
               shape_flat: int,
               maxiter: int = 20,
               rtol: float = 1e-3,
               x0:Optional[np.ndarray]= None,
               PreconM:Union[LinearOperator,np.ndarray] = None,
               ):

    pbar = tqdm(total=maxiter)

    def _cb(xk):
        pbar.update(1)
        # current, peak = tracemalloc.get_traced_memory()
        # print(f"Current: {current/1e9:.2f} GB")
        # print(f"Peak: {peak/1e9:.2f} GB")

    if x0 is not None:
        x0 = x0
    else:
        x0 = np.zeros(shape_flat, dtype=D_TYPE)

    Hess_inv_sig, info = cg(
        Hess_op,
        x_vec,
        x0=x0,
        maxiter=maxiter,
        rtol=rtol,
        callback=_cb,
        M = PreconM,
    )
    r = Hess_op@ Hess_inv_sig - x_vec
    norm_b = np.linalg.norm(x_vec) + 1e-16
    rel_res = np.linalg.norm(r) / norm_b

    pbar.write(f'Final rel_res: {rel_res}')
    pbar.close()
    return Hess_inv_sig, info




def build_lowrank_from_Hlinop_lobpcg(H_linop, d, k=50, maxiter=70, tol=1e-6, damp=0.0, clip_eps=1e-12, M_precon = None):
    # 可选 damping
    if damp != 0.0:
        def matvec(x):
            return H_linop.matvec(x) + damp * x
        from scipy.sparse.linalg import LinearOperator
        H_use = LinearOperator(H_linop.shape, matvec=matvec, dtype=H_linop.dtype)
        print(f"[INFO] Running lobpcg on damped H (damp={damp})")
    else:
        H_use = H_linop
        print("[INFO] Running lobpcg on H (no damping)")

    # 初始块：要线性无关
    rng = np.random.default_rng()
    if np.iscomplexobj(np.zeros(1, dtype=H_linop.dtype)):
        X = (rng.standard_normal((d, k)) + 1j * rng.standard_normal((d, k))) / np.sqrt(2)
    else:
        X = rng.standard_normal((d, k))
    X = X.astype(H_linop.dtype, copy=False)

    # 找最小特征值：largest=False
    vals_raw, vecs = lobpcg(
        H_use,
        X,
        largest=False,
        maxiter=maxiter,
        tol=tol,
        M= M_precon,
    )

    # lobpcg 返回的通常已接近你要的最小端，自己再排序一下更稳
    order = np.argsort(vals_raw)
    vals_raw = np.asarray(vals_raw)[order]
    Q = np.asarray(vecs)[:, order].astype(H_linop.dtype, copy=False)

    vals_safe = np.array(vals_raw, copy=True, dtype=float)
    n_bad = np.sum(~np.isfinite(vals_safe)) + np.sum(vals_safe <= 0)
    if n_bad > 0:
        print(f"[WARN] Found {n_bad} non-positive or non-finite eigenvalues. Clipping to eps={clip_eps}")
    vals_safe[~np.isfinite(vals_safe)] = clip_eps
    vals_safe = np.maximum(vals_safe, clip_eps)

    return Q, vals_safe, vals_raw

# -------------------------
# 2) Sampler using sanitized eigenvalues (no further change)
# -------------------------
def make_lowrank_sampler_from_Q_vals_safe(Q: np.ndarray, vals_safe: np.ndarray, sigma2: float = 1.0, rng_seed: int = None):
    rng = np.random.default_rng(rng_seed)
    d, k = Q.shape
    lambda_inv_sqrt = np.sqrt(1.0 / (vals_safe + 1e-20))  # safe denominator

    def sampler(n_samples: int, batch_size: int = 100):
        samples = np.zeros((n_samples, d), dtype=D_TYPE)
        n_batches = (n_samples + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches), desc="Sampling (lowrank-safe)"):
            b0 = bi * batch_size
            b1 = min(n_samples, b0 + batch_size)
            bs = b1 - b0
            Zk = rng.standard_normal(size=(bs, k))
            low = ((Zk * lambda_inv_sqrt[None, :]) @ Q.T).astype(D_TYPE)
            if sigma2 != 1.0:
                low = low * np.sqrt(sigma2).astype(D_TYPE)
            samples[b0:b1, :] = low
        return samples
    return sampler

def laplace_samples_lowrank_only_lobpcg(H_linop, spice_est_flat, n_samples=100, k=50, sigma2=1.0,
                                        tol=1e-6, rng_seed=None, sample_batch_size=100, damp=0.0, maxiter=200, M_precon = None):
    d = H_linop.shape[0]

    Q, vals_s, vals_r = build_lowrank_from_Hlinop_lobpcg(
        H_linop, d, k=k, maxiter=maxiter, tol=tol, damp=damp, M_precon= M_precon
    )

    sampler = make_lowrank_sampler_from_Q_vals_safe(Q, vals_s, sigma2=sigma2, rng_seed=rng_seed)

    samples_zero_mean = sampler(n_samples, batch_size=sample_batch_size)
    samples = samples_zero_mean #+ spice_est_flat[None, :]

    return samples, (Q, vals_s, vals_r)


def make_Q_diag_QH_op(Q, vals, eps=1e-12):
    """
    LinearOperator for A = Q @ diag(1/vals) @ Q.H

    Parameters
    ----------
    Q : ndarray, shape (n, k) or (n, n)
    vals : ndarray, shape (k,)
        Eigenvalues / singular values / diagonal entries.
    eps : float
        Safeguard against division by zero.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
    """
    Q = np.asarray(Q)
    vals = np.asarray(vals)

    n = Q.shape[0]
    k = Q.shape[1]
    inv_vals = 1.0 / np.where(np.abs(vals) < eps, eps, vals)

    def matvec(x):
        x = np.asarray(x)
        tmp = Q.conj().T @ x          # shape (k,)
        tmp = inv_vals * tmp          # shape (k,)
        y = Q @ tmp                   # shape (n,)
        return y

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.result_type(Q.dtype, vals.dtype)
    )






# # ============================================================
# # 0) 可选：限制每个进程内部的 BLAS 线程数，避免超卖 CPU
# # ============================================================
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
# 5) 每个 worker 初始化一次
# ============================================================
def init_worker(V_, B0_mat_, WW_, #Q_, Vals_,
                ktraj_, im_size_, grid_size_, kernel_,
                NUM_SPICE_RANK_, N_VOXEL_, D_TYPE_, device_="cpu"):
    global V, B0_mat, WW, ktraj, im_size, grid_size #,Q, Vals
    global NUM_SPICE_RANK, N_VOXEL, D_TYPE
    global F1D, Gram_OP, fFHFf, Hess_op#, H_inv_lobpcg_OP

    V = V_
    B0_mat = B0_mat_
    WW = WW_
    kernel = kernel_
    # Q = Q_
    # Vals = Vals_
    ktraj = ktraj_
    im_size = tuple(im_size_)
    grid_size = tuple(grid_size_)
    NUM_SPICE_RANK = NUM_SPICE_RANK_
    N_VOXEL = N_VOXEL_
    D_TYPE = D_TYPE_

    # 这里在子进程里重新构造，不要从主进程传
    F1D = make_fft1d_op(im_size, "fid2spec")
    toep_op = ToepNUFFTOp(
        im_size=im_size,
        ktraj=ktraj,
        grid_size=grid_size,
        kernel = kernel,
        smaps=None,
        device=device_,
        norm="ortho",
        # flip_kernel=False,
    )
    Gram_OP = make_toep_linop(toep_op)
    fFHFf = F1D.H @ Gram_OP @ F1D

    # H_inv_lobpcg_OP = make_Q_diag_QH_op(Q, Vals)
    Hess_op = make_H_linop(N_VOXEL, NUM_SPICE_RANK, V, B0_mat, fFHFf, WW, Lamda_1)

# ============================================================
# 6) 单个 voxel 的工作函数
# ============================================================
def solve_one_voxel(vox_idex):
    try:
        cols = []
        # tracemalloc.start()
        for r in range(NUM_SPICE_RANK):
            # b = M[vox_idex][:, r]
            def make_b(vox_idx, r, d, dtype=D_TYPE):
                b = np.zeros(d, dtype=dtype)
                b[vox_idx * NUM_SPICE_RANK + r] = 1
                return b
            d = N_VOXEL * NUM_SPICE_RANK
            b = make_b(vox_idex, r, d)
            # x0 = H_inv_lobpcg_OP @ b
            print(f'Running cg solver on voxel{vox_idex}, rank {r}')
            x, info = calc_H_inv(
                b,
                Hess_op,
                N_VOXEL * NUM_SPICE_RANK,
                maxiter=MITER,
                rtol=1e-3,
                # x0=x0,
            )
            cols.append(x)
            X = np.array(np.column_stack(cols))
        B = np.zeros((d, NUM_SPICE_RANK), dtype=D_TYPE)
        for r in range(NUM_SPICE_RANK):
            B[vox_idex * NUM_SPICE_RANK + r, r] = 1
        os.makedirs(f_loc_hess, exist_ok=True)
        # np.save(f_loc_hess+f'vHv{vox_idex}.npy',V.conj() @ B.T @ X @ V.conj().T)
        np.save(f_loc_hess+f'mHm_{vox_idex}.npy',B.T @ X)

        print(f'Saved voxel{vox_idex}')
        # tracemalloc.stop()
        return vox_idex, np.array(cols).T
    except Exception:
        print(f"[ERROR] voxel {vox_idex}", flush=True)
        traceback.print_exc()
        raise

# ============================================================
# 7) 主程序入口
# ============================================================
def main():
    ''' 
    LOAD DATA
    '''


    print('process start.....Loading Data.....')
    wref_raw = np.load(f_loc_jlt+'wref_data.npy', mmap_mode='r')
    # wref_ksp = np.load(f_loc_jlt+'wref_ksp.npy')
    wref_img = np.load(f_loc_jlt+'wref_o.npy', mmap_mode='r')

    noise_r = np.load(f_loc_jlt+'noise.npy', mmap_mode='r')


    ''' 
    rescale and save the trejectory for nufft
    ''' 

    # mrsi_ksp[:2, ...] *= 30.37478212844472 / 32
    # mrsi_ksp          /= 2 * np.pi
    # mrsi_ksp[2,...]   -= 0.5
    # mrsi_ksp[2,...] = np.flip(mrsi_ksp[2,...])

    # np.save(f_loc_jlt + 'mrsi_ksp_scaled.npy', mrsi_ksp)
    mrsi_ksp_scaled = np.load(f_loc_jlt + 'mrsi_ksp_scaled.npy', mmap_mode='r')
    # del mrsi_ksp

    print(np.max(mrsi_ksp_scaled[:2, ...]))
    print(np.min(mrsi_ksp_scaled[:2, ...]))
    print(np.max(mrsi_ksp_scaled[2, ...]))
    print(np.min(mrsi_ksp_scaled[2, ...]))


    trej_mrsi = mrsi_ksp_scaled.T.astype(Trej_D_TYPE)

    ''' 
    LOAD COIL SMAP
    '''
    coil_smap = np.load(f_loc_jlt+'ecalib.npy', mmap_mode='r')
    print(f'coilsmap shape: {coil_smap.shape}')
    NUM_CMAP_CHANNEL = coil_smap.shape[-1]
    coil_smap = (np.moveaxis(coil_smap, -1, 0).squeeze()).astype(D_TYPE)
    print(f'coilsmap shape: {coil_smap.shape}')
    coil_smap = np.repeat(coil_smap[None, :, :, :, None], N_SEQ_POINTS, axis=-1)   # (1, C, Ny, Nx, Nt)
    coil_smap = coil_smap.astype(D_TYPE, copy=False)
    print(f'coilsmap shape: {coil_smap.shape}')

    smap_time = coil_smap.squeeze(0)                  # (n_coils, Ny, Nx, T)
    print(f'smap_time shape: {smap_time.shape}')

    '''
    Build Torchnufft operator
    '''
    print('Data Loaded!')
    print('Building Fourier Operator.....')
    # image dims
    im_size = (Dim_Voxel[0],Dim_Voxel[1],N_SEQ_POINTS)
    Ny, Nx, T = (Dim_Voxel[0], Dim_Voxel[1], N_SEQ_POINTS)
    osamp = 2.0
    ost = 2.0
    grid_size = (int(np.ceil(osamp * Ny)), int(np.ceil(osamp * Nx)), int(np.ceil(ost * T)))

    # Keep float32 for memory / GPU
    ktraj_torch = torch.from_numpy(trej_mrsi)
    ktraj_flat = ktraj_torch.permute(2, 0, 1).reshape(3,-1)  # shape (3, M_total)
    # ktraj_flat = ktraj_torch.reshape(3,-1)  # shape (3, M_total)

    # # convert k-space trajectory to a tensor
    ktraj = torch.tensor(ktraj_flat, device=device)
    print('ktraj shape: {}'.format(ktraj.shape))

    smap = torch.tensor(coil_smap, device=device).to(T_D_TYPE) # shape (n_batch,n_coil,x,y, n_seq_points)
    print('smap shape: {}'.format(smap.shape))



    tnufft_ob = tkbn.KbNufft(im_size=(Ny, Nx, T), grid_size=grid_size,dtype=T_D_TYPE).to(device)
    tadjnufft_ob = tkbn.KbNufftAdjoint(im_size=(Ny, Nx, T), grid_size=grid_size,dtype=T_D_TYPE).to(device)

    NUM_CMAP_CHANNEL = coil_smap.shape[1]

    F_tkbn = NUFFTOp(im_size=im_size,
                grid_size=grid_size,        # None or ~1.5–2.0× spatial dims
                omega=ktraj,             # numpy OK
                smaps=coil_smap,#None, #coil_smap,             # optional numpy/torch
                dcf=None,                 # optional numpy
                norm='ortho',
                device=device,
                nufft_ob=tnufft_ob,
                adjnufft_ob=tadjnufft_ob)

    print(F_tkbn.A)
    print(F_tkbn.AH)

    F_OP = make_linop(F_tkbn,Dim_Voxel,K_POINTS,N_SHOTs,N_SEQ_POINTS,NUM_CMAP_CHANNEL) 
    F1D = make_fft1d_op(im_size,'fid2spec')

    osamp = 2.0
    ost =2.0
    toep_grid_size = (int(np.ceil(osamp * Ny)), int(np.ceil(osamp * Nx)), int(np.ceil(ost * T)))
    kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size, grid_size=toep_grid_size, norm = 'ortho')


    toep_op = ToepNUFFTOp(
        im_size=im_size,
        ktraj=ktraj,
        grid_size=toep_grid_size,
        kernel =kernel,
        smaps=smap,
        norm='ortho'
    )

    Gram_OP = make_toep_linop(toep_op)
    print('Building Fourier Operator.....Done!')

    '''
    Take the svd of the data
    '''
    print('Taking SVD.....')
    # Read the training data from the csv
    training_dataset = read_training_data_from_csv(save_dir=SAVE_DIR,filename=CSV_FILE_NAME).astype(D_TYPE)


    u,s,vh = np.linalg.svd(training_dataset)
    print('SVD Done')

    '''  
    Prior mask processing
    '''

    print('SPICE Pre-steps.....')

    wref_img_2d = wref_img.squeeze(-1)
    # water_rou_2D = np.abs(wref_img_2d)*100

    #normalize wref map to 0-1
    water_rou_2D = np.abs(wref_img_2d)

    wrefmin = water_rou_2D.min()
    wrefmax = water_rou_2D.max()

    water_rou_2D_norm = (water_rou_2D - wrefmin) / (wrefmax - wrefmin + 1e-12)

    #2. Calculate Edge Preserving Matrix
    # calculate the edge preserving matrix W_edge for given constraint
    minpooling_Handler = True # use min pooling
    pool_size = 1 # can adjust the pool size here
    W_max = LAMBDA_WE_max
    lamda_1 = Lamda_1
    # W_edge, _W, _P = constraints_to_B(water_rou_1D, W_max=W_max, pool_size=pool_size, minpooling_Handler = minpooling_Handler) 



    ''' 
    Load B0 map and post-process for global ppm-shift compared to basis-set
    '''

    B0_map = np.load(f_loc_jlt+'B0_map.npy')
    B0_map_0s = np.nan_to_num(-B0_map, nan=0.0) #change nan in b0 to 0 for global b0 corr
    B0_mat = Calc_B0_matrix_mx(B0_map_0s,TIME_AXIS)

    ''' 
    extract brain region
    '''
    # ---------- threshold ----------
    brain_region_threshold = 0.08#0.00035
    # -----------------------------------------------

    img = water_rou_2D_norm

    # brain mask
    brain_mask = img > brain_region_threshold   # True / False

    from scipy.ndimage import binary_erosion

    # 去掉外围 4 圈
    brain_mask_inner = binary_erosion(brain_mask, iterations=3)



    ''' 
    edge preserving and vh calculation
    '''
    W_edge, A, _W, Nb = calc_Bmatrix(water_rou_2D_norm, wmax=5e3, adj=8, pool_size=pool_size,minpooling_Handler=minpooling_Handler,brain_mask=brain_mask,mask_dilate_layers=3)
    # Plot_W_WE(_W, _P)
    # Nb is shape (N_edges, 2), which has to be turned into a list of tuples before it can be fed into the plot function.


    edge_index = [tuple(pair) for pair in Nb]

    # visualisation
    # plot_anatomical_mask_points_size(mask=_W, anatomical_prior=water_rou_1D, edge_index=edge_index)
    # plot_anatomical_mask_points_size_directional(mask=_W, anatomical_prior=water_rou_2D_norm, edge_index=edge_index)

    V = vh[0:NUM_SPICE_RANK, :].conj().T
    Vs = FID2Spec(vh[0:NUM_SPICE_RANK,:].T).conj()
    Vh = vh[0:NUM_SPICE_RANK, :]
    Vsh = Vs.conj().T

    # GT_KT_SPACE = gen_gt_ktspace(GT_IT_SPACE,K_POINTS,F,TIME_AXIS,True)

    WW = W_edge.conj().T @ W_edge

    
    # M = generate_M_matrices_looponr(N_VOXEL,NUM_SPICE_RANK)
    fFHFf = F1D.H @ Gram_OP @ F1D
    Hess_op = make_H_linop(N_VOXEL,NUM_SPICE_RANK,V,B0_mat,fFHFf,WW,Lamda_1)

    # ############### comented LOBPCG #################

    # print('Starting LOBPCG.....')
    # lac_samples,others = laplace_samples_lowrank_only_lobpcg(Hess_op,est_U.ravel(),10,40,sigma2=1,tol=1e-3,damp = 0.0,maxiter=50)  #16,30

    # Q_,Vals_,_ = others
    # sampler = make_lowrank_sampler_from_Q_vals_safe(Q_, Vals_, sigma2=100, rng_seed=None)

    # samples_zero_mean = sampler(100, batch_size=100)
    # lac_samples = samples_zero_mean #+ spice_est_cg.resha[None, :]
    # allsample_U = lac_samples.reshape(100,est_U.shape[0],est_U.shape[1])
    # sim_spice = allsample_U @ Vh
    # _ = plot_voxel_spectrum_and_maps(np.std(fid_to_spec(sim_spice),axis = 0),im_size,27,37,0,0,brain_mask_inner,PPM_AXIS)
    # H_inv_lobpcg_OP = make_Q_diag_QH_op(Q_,Vals_)
    # print('LOBPCG Done!')
    # ############### comented LOBPCG #################

    # print('trying timing...')
    # import time
    # d = N_VOXEL * NUM_SPICE_RANK
    # x = (np.random.randn(d) + 1j*np.random.randn(d)).astype(D_TYPE)

    # t0 = time.perf_counter()
    # y = Hess_op @ x
    # print("Hess_op @ x:", time.perf_counter() - t0, "s")

    # BX = (B0_mat * (x.reshape(N_VOXEL, NUM_SPICE_RANK) @ V.conj().T)).ravel()
    # t0 = time.perf_counter()
    # z = fFHFf @ BX
    # print("FHF @ BX:", time.perf_counter() - t0, "s")

    # t0 = time.perf_counter()
    # w = Gram_OP @ np.random.randn(np.prod(im_size)).astype(D_TYPE)
    # print("Gram_OP @ x:", time.perf_counter() - t0, "s")
    


    print('Starting Uncertainty Quant.....')
    vox_list = np.flatnonzero(brain_mask.ravel())[540:600]
    print('Total voxel count:', len(vox_list))
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=init_worker,
        initargs=(
            V, B0_mat, WW, #Q_, Vals_,
            ktraj, im_size, grid_size,kernel,
            NUM_SPICE_RANK, N_VOXEL, D_TYPE,
        ),
    ) as ex:
        # results = list(ex.map(solve_one_voxel, vox_list))
        for _ in ex.map(solve_one_voxel, vox_list):
            pass

    # H_inv_mul_M_all = {vox: mat for vox, mat in results}
    return 0

if __name__ == "__main__":
    out = main()
"""Checks for learned_spice: graph construction, solver, export, noise propagation, apply CLI."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree

from learned_spice.data import SimConfig, relative_error, simulate_sample, to_batch
from learned_spice.model import (EdgeWeightNet, export_system, handcrafted_weights, laplacian,
                                 learned_solve, spice_solve, spice_system, voxel_noise_std)
from learned_spice.phantom import spectral_subspace, training_fids_pk2pi

REPO = Path(__file__).resolve().parents[2]
CFG = SimConfig(n_spec=128)


def _subspace():
    return spectral_subspace(training_fids_pk2pi(CFG.n_spec, CFG.dwell_s, CFG.larmor_mhz, 500), CFG.rank)


def _calc_bmatrix_reference(ref_img, wmax=5e4):
    """Core of calc_Bmatrix_simple (05c), 8-connected, without masks."""
    xy = np.indices(ref_img.shape).reshape((2, -1)).T
    Nb = KDTree(xy).query_pairs(p=2, r=1.415, output_type="ndarray")
    ref_vec = ref_img.flatten()
    C = (np.diff(ref_vec[Nb], axis=1)**2).flatten()
    with np.errstate(divide="ignore"):
        W = np.minimum(1.0 / C, wmax)
    A = csr_matrix((W, (Nb[:, 0], Nb[:, 1])), shape=(ref_vec.size,) * 2).toarray()
    A = 0.5 * (A + A.T)
    return np.diag(A.sum(0)) + np.diag(A.sum(1)) - 2.0 * A


def test_laplacian_matches_calc_bmatrix_simple():
    ref = np.random.default_rng(0).uniform(size=(7, 7))
    L = laplacian(torch.as_tensor(handcrafted_weights(ref)[None]))[0].numpy()
    np.testing.assert_allclose(L, _calc_bmatrix_reference(ref), rtol=1e-10, atol=1e-8)


def test_gram_is_hermitian_psd():
    s = simulate_sample(CFG, _subspace(), np.random.default_rng(1), 8)
    G = s["G"]
    np.testing.assert_allclose(G, G.conj().T, atol=1e-10)
    assert np.linalg.eigvalsh(G).min() > -1e-8
    assert (s["dcf"] > 0).all()


def test_export_reproduces_learned_solve(tmp_path):
    V = _subspace()
    s = simulate_sample(CFG, V, np.random.default_rng(2), 8)
    b = to_batch([s])
    net = EdgeWeightNet()
    with torch.no_grad():
        U, w, tau = learned_solve(net, b["feats"], b["G"], b["B"])
    export_system(tmp_path / "sys.npz", s["G"], w[0].numpy(), tau[0].numpy())
    sysf = np.load(tmp_path / "sys.npz")
    H = sysf["G"] + sysf["lam"] * sysf["WW"] + np.diag(sysf["lout_map"].ravel())
    np.testing.assert_allclose(np.linalg.solve(H, s["B"]), U[0].numpy(), rtol=1e-6, atol=1e-8)


def test_voxel_noise_std_matches_monte_carlo():
    V = _subspace()
    rng = np.random.default_rng(3)
    s = simulate_sample(CFG, V, rng, 8)
    n2 = s["G"].shape[0]
    w = torch.full((1, 4, 8, 8), 0.05, dtype=torch.float64)
    tau = torch.full((1, n2), 1e-3, dtype=torch.float64)
    H = spice_system(torch.as_tensor(s["G"])[None], w, tau)[0].numpy()
    pred = voxel_noise_std(H, s["E"], s["dcf"], sigma_k=1.0)
    nk = s["E"].shape[0]
    noise = (rng.standard_normal((nk, 4000)) + 1j * rng.standard_normal((nk, 4000))) / np.sqrt(2)
    U = np.linalg.solve(H, s["E"].conj().T @ (s["dcf"][:, None] * noise))
    np.testing.assert_allclose(U.std(1), pred, rtol=0.08)


def test_gradients_flow_and_loss_is_finite():
    V = _subspace()
    rng = np.random.default_rng(4)
    b = to_batch([simulate_sample(CFG, V, rng, 10) for _ in range(3)])
    net = EdgeWeightNet()
    U, _, _ = learned_solve(net, b["feats"], b["G"], b["B"])
    loss = relative_error(U, b["U_true"]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_apply_cli_on_simulated_dynamic_data(tmp_path):
    V = _subspace()
    s = simulate_sample(CFG, V, np.random.default_rng(5), 8)
    torch.save({"state_dict": EdgeWeightNet().state_dict()}, tmp_path / "model.pt")
    np.savez(tmp_path / "system.npz", G=s["G"], prior=s["prior"] if s["prior"] is not None else s["pcr_true"])
    adj = np.stack([s["B"] @ V.conj().T] * 3)                        # 3 identical "reps"
    np.save(tmp_path / "adj.npy", adj)
    np.save(tmp_path / "V.npy", V)
    np.save(tmp_path / "leg.npy", s["leg"].astype(int))
    subprocess.run([sys.executable, "-m", "learned_spice.apply", "--model", str(tmp_path / "model.pt"),
                    "--system", str(tmp_path / "system.npz"), "--use-prior", "--adj-fid", str(tmp_path / "adj.npy"),
                    "--V", str(tmp_path / "V.npy"), "--leg", str(tmp_path / "leg.npy"), "--noise-std-b", "1e-2",
                    "--out-dir", str(tmp_path / "out")], check=True, cwd=REPO)
    fid = np.load(tmp_path / "out" / "fid_learned.npy")
    assert fid.shape == adj.shape and np.isfinite(fid).all()

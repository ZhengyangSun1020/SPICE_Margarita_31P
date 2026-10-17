"""Verify the direct-DFT operator matches the finufft convention (same orientation)."""
import numpy as np, importlib.util as ilu
s=ilu.spec_from_file_location("m","scripts/06c_iterative_recon_31p_3d.py")
m=ilu.module_from_spec(s); s.loader.exec_module(m)
import mrinufft
info=m.load_concept_twix_3d("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
                            "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", rep_sel=0)
N=info["matrix_size"]; Nz=1
traj=m.build_trajectory_3d(info["first_circle"], info["n_rings_total"], info["traj_pts"],
                           info["kz_cpp_per_ring"], info["gamma_2pi"]/1e9, info["fov_m"], N)
E=m.build_dft_matrix(traj, N, N, Nz)
op=mrinufft.get_operator("finufft", samples=traj, shape=(N,N,Nz), n_coils=1, density=False)
rng=np.random.default_rng(0)
img=(rng.standard_normal((N,N,Nz))+1j*rng.standard_normal((N,N,Nz))).astype(np.complex64)
# forward
y_nufft=op.op(img).ravel()
y_dft=E@img.ravel()
def corr(a,b):
    a=a.ravel(); b=b.ravel(); return np.abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
print("forward  corr(nufft, dft) =", round(float(corr(y_nufft,y_dft)),4))
# adjoint
k=(rng.standard_normal(len(traj))+1j*rng.standard_normal(len(traj))).astype(np.complex64)
x_nufft=op.adj_op(k).ravel()
x_dft=(E.conj().T@k).ravel()
print("adjoint  corr(nufft, dft) =", round(float(corr(x_nufft,x_dft)),4))
print("scale ratio |nufft|/|dft| (adjoint) =", round(float(np.linalg.norm(x_nufft)/(np.linalg.norm(x_dft)+1e-12)),4))

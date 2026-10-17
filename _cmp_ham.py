import numpy as np, importlib.util as ilu
s=ilu.spec_from_file_location("m","scripts/06c_iterative_recon_31p_3d.py")
m=ilu.module_from_spec(s); s.loader.exec_module(m)
info=m.load_concept_twix_3d("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
                            "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", rep_sel=0)
N=info["matrix_size"]
traj=m.build_trajectory_3d(info["first_circle"], info["n_rings_total"], info["traj_pts"],
                           info["kz_cpp_per_ring"], info["gamma_2pi"]/1e9, info["fov_m"], N)
r=np.sqrt(traj[:,0]**2+traj[:,1]**2)
rings=np.sort(np.unique(np.round(r,6)))
# python ham radius (06c): r_xy * (k_norm_max_phys/0.5)
ham_r = r*(float(info["k_norm_max_phys"])/0.5)
py_ham = 0.54+0.46*np.cos(2*np.pi*ham_r)
# matlab ham: 0.54+0.46*cos(pi*r/max(r))
ml_ham = 0.54+0.46*np.cos(np.pi*r/r.max())
print(f"{'ring_r':>8} {'PY_HAM':>8} {'MATLAB_HAM':>11}")
for rr in rings:
    mask=np.abs(r-rr)<1e-5
    print(f"{rr:8.4f} {py_ham[mask].mean():8.3f} {ml_ham[mask].mean():11.3f}")
print(f"\nouter-ring taper:  Python -> {py_ham[np.abs(r-rings[-1])<1e-5].mean():.3f}   "
      f"MATLAB -> {ml_ham[np.abs(r-rings[-1])<1e-5].mean():.3f}")

import numpy as np, importlib.util as ilu
s=ilu.spec_from_file_location("m","scripts/05c_spice_recon_31p.py")
m=ilu.module_from_spec(s); s.loader.exec_module(m)
info=m.load_concept_twix_3d("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
                            "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", rep_sel=0)
ppm = np.asarray(info["ppm_axis"])
a=np.load("output_dynamic_3T_0724_5rings_06c/data/iter3d_recon_spec_rep30.npy")[:,:,0,:]
z=np.abs(a[4,5])
print("=== 06c STORED ppm_axis ===")
print("ppm[0]=%.2f  ppm[-1]=%.2f  increasing L->R? %s" % (ppm[0], ppm[-1], ppm[0]<ppm[-1]))
idx=np.argsort(z)[::-1][:8]
print("Top peaks of voxel[4,5] rep30 (STORED axis):")
for i in sorted(idx, key=lambda i:-z[i]):
    print("   %+6.2f ppm   amp=%.2e" % (ppm[i], z[i]))
# where is PCr (largest)?
pk=idx[np.argmax(z[idx])]
print("PCr (max) at %.2f ppm" % ppm[pk])

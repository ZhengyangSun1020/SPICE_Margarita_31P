import numpy as np, importlib.util as ilu
s=ilu.spec_from_file_location("m","scripts/05c_spice_recon_31p.py")
m=ilu.module_from_spec(s); s.loader.exec_module(m)
info=m.load_concept_twix_3d("/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_CRT_0724/"
                            "meas_MID00277_FID221482_240FOV_10x10_5rings.dat", rep_sel=0)
ppm=np.asarray(info["ppm_axis"])
a=np.load("output_dynamic_3T_0724_5rings_06c/data/iter3d_recon_spec_rep30.npy")[:,:,0,:]
def side(ix,iy):
    z=np.abs(a[ix,iy])
    out = ~((ppm>-2)&(ppm<2))          # exclude PCr
    i=np.argmax(z*out)
    # amps near +5 and -5
    ip=np.argmin(np.abs(ppm-4.85)); im=np.argmin(np.abs(ppm+4.85))
    print("[%d,%d]: biggest side-peak at %+.2f ppm (amp %.2e) | amp@+4.85=%.2e amp@-4.85=%.2e"
          % (ix,iy, ppm[i], z[i], z[ip], z[im]))
for (ix,iy) in [(4,5),(5,4),(4,4)]:
    side(ix,iy)

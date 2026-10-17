import sys, os, numpy as np, nibabel as nib
d=sys.argv[1]  # fit dir containing concs/internal
maskp=sys.argv[2]
mk=np.squeeze(np.asarray(nib.load(maskp).dataobj)).astype(bool)
print(f"--- {d} (median over {int(mk.sum())} mask voxels, rel PCr) ---")
for m in ["PCr","Pi","Pi2","PDE","ATP_gamma","ATP_alpha","ATP_beta"]:
    p=os.path.join(d,"concs","internal",m+".nii.gz")
    if os.path.exists(p):
        v=np.squeeze(np.asarray(nib.load(p).dataobj))
        print(f"  {m:12s} = {float(np.nanmedian(v[mk])):.3f}")

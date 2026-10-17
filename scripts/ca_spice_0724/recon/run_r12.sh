#!/bin/bash
# CC-SPICE rank 12 (first 12 columns of VD), mu 30, drawn masks, all four 24-07 bouts; FIDs + spectra saved
cd /home/zsun/SPICE_Margarita/scripts; PY=/home/zsun/miniconda3/envs/zs_SPICE_MARGARITA/bin/python
S=/home/zsun/SPICE_Margarita/scripts/ca_spice_0724/recon   # hybrid_slam_spice.py
for d in 283 279 281 277; do
  O=/home/zsun/SPICE_Margarita/output_study_mid$d
  $PY -c "import numpy as np; np.save('$O/VD12.npy', np.load('$O/VD.npy')[:, :12])"
  VFILE=VD12.npy PIXMASK=1 LABELS=mask MUS=30 SAVE_FID=30 SAVE_SPEC=1 HOUT=hybrid_v2_r12 $PY $S/hybrid_slam_spice.py $d > /mnt/e/claude_tmp/study/fsl31p/r12_$d.out 2>&1
  echo "== MID00$d exit $?"; grep -i -E "traceback|error" -A5 /mnt/e/claude_tmp/study/fsl31p/r12_$d.out | tail -6
  echo "  r12: $(grep -E '^hybrid μ 30 ' $O/hybrid_v2_r12/log.txt)"; echo "  r16: $(grep -E '^hybrid μ 30 ' $O/hybrid_v2/log.txt)"
done
echo ALL DONE

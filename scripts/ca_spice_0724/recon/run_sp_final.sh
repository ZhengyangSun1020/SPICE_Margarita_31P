#!/bin/bash
# Final compartment-aware SPICE for the abstract: CSI grid, hard compartment boundary, edited voxel labels, rank 12, published per-bout lambda; save voxel FIDs
cd /home/zsun/SPICE_Margarita/scripts; PY=/home/zsun/miniconda3/envs/zs_SPICE_MARGARITA/bin/python
S=/home/zsun/SPICE_Margarita/scripts/ca_spice_0724/recon   # hybrid_slam_spice.py
for d in 283 279 281 277; do
  O=/home/zsun/SPICE_Margarita/output_study_mid$d
  COMP_BOUND=hard SAVE_SP=1 VFILE=VD12.npy PIXMASK=1 LABELS=mask MUS=30 SAVE_SPEC=1 HOUT=spice_final $PY $S/hybrid_slam_spice.py $d > /mnt/e/claude_tmp/study/fsl31p/spfinal_$d.out 2>&1
  echo "== MID00$d exit $?"; grep -i -E "traceback|error" -A5 /mnt/e/claude_tmp/study/fsl31p/spfinal_$d.out | tail -6
  grep -E "saved compartment|^NUFFT |^SPICE \(comp" $O/spice_final/log.txt | cut -c1-120
done
echo ALL DONE

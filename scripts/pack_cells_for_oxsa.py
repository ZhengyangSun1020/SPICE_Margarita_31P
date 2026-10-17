#!/usr/bin/env python3
"""Package SPICE recon FIDs into a MATLAB `cells.mat` for Leg_eval_twix.m (OXSA/AMARES).

Leg_eval_twix only READS fields off the Spectro.PlotCsi objects, so we mimic the
accessed structure with plain nested structs / cells — no real OXSA object needed.

Per acquisition (rep) it builds:
  cells{1,acq}.data.spec.signals{1,1}   = FID  [nPts x nx x ny]  (complex)
  cells{1,acq}.data.spec.timeAxis       = [nPts x 1]
  cells{1,acq}.data.spec.ppmAxis        = [nPts x 1]
  cells{1,acq}.data.spec.imagingFrequency (MHz)
  cells{1,acq}.data.spec.dwellTime      (s)
  cells{1,acq}.data.spec.info{1,1}      = metadata struct
      .PatientSex/.PatientAge/.PatientSize/.PatientWeight
      .csa.MrPhoenixProtocol.alTR{1,1}          (ms; script divides by 1000 -> s)
      .csa.MrPhoenixProtocol.adFlipAngleDegree{1,1}
      .csa.MrPhoenixProtocol.lScanTimeSec       (s per acquisition)

Metadata is pulled straight from the twix header.
"""
import os, glob, numpy as np
from scipy.io import savemat

ROOT = "/home/zsun/SPICE_Margarita"
# Defaults kept so calling with no args still packs MID00644 (original behaviour).
DEFAULT_DYN  = os.path.join(ROOT, "output_MID00644_dyn_r10")
DEFAULT_DAT  = "/mnt/f/AliceConceptB1/in_vivo_3T/dynamic_crt_10_10/meas_MID00644_FID214692_LV_crt_dynamic_10x10_5rings.dat"
DEFAULT_RANK = 10
DEFAULT_OUT  = os.path.join(ROOT, "cells_MID00644_r10.mat")

# If AMARES puts ATP at +16 instead of -16 (mirrored), flip this to False.
CONJ_FID = True   # feed conj(spice_fid): matches spice_spec = FIDToSpec(fid.conj())


def twix_meta(dat_path):
    import importlib.util as ilu, mapvbvd
    spec = ilu.spec_from_file_location('_06c', os.path.join(ROOT, 'scripts', '06c_iterative_recon_31p_3d.py'))
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    tw = mapvbvd.mapVBVD(dat_path); tw = tw[-1] if isinstance(tw, list) else tw
    hdr = tw['hdr']; H = mod._hdr_float
    larmor_mhz = H(hdr, 'MeasYaps', ('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency'), default=49894571) / 1e6
    adc_dwell_s = H(hdr, 'MeasYaps', ('sRXSPEC', 'alDwellTime', '0'), default=180000) * 1e-9
    spec_dwell_s = adc_dwell_s * 2.0                      # spectral dwell after 2x OS removal (=FID dwell)
    tr_s = H(hdr, 'MeasYaps', ('alTR', '0'), default=2000000) / 1e6
    fa = H(hdr, 'MeasYaps', ('adFlipAngleDegree', '0'), default=42)
    scan_s = H(hdr, 'MeasYaps', ('lScanTimeSec',), default=14)
    d = hdr.get('Meas', {}) if hasattr(hdr, 'get') else {}
    return dict(larmor_mhz=larmor_mhz, spec_dwell_s=spec_dwell_s, tr_s=tr_s, fa=float(fa),
                scan_s=float(scan_s), weight=float(d.get('flUsedPatientWeight', 67.0)),
                age=float(d.get('flPatientAge', 25.0)), sex=int(float(d.get('PatientSex', 2.0))),
                size=float(d.get('flPatientHeight', 1.75) or 1.75))


def cell(x):
    c = np.empty((1, 1), dtype=object); c[0, 0] = x; return c


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dat', default=DEFAULT_DAT)
    p.add_argument('--dyn', default=DEFAULT_DYN, help='parent dir holding repNN/data/spice_fid_rank*.npy')
    p.add_argument('--rank', type=int, default=DEFAULT_RANK)
    p.add_argument('--out', default=DEFAULT_OUT, help='.mat output path')
    args = p.parse_args()
    DAT, DYN, RANK, OUT = args.dat, args.dyn, args.rank, args.out

    m = twix_meta(DAT)
    fids = sorted(glob.glob(os.path.join(DYN, 'rep*', 'data', f'spice_fid_rank{RANK}.npy')))
    reps = [int(os.path.basename(os.path.dirname(os.path.dirname(p))).replace('rep', '')) for p in fids]
    nAcq = len(reps)
    s0 = np.load(fids[0]); ny0, nx0, nPts = s0.shape
    PPM = -np.fft.fftshift(np.fft.fftfreq(nPts, d=m['spec_dwell_s'])) / m['larmor_mhz']
    timeAxis = (np.arange(nPts) * m['spec_dwell_s']).reshape(-1, 1)
    ppmAxis = PPM.reshape(-1, 1)

    mrpp = {'alTR': cell(m['tr_s'] * 1000.0),          # ms  -> Leg_eval does /1000 -> s
            'adFlipAngleDegree': cell(m['fa']),
            'lScanTimeSec': m['scan_s']}
    info = {'PatientSex': str(m['sex']), 'PatientAge': f"{int(m['age']):03d}Y",
            'PatientSize': m['size'], 'PatientWeight': m['weight'],
            'csa': {'MrPhoenixProtocol': mrpp}}

    cells = np.empty((1, nAcq), dtype=object)
    for k, r in enumerate(reps):
        fid = np.load(os.path.join(DYN, f'rep{r:02d}', 'data', f'spice_fid_rank{RANK}.npy'))  # [yi,xi,ns]
        if CONJ_FID:
            fid = np.conjugate(fid)
        signals = np.ascontiguousarray(fid.transpose(2, 1, 0).astype(np.complex128))          # [nPts, nx(xi), ny(yi)]
        spec = {'signals': cell(signals), 'timeAxis': timeAxis, 'ppmAxis': ppmAxis,
                'imagingFrequency': float(m['larmor_mhz']), 'dwellTime': float(m['spec_dwell_s']),
                'info': cell(info)}
        cells[0, k] = {'data': {'spec': spec}}

    savemat(OUT, {'cells': cells}, do_compression=True, long_field_names=True)
    print(f"wrote {OUT}")
    print(f"  nAcq={nAcq}  nx={nx0}  ny={ny0}  nPts={nPts}   (MATLAB ix=xi, iy=yi; voxel id = iy0*10+ix0)")
    print(f"  meta: {m}")


if __name__ == "__main__":
    main()

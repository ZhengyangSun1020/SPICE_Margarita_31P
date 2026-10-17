#!/usr/bin/env python3
"""Inspect CRT launch-track ADC structure (counters only, no full data pull)."""
import sys, numpy as np
import mapvbvd

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/260911_yife_liver_launchtrack_BS/meas_MID00057_FID98135_ZS_crt_dualband_sat_BS_Launchtrack_1000.dat"

tw = mapvbvd.mapVBVD(dat, quiet=True)
if isinstance(tw, list):
    tw = tw[-1]

try:
    cats = list(tw.keys())
except Exception:
    cats = [a for a in ['image','noise','phasecor','refscan','refscanPC'] if a in tw]
print("twix categories:", cats)

img = tw['image']

def show(o, name):
    print(f"\n=== {name} dims ===")
    for a in ['NCol','NCha','NLin','NPar','NSli','NAve','NSeg','NRep','NSet','NIda','NIdb','NIdc','NIdd','NAcq']:
        print(f"  {a:6s} = {getattr(o, a, '?')}")

show(img, "image")

Lin = img.Lin.astype(int); Par = img.Par.astype(int); Seg = img.Seg.astype(int)
Ave = img.Ave.astype(int); Rep = img.Rep.astype(int); Sli = img.Sli.astype(int)
Set = img.Set.astype(int)
Ida = img.Ida.astype(int); Idb = img.Idb.astype(int); Idc = img.Idc.astype(int); Idd = img.Idd.astype(int)
ice = img.iceParam

nacq = len(Lin)
print(f"\nnAcq (ADC events) = {nacq}")
for nm, arr in [('Lin',Lin),('Par',Par),('Seg',Seg),('Ida',Ida),('Idb',Idb),
                ('Idc',Idc),('Idd',Idd),('Set',Set),('Ave',Ave),('Rep',Rep)]:
    u = np.unique(arr)
    print(f"  {nm:4s}: {u.min()}..{u.max()}  (nuniq={len(u)})  first20={u[:20].tolist()}")

# try to get per-ADC sample count
for attr in ['NCol','freeParam','sz','memPos']:
    v = getattr(img, attr, None)
    print(f"img.{attr}: {type(v).__name__} shape={getattr(v,'shape',None)}")

print("\n=== first 48 ADC events (acquisition order) ===")
print("  idx  Lin Par Seg Ida Idb Idc Idd Set Rep | ice0 ice1 ice2 ice3 ice4 ice5 ice6 ice7")
for s in range(min(48, nacq)):
    print(f"  {s:4d} {Lin[s]:4d}{Par[s]:4d}{Seg[s]:4d}{Ida[s]:4d}{Idb[s]:4d}{Idc[s]:4d}{Idd[s]:4d}{Set[s]:4d}{Rep[s]:4d} | "
          + "".join(f"{int(ice[s,j]):6d}" for j in range(8)))

# Look for a distinct "first ADC per TR". Hypothesis: launch track has Idc/Idd differing,
# or Seg toggling. Count ADCs per (Rep,Lin,Par) group in first rep.
print("\n=== ADC count per (Lin,Par) in Rep==min ===")
r0 = Rep.min()
m = Rep == r0
import collections
cnt = collections.Counter(zip(Lin[m].tolist(), Par[m].tolist()))
items = sorted(cnt.items())[:12]
for (l,p),c in items:
    print(f"  Lin={l:3d} Par={p:3d} -> {c} ADCs")
print(f"  total groups in rep{r0}: {len(cnt)}")

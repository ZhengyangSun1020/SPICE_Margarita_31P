#!/usr/bin/env python3
"""Inspect interleaved FLASH-3dCRT: categories + phasecor/image MDH layout (counters only)."""
import os, sys, numpy as np
sys.path.insert(0, "/home/zsun/SPICE_Margarita/scripts")
from twix_fast import accelerate_mdh_read
accelerate_mdh_read()
import mapvbvd

dat = "/mnt/f/AliceConceptB1/in_vivo_data_7T/07092026/meas_MID00054_FID97691_ZS_FLASH_3dCRT_1Hpos_242424_500.dat"

tw = mapvbvd.mapVBVD(dat, quiet=True)
if isinstance(tw, list): tw = tw[-1]
try:
    cats = list(tw.keys())
except Exception:
    cats = [a for a in ['hdr','image','noise','phasecor','refscan'] if a in tw]
print("twix categories:", cats)

def dims(o, name):
    print(f"\n=== {name} dims ===")
    for a in ['NCol','NCha','NLin','NPar','NSli','NAve','NSeg','NRep','NSet',
              'NIda','NIdb','NIdc','NIdd','NAcq']:
        print(f"  {a:6s} = {getattr(o, a, '?')}")

def counters(o, name, nshow=40):
    print(f"\n=== {name}: MDH counters ===")
    Lin=o.Lin.astype(int); Par=o.Par.astype(int); Seg=o.Seg.astype(int)
    Ida=o.Ida.astype(int); Idb=o.Idb.astype(int); Idc=o.Idc.astype(int)
    Idd=o.Idd.astype(int); Set=o.Set.astype(int); Rep=o.Rep.astype(int)
    Ave=o.Ave.astype(int)
    ice=o.iceParam
    n=len(Lin)
    print(f"  NAcq={n}")
    for nm,a in [('Lin',Lin),('Par',Par),('Seg',Seg),('Ida',Ida),('Idb',Idb),
                 ('Idc',Idc),('Idd',Idd),('Set',Set),('Ave',Ave),('Rep',Rep)]:
        u=np.unique(a)
        print(f"  {nm:4s}: {a.min()}..{a.max()} nuniq={len(u)} first15={u[:15].tolist()}")
    print(f"  --- first {nshow} ADC events ---")
    print("  idx  Lin Par Seg Ida Idb Idc Idd Set Rep | ice0..7")
    for s in range(min(nshow,n)):
        print(f"  {s:4d} {Lin[s]:4d}{Par[s]:4d}{Seg[s]:6d}{Ida[s]:4d}{Idb[s]:4d}{Idc[s]:4d}{Idd[s]:4d}{Set[s]:4d}{Rep[s]:4d} | "
              + "".join(f"{int(ice[s,j]):6d}" for j in range(8)))
    return dict(Lin=Lin,Ida=Ida,Idb=Idb,Idc=Idc,Seg=Seg,ice=ice,NCol=int(o.NCol))

for name in ['image','phasecor','refscan','noise']:
    o = tw.get(name, None) if hasattr(tw,'get') else (tw[name] if name in tw else None)
    if o is None:
        print(f"\n(no '{name}' category)")
        continue
    dims(o, name)

# focus phasecor (the 31P CRT stream)
pc = tw['phasecor'] if 'phasecor' in cats else None
if pc is not None:
    counters(pc, 'phasecor', 48)
im = tw['image'] if 'image' in cats else None
if im is not None:
    counters(im, 'image', 16)

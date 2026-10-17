#!/usr/bin/env python3
"""Speed-ups for pymapvbvd twix reading.

Two independent accelerations, both opt-in from the recon scripts:

1. ``accelerate_mdh_read()`` — monkey-patches ``mapvbvd.mapVBVD.loop_mdh_read``.
   The stock implementation walks the file one ADC at a time doing ``fid.seek``
   + ``np.fromfile(count=184)`` per MDH, so the scan costs one round trip per
   ADC.  That is merely wasteful on a local disk, but catastrophic on a
   high-latency filesystem (WSL /mnt/<drive>, network shares), where each round
   trip is tens of milliseconds.

   The replacement keeps the identical branch logic but picks its access
   pattern to match the filesystem, timing the first 256 MDHs to decide:

     sparse (mmap)  — touch only the 184 bytes per MDH.  Minimal I/O; best on
                      local disks.
     bulk (window)  — stream the measurement in 16 MB sequential reads and
                      serve MDHs from the window, ~200 ADCs per round trip.
                      Reads more bytes but amortises latency; best on drvfs.

   Measured on an 11.5 GB interleaved file: 6x faster on local disk (sparse),
   and ~6x on WSL /mnt/f (bulk; 87 min -> ~15 min).  Output is byte-identical
   to the stock reader in both modes.

2. ``cached_load()`` — pickles whatever a loader returns, keyed on the .dat
   file identity plus the loader options, so repeated recon runs on the same
   raw file skip the twix parse entirely (both the MDH scan and ``unsorted()``).

Neither changes reconstruction results.
"""

import hashlib
import mmap
import os
import pickle
import time
import warnings

import numpy as np

__all__ = ["accelerate_mdh_read", "cached_load", "cache_path_for"]


# ── 1. Fast MDH scan ─────────────────────────────────────────────────────────

_ORIG_LOOP_MDH_READ = None

# Byte layout of the fields the scan loop branches on.  VD keeps 20 fill bytes
# up front, hence the +20 offsets (see mapVBVD.loop_mdh_read).
_LAYOUT = {
    #          byteMDH, evIdx, dmaIdx0, dmaOff, dmaSkip
    "vd": (184, 20 + 20, 28 + 20, 192, 32),
    "vb": (128, 20 + 0, 28 + 0, 0, 128),
}

_MDH_ACQEND = 1 << 0
_MDH_SYNCDATA = 1 << 5

# Access-pattern tuning
_PROBE_N = 256           # MDHs per timing block
_PROBE_MS = 0.5          # ms/MDH above which sparse access is a losing game
_WINDOW = 16 << 20       # bulk read size

_STRATEGY = "auto"       # "auto" | "sparse" | "bulk"


def _fast_loop_mdh_read(fid, version, Nscans, scan, measOffset, measLength,
                        print_prog=True):
    """Drop-in replacement for mapvbvd.mapVBVD.loop_mdh_read."""
    if version not in _LAYOUT:
        return _ORIG_LOOP_MDH_READ(fid, version, Nscans, scan, measOffset,
                                   measLength, print_prog=print_prog)

    byteMDH, evIdx, dmaIdx0, dmaOff, dmaSkip = _LAYOUT[version]

    cPos = int(fid.tell())
    file_size = os.fstat(fid.fileno()).st_size
    isEOF = False
    positions = []
    blobs = []                     # raw 184-byte MDHs, collected as we go

    t = None
    if print_prog:
        from sys import stdout
        from tqdm import tqdm
        t = tqdm(total=int(measLength), unit="B", unit_scale=True,
                 unit_divisor=1024,
                 desc="Scan %d/%d, read all mdhs" % (scan + 1, Nscans),
                 leave=False, file=stdout)
    last_progress = cPos

    mm = None
    bulk = (_STRATEGY == "bulk")
    if not bulk:
        try:
            mm = mmap.mmap(fid.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            bulk = True            # e.g. empty file or a filesystem without mmap

    buf = b""
    buf_start = -1
    t_probe = time.time()

    try:
        while True:
            if cPos + byteMDH > file_size:
                warnings.warn(
                    "\nAn unexpected read error occurred at this byte offset:"
                    f" {cPos} ({cPos / 1024 ** 3} GiB)\nWill stop reading now.\n")
                isEOF = True
                break

            if bulk:
                # sliding window: cPos only ever moves forward
                if cPos < buf_start or cPos + byteMDH > buf_start + len(buf):
                    fid.seek(cPos, 0)
                    buf = fid.read(_WINDOW)
                    buf_start = cPos
                    if len(buf) < byteMDH:
                        warnings.warn(
                            "\nAn unexpected read error occurred at this byte offset:"
                            f" {cPos} ({cPos / 1024 ** 3} GiB)\nWill stop reading now.\n")
                        isEOF = True
                        break
                o = cPos - buf_start
                b = buf[o:o + byteMDH]
            else:
                b = mm[cPos:cPos + byteMDH]

            bitMask = b[evIdx]

            # ACQEND, or a genuinely zero ulDMALength -> end of measurement
            if (b[0] == 0 and b[1] == 0 and b[2] == 0) or (bitMask & _MDH_ACQEND):
                # ubit24: only bit 0 of the 4th byte belongs to ulDMALength
                ulDMALength = b[0] | (b[1] << 8) | (b[2] << 16) | ((b[3] & 1) << 24)
                if ulDMALength == 0 or (bitMask & _MDH_ACQEND):
                    cPos += ulDMALength
                    if cPos % 512:                      # jump to next full 512 B
                        cPos += 512 - cPos % 512
                    break

            if bitMask & _MDH_SYNCDATA:
                ulDMALength = b[0] | (b[1] << 8) | (b[2] << 16) | ((b[3] & 1) << 24)
                cPos += ulDMALength
                continue

            # ulDMALength recomputed from NCol/NCha (the pack bit is unreliable)
            nCol = b[dmaIdx0] | (b[dmaIdx0 + 1] << 8)
            nCha = b[dmaIdx0 + 2] | (b[dmaIdx0 + 3] << 8)
            ulDMALength = dmaOff + (8 * nCol + dmaSkip) * nCha

            positions.append(cPos)
            blobs.append(b)
            cPos += ulDMALength

            n_done = len(positions)
            if t is not None and (n_done & 0x3FF) == 0:
                t.update(cPos - last_progress)
                last_progress = cPos

            # On a high-latency filesystem, per-MDH access loses badly to
            # streaming the file in big sequential blocks.  Re-time every
            # block rather than probing once: the head of the file is often
            # still in page cache from an earlier run, which makes a one-shot
            # probe at the start report a speed the rest of the file won't hold.
            if (not bulk) and _STRATEGY == "auto" and (n_done % _PROBE_N) == 0:
                now = time.time()
                ms = (now - t_probe) / _PROBE_N * 1e3
                t_probe = now
                if ms > _PROBE_MS:
                    bulk = True
                    mm.close()
                    mm = None
                    print(f"\n[twix_fast] {ms:.1f} ms/MDH at {cPos/1e9:.1f} GB — "
                          f"high-latency filesystem, switching to bulk reads "
                          f"({_WINDOW >> 20} MB window)", flush=True)

        if isEOF and positions:
            positions.pop()                             # ignore the last attempt
            blobs.pop()

        n_acq = len(positions)
        if n_acq:
            mdh_blob = np.frombuffer(b"".join(blobs), dtype=np.uint8)
            mdh_blob = mdh_blob.reshape(n_acq, byteMDH).T.copy()  # writable
        else:
            mdh_blob = np.empty((byteMDH, 0), dtype=np.uint8)
        filePos = np.asarray(positions, dtype=float)
    finally:
        if t is not None:
            t.close()
        if mm is not None:
            mm.close()

    fid.seek(int(cPos), 0)
    return mdh_blob, filePos, isEOF


def accelerate_mdh_read(verbose=True, strategy="auto"):
    """Patch pymapvbvd's MDH scan with the adaptive reader (idempotent).

    ``strategy``: "auto" (probe and pick), "sparse" (mmap), "bulk" (streaming).
    """
    global _STRATEGY
    _STRATEGY = strategy

    global _ORIG_LOOP_MDH_READ
    import importlib
    # NB: ``import mapvbvd.mapVBVD as _m`` binds the *function* — the package
    # __init__ shadows the submodule name.  Go through importlib instead.
    _m = importlib.import_module("mapvbvd.mapVBVD")

    if getattr(_m.loop_mdh_read, "_is_fast_twix", False):
        return True
    _ORIG_LOOP_MDH_READ = _m.loop_mdh_read
    _fast_loop_mdh_read._is_fast_twix = True
    _m.loop_mdh_read = _fast_loop_mdh_read
    if verbose:
        print(f"[twix_fast] MDH scan accelerated (strategy={strategy})")
    return True


# ── 2. Parsed-data disk cache ────────────────────────────────────────────────

def cache_path_for(dat_path, cache_dir, key_extras=None, tag="load"):
    """Cache filename keyed on the .dat identity plus the loader options."""
    st = os.stat(dat_path)
    key = "|".join([
        os.path.abspath(dat_path), str(st.st_size), str(int(st.st_mtime)),
        repr(sorted((key_extras or {}).items())),
    ])
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(dat_path))[0]
    return os.path.join(cache_dir, f"{base}.{tag}.{h}.pkl")


def cached_load(loader, dat_path, cache_dir=None, key_extras=None,
                refresh=False, tag="load", **loader_kwargs):
    """Run ``loader(dat_path, **loader_kwargs)``, caching the result on disk.

    ``cache_dir=None`` disables caching entirely (loader runs as usual).
    ``key_extras`` must contain every option that changes the loader output.
    """
    if not cache_dir:
        return loader(dat_path, **loader_kwargs)

    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path_for(dat_path, cache_dir, key_extras, tag=tag)

    if os.path.exists(path) and not refresh:
        t0 = time.time()
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            print(f"[twix_fast] cache hit: {path}  "
                  f"({os.path.getsize(path)/1e6:.0f} MB, {time.time()-t0:.1f}s)")
            return obj
        except Exception as e:
            print(f"[twix_fast] cache unreadable ({type(e).__name__}: {e}) — reparsing")

    obj = loader(dat_path, **loader_kwargs)

    t0 = time.time()
    tmp = path + f".tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=5)
        os.replace(tmp, path)
        print(f"[twix_fast] cache written: {path}  "
              f"({os.path.getsize(path)/1e6:.0f} MB, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[twix_fast] cache write failed ({type(e).__name__}: {e})")
        if os.path.exists(tmp):
            os.remove(tmp)
    return obj

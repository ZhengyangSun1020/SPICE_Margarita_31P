import os, json, numpy as np
os.environ.setdefault("HDIR", "spice_final"); os.environ.setdefault("MU_SHOW", "30")
src = open("/mnt/e/claude_tmp/study/hybrid_fig.py").read(); exec(src[:src.index("for key, cmap, lo, hi, qname in")])
LAB = json.load(open("/mnt/e/claude_tmp/study/masks_hr/mask_labels.json"))
def excluded(B, frac_min=0.5):
    """QC noise (CA-SPICE PCr rep-to-rep noise in rest + late recovery >= 25 %) OR muscle area fraction < frac_min OR 8-neighbour of a phantom voxel."""
    h = H[B]; N = int(h["N"]); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg; li = list(h["leg_idx"])
    R0, R1 = map(int, h["rest"]); rep = float(h["REP"]); cc = arr(h, SP, "vox_pcr"); NR = cc.shape[0]; quiet = np.r_[R0:R1, NR - int(round(144 / rep)):NR]
    ph = (np.load(f"{ROOT}/output_study_mid{B}/reg_TRUE.npy") == 2).reshape(N, N); fS, fG = np.array(LAB[B]["fS"]), np.array(LAB[B]["fG"])
    disp = lambda q: (N - 1 - q // N) * N + q % N; out = {}
    for q in np.flatnonzero(vg | vs):
        d = disp(q); r, c = divmod(q, N); why = []
        if np.std(np.diff(cc[quiet, li.index(q)])) / np.sqrt(2) >= 0.25: why.append("noise")
        if (fG if vg[q] else fS)[d // N][d % N] < frac_min: why.append("frac")
        if any(0 <= r + dr < N and 0 <= c + dc < N and ph[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)): why.append("phantom")
        if why: out[q] = why
    return out
if __name__ == "__main__":
    for B in ("283", "279", "281", "277"):
        h = H[B]; N = int(h["N"]); leg = h["legv"].astype(bool); vg, vs = h["vGM"] & leg, h["vSOL"] & leg; disp = lambda q: (N - 1 - q // N) * N + q % N; ex = excluded(B)
        for nm, msk in (("GM", vg), ("soleus", vs)):
            qs = np.flatnonzero(msk); keep = [disp(q) for q in qs if q not in ex]
            print(f"MID00{B} {nm:6s}: keep {sorted(keep)} ({len(keep)}/{len(qs)}); excluded " + ", ".join(f"{disp(q)}[{'+'.join(ex[q])}]" for q in sorted(qs, key=disp) if q in ex))

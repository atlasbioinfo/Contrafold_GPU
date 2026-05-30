"""GPU MEA structure prediction for CONTRAfold (complementary model).

Callable, NON-default companion to ``cpu.mea``. Computes the base-pair
posterior matrix (BPP) on the GPU via inside + outside + posterior, in the
SAME float32 CONTRAfold arithmetic as ``cpu.posterior`` (Fast_LogPlusEquals /
Fast_Exp polynomials), then runs Maximum-Expected-Accuracy (Nussinov-style)
posterior decoding.

Design / what runs where
------------------------
* ``post_kernel`` (this file): one CUDA **block per sequence**. The full
  inside -> outside -> posterior recurrence is executed by **a single thread**
  (threadIdx 0) per block. This is a deliberate choice: CONTRAfold's outside
  pass accumulates with ``lse32`` in a fixed scatter order, and float32
  log-sum-exp is not associative, so any thread-level reordering of those
  accumulations can perturb borderline posteriors and flip the discrete MEA
  decode. Running the recurrence serially per block reproduces
  ``cpu.posterior`` bit-for-bit while still giving throughput by folding many
  sequences concurrently (grid = #sequences). The array layout, padding, canon
  table, param packing and float32 device functions mirror ``gpu.exact_kernel``.
* MEA decode (``mea_decode``, O(L^3)): the existing ``cpu.mea_decode`` njit is
  called per sequence on the host. The benchmark measures both phases so the
  decode share is visible.

A second entry point ``bpp_gpu`` returns the BPP matrices alone (useful for
accuracy comparison against ``cpu.bpp``).
"""
import numpy as np
import math
from numba import cuda, float32, int32

from . import cpu

NEG = np.float32(-1e30)
HALF = np.float32(-5e29)
CMAX = 30   # C_MAX_SINGLE_LENGTH
DH = 30     # D_MAX_HAIRPIN


# ----------------------------------------------------------------------------
# float32 device functions — identical polynomials to cpu.lse32 / cpu.fexp32
# ----------------------------------------------------------------------------
@cuda.jit(device=True, inline=True)
def lse(a, b):
    if a < b:
        t = a; a = b; b = t
    if b < HALF:
        return a
    d = a - b
    if d >= float32(11.8624794162):
        return a
    if d < float32(3.3792499610):
        if d < float32(1.6320158198):
            if d < float32(0.6615367791):
                r = ((float32(-0.0065591595) * d + float32(0.1276442762)) * d + float32(0.4996554598)) * d + float32(0.6931542306)
            else:
                r = ((float32(-0.0155157557) * d + float32(0.1446775699)) * d + float32(0.4882939746)) * d + float32(0.6958092989)
        elif d < float32(2.4912588184):
            r = ((float32(-0.0128909247) * d + float32(0.1301028251)) * d + float32(0.5150398748)) * d + float32(0.6795585882)
        else:
            r = ((float32(-0.0072142647) * d + float32(0.0877540853)) * d + float32(0.6208708362)) * d + float32(0.5909675829)
    elif d < float32(5.7890710412):
        if d < float32(4.4261691294):
            r = ((float32(-0.0031455354) * d + float32(0.0467229449)) * d + float32(0.7592532310)) * d + float32(0.4348794399)
        else:
            r = ((float32(-0.0010110698) * d + float32(0.0185943421)) * d + float32(0.8831730747)) * d + float32(0.2523695427)
    elif d < float32(7.8162726752):
        r = ((float32(-0.0001962780) * d + float32(0.0046084408)) * d + float32(0.9634431978)) * d + float32(0.0983148903)
    else:
        r = ((float32(-0.0000113994) * d + float32(0.0003734731)) * d + float32(0.9959107193)) * d + float32(0.0149855051)
    return b + r


@cuda.jit(device=True, inline=True)
def fexp(x):
    if x < float32(-2.4915033807):
        if x < float32(-5.8622823336):
            if x < float32(-9.91152):
                return float32(0.0)
            return ((float32(0.0000803850) * x + float32(0.0021627428)) * x + float32(0.0194708555)) * x + float32(0.0588080014)
        if x < float32(-3.8396630909):
            return ((float32(0.0013889414) * x + float32(0.0244676474)) * x + float32(0.1471290604)) * x + float32(0.3042757740)
        return ((float32(0.0072335607) * x + float32(0.0906002677)) * x + float32(0.3983111356)) * x + float32(0.6245959221)
    if x < float32(-0.6725053211):
        if x < float32(-1.4805375919):
            return ((float32(0.0232410351) * x + float32(0.2085645908)) * x + float32(0.6906367911)) * x + float32(0.8682322329)
        return ((float32(0.0573782771) * x + float32(0.3580258429)) * x + float32(0.9121133217)) * x + float32(0.9793091728)
    if x < float32(0.0):
        return ((float32(0.1199175927) * x + float32(0.4815668234)) * x + float32(0.9975991939)) * x + float32(0.9999505077)
    if x > float32(46.052):
        return float32(1e20)
    return float32(math.exp(x))


# ----------------------------------------------------------------------------
# Posterior kernel: inside + outside + posterior, one block per sequence,
# single-thread recurrence (faithful to cpu.posterior).
# ----------------------------------------------------------------------------
@cuda.jit(cache=True)
def post_kernel(seqs, lengths, forced, canon,
                bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
                mb, mu, mp, eu, ep,
                FC, FM, FM1, F5,
                FCo, FMo, FM1o, F5o, POST):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    nt = cuda.blockDim.x
    L = lengths[b]
    s = seqs[b]
    fo = forced[b]
    nn = L + 2

    ZERO = float32(0.0)

    # ---- parallel init of all matrices ----
    idx = t
    tot = nn * nn
    while idx < tot:
        i = idx // nn
        j = idx % nn
        FC[b, i, j] = NEG
        FM[b, i, j] = NEG
        FM1[b, i, j] = NEG
        FCo[b, i, j] = NEG
        FMo[b, i, j] = NEG
        FM1o[b, i, j] = NEG
        POST[b, i, j] = ZERO
        idx += nt
    idx = t
    while idx < nn:
        F5[b, idx] = NEG
        F5o[b, idx] = NEG
        idx += nt
    cuda.syncthreads()

    if t != 0:
        return

    # ====================== INSIDE ======================
    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            FM2 = NEG
            for k in range(i + 1, j):
                if FM1[b, i, k] > HALF and FM[b, k, j] > HALF:
                    FM2 = lse(FM2, FM1[b, i, k] + FM[b, k, j])
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and fo[i] == 0 and fo[j + 1] == 0:
                sum_i = NEG
                jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                d = j - i
                sum_i = lse(sum_i, jB + hp_cum[d if d <= DH else DH])
                pmax = i + CMAX
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - CMAX
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and fo[p + 1] == 0 and fo[q] == 0 and FC[b, p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = ZERO
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB + jB2 + snuc
                            sum_i = lse(sum_i, e + FC[b, p + 1, q - 1])
                if FM2 > HALF:
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    sum_i = lse(sum_i, FM2 + jA + mp + mb)
                FC[b, i, j] = sum_i
            if 0 < i and i + 2 <= j and j < L:
                sum_i = NEG
                if canon[s[i + 1], s[j]] == 1 and fo[i + 1] == 0 and fo[j] == 0 and FC[b, i + 1, j - 1] > HALF:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    sum_i = lse(sum_i, FC[b, i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]])
                if FM1[b, i + 1, j] > HALF:
                    sum_i = lse(sum_i, FM1[b, i + 1, j] + mu)
                FM1[b, i, j] = sum_i
            if 0 < i and i + 2 <= j and j < L:
                sum_i = NEG
                if FM2 > HALF:
                    sum_i = lse(sum_i, FM2)
                if FM[b, i, j - 1] > HALF:
                    sum_i = lse(sum_i, FM[b, i, j - 1] + mu)
                if FM1[b, i, j] > HALF:
                    sum_i = lse(sum_i, FM1[b, i, j])
                FM[b, i, j] = sum_i
    F5[b, 0] = ZERO
    for j in range(1, L + 1):
        sum_i = F5[b, j - 1] + eu
        for k in range(0, j):
            if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0 and FC[b, k + 1, j - 1] > HALF and F5[b, k] > HALF:
                jA = hc[s[j], s[k + 1]]
                if j < L:
                    jA += dl[s[j], s[k + 1], s[j + 1]]
                if k > 0:
                    jA += dr[s[j], s[k + 1], s[k]]
                sum_i = lse(sum_i, F5[b, k] + FC[b, k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA)
        F5[b, j] = sum_i
    Z = F5[b, L]

    # ====================== OUTSIDE ======================
    F5o[b, L] = ZERO
    for j in range(L, 0, -1):
        F5o[b, j - 1] = lse(F5o[b, j - 1], F5o[b, j] + eu)
        for k in range(0, j):
            if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0:
                jA = hc[s[j], s[k + 1]]
                if j < L:
                    jA += dl[s[j], s[k + 1], s[j + 1]]
                if k > 0:
                    jA += dr[s[j], s[k + 1], s[k]]
                temp = F5o[b, j] + ep + bp[s[k + 1], s[j]] + jA
                if FC[b, k + 1, j - 1] > HALF:
                    F5o[b, k] = lse(F5o[b, k], temp + FC[b, k + 1, j - 1])
                if F5[b, k] > HALF:
                    FCo[b, k + 1, j - 1] = lse(FCo[b, k + 1, j - 1], temp + F5[b, k])
    for i in range(0, L + 1):
        for j in range(L, i - 1, -1):
            FM2o = NEG
            if 0 < i and i + 2 <= j and j < L:
                FM2o = lse(FM2o, FMo[b, i, j])
                FMo[b, i, j - 1] = lse(FMo[b, i, j - 1], FMo[b, i, j] + mu)
                FM1o[b, i, j] = lse(FM1o[b, i, j], FMo[b, i, j])
            if 0 < i and i + 2 <= j and j < L:
                if canon[s[i + 1], s[j]] == 1 and fo[i + 1] == 0 and fo[j] == 0:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    FCo[b, i + 1, j - 1] = lse(FCo[b, i + 1, j - 1], FM1o[b, i, j] + jAji + mp + bp[s[i + 1], s[j]])
                FM1o[b, i + 1, j] = lse(FM1o[b, i + 1, j], FM1o[b, i, j] + mu)
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and fo[i] == 0 and fo[j + 1] == 0:
                fco = FCo[b, i, j]
                if fco > HALF:
                    jB_ij = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                    pmax = i + CMAX
                    if pmax > j:
                        pmax = j
                    for p in range(i, pmax + 1):
                        l1 = p - i
                        qmin = p + 2
                        alt = p - i + j - CMAX
                        if alt > qmin:
                            qmin = alt
                        for q in range(j, qmin - 1, -1):
                            l2 = j - q
                            if canon[s[p + 1], s[q]] == 1 and fo[p + 1] == 0 and fo[q] == 0:
                                if p == i and q == j:
                                    e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                                else:
                                    jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                    snuc = ZERO
                                    if l1 == 0 and l2 == 1:
                                        snuc = b01[s[j]]
                                    elif l1 == 1 and l2 == 0:
                                        snuc = b01[s[i + 1]]
                                    elif l1 == 1 and l2 == 1:
                                        snuc = i11[s[i + 1], s[j]]
                                    e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB_ij + jB2 + snuc
                                FCo[b, p + 1, q - 1] = lse(FCo[b, p + 1, q - 1], fco + e)
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    FM2o = lse(FM2o, fco + jA + mp + mb)
            if FM2o > HALF:
                for k in range(i + 1, j):
                    if FM[b, k, j] > HALF:
                        FM1o[b, i, k] = lse(FM1o[b, i, k], FM2o + FM[b, k, j])
                    if FM1[b, i, k] > HALF:
                        FMo[b, k, j] = lse(FMo[b, k, j], FM2o + FM1[b, i, k])

    # ====================== POSTERIOR ======================
    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and fo[i] == 0 and fo[j + 1] == 0:
                outside = FCo[b, i, j] - Z
                if outside > HALF:
                    jB_ij = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                    pmax = i + CMAX
                    if pmax > j:
                        pmax = j
                    for p in range(i, pmax + 1):
                        l1 = p - i
                        qmin = p + 2
                        alt = p - i + j - CMAX
                        if alt > qmin:
                            qmin = alt
                        for q in range(j, qmin - 1, -1):
                            l2 = j - q
                            if canon[s[p + 1], s[q]] == 1 and fo[p + 1] == 0 and fo[q] == 0 and FC[b, p + 1, q - 1] > HALF:
                                if p == i and q == j:
                                    e = outside + bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]] + FC[b, p + 1, q - 1]
                                else:
                                    jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                    snuc = ZERO
                                    if l1 == 0 and l2 == 1:
                                        snuc = b01[s[j]]
                                    elif l1 == 1 and l2 == 0:
                                        snuc = b01[s[i + 1]]
                                    elif l1 == 1 and l2 == 1:
                                        snuc = i11[s[i + 1], s[j]]
                                    e = outside + jB_ij + cs[l1, l2] + FC[b, p + 1, q - 1] + bp[s[p + 1], s[q]] + jB2 + snuc
                                POST[b, p + 1, q] += fexp(e)
            if 0 < i and i + 2 <= j and j < L:
                if canon[s[i + 1], s[j]] == 1 and fo[i + 1] == 0 and fo[j] == 0 and FC[b, i + 1, j - 1] > HALF and FM1o[b, i, j] > HALF:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    POST[b, i + 1, j] += fexp(FM1o[b, i, j] + FC[b, i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]] - Z)
    for j in range(1, L + 1):
        outside = F5o[b, j] - Z
        if outside > HALF:
            for k in range(0, j):
                if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0 and FC[b, k + 1, j - 1] > HALF and F5[b, k] > HALF:
                    jA = hc[s[j], s[k + 1]]
                    if j < L:
                        jA += dl[s[j], s[k + 1], s[j + 1]]
                    if k > 0:
                        jA += dr[s[j], s[k + 1], s[k]]
                    POST[b, k + 1, j] += fexp(outside + F5[b, k] + FC[b, k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA)
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            v = POST[b, i, j]
            if v < ZERO:
                v = ZERO
            elif v > float32(1.0):
                v = float32(1.0)
            POST[b, i, j] = v


# ----------------------------------------------------------------------------
# Host wrappers
# ----------------------------------------------------------------------------
_BASE = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
_CANON_PAIRS = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]


def _prepare(seqs, P, forced_list):
    n_max = max(len(s) for s in seqs)
    Bn = len(seqs)
    S = np.full((Bn, n_max + 2), 4, np.int32)
    L = np.zeros(Bn, np.int32)
    FO = np.zeros((Bn, n_max + 2), np.int32)
    for bi, sq in enumerate(seqs):
        for k, c in enumerate(sq.upper()):
            S[bi, k + 1] = _BASE.get(c, 4)
        Lb = len(sq)
        L[bi] = Lb
        if forced_list is not None and forced_list[bi] is not None:
            FO[bi, 1:Lb + 1] = np.asarray(forced_list[bi], np.int32)[:Lb]
    canon = np.zeros((5, 5), np.int32)
    for (a, b2) in _CANON_PAIRS:
        canon[a, b2] = 1
    f = cpu._params_f32(P)
    return S, L, FO, canon, f, n_max, Bn


def _run_post_kernel(seqs, P, forced_list, threads):
    S, L, FO, canon, f, n_max, Bn = _prepare(seqs, P, forced_list)
    nn = n_max + 2

    d = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    di = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.int32))

    FC = cuda.device_array((Bn, nn, nn), np.float32)
    FM = cuda.device_array((Bn, nn, nn), np.float32)
    FM1 = cuda.device_array((Bn, nn, nn), np.float32)
    F5 = cuda.device_array((Bn, nn), np.float32)
    FCo = cuda.device_array((Bn, nn, nn), np.float32)
    FMo = cuda.device_array((Bn, nn, nn), np.float32)
    FM1o = cuda.device_array((Bn, nn, nn), np.float32)
    F5o = cuda.device_array((Bn, nn), np.float32)
    POST = cuda.device_array((Bn, nn, nn), np.float32)

    post_kernel[Bn, threads](
        di(S), di(L), di(FO), di(canon),
        d(f["bp"]), d(f["stack"]), d(f["tm"]), d(f["hc"]), d(f["dl"]), d(f["dr"]),
        d(f["hp_cum"]), d(f["cs"]), d(f["i11"]), d(f["b01"]),
        np.float32(f["mb"]), np.float32(f["mu"]), np.float32(f["mp"]),
        np.float32(f["eu"]), np.float32(f["ep"]),
        FC, FM, FM1, F5, FCo, FMo, FM1o, F5o, POST)
    cuda.synchronize()
    POSTh = POST.copy_to_host()
    return POSTh, S, L, FO


def bpp_gpu(seqs, P, forced_list=None, threads=128):
    """Per-sequence base-pair probability matrices computed on the GPU.

    Returns a list of (Lb x Lb) float64 upper-triangle matrices (0-based),
    matching the semantics of ``cpu.bpp`` for each sequence."""
    POSTh, S, L, FO = _run_post_kernel(seqs, P, forced_list, threads)
    out = []
    for bi in range(len(seqs)):
        Lb = int(L[bi])
        out.append(POSTh[bi, 1:Lb + 1, 1:Lb + 1].astype(np.float64))
    return out


def mea_gpu(seqs, P, gamma=6.0, forced_list=None, threads=128):
    """GPU Maximum-Expected-Accuracy structures (dot-bracket), mirroring
    ``cpu.mea`` semantics. Posterior (inside/outside) runs on GPU; the
    O(L^3) MEA decode runs on the host via ``cpu.mea_decode``."""
    POSTh, S, L, FO = _run_post_kernel(seqs, P, forced_list, threads)
    g = np.float32(gamma)
    res = []
    for bi in range(len(seqs)):
        Lb = int(L[bi])
        nn = Lb + 2
        POST = np.ascontiguousarray(POSTh[bi, :nn, :nn])
        s = np.ascontiguousarray(S[bi, :nn].astype(np.int64))
        fo = np.ascontiguousarray(FO[bi, :nn].astype(np.int64))
        pair = cpu.mea_decode(POST, Lb, fo, cpu._CANON, s, g)
        ch = ["."] * Lb
        for t in range(1, Lb + 1):
            if pair[t] > t:
                ch[t - 1] = "("; ch[pair[t] - 1] = ")"
        res.append("".join(ch))
    return res


if __name__ == "__main__":
    P = cpu.load()
    tests = ["GGGGAAAACCCC", "GCGCGCAAAAGCGCGCAAAAGCGC",
             "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"]
    gpu_mea = mea_gpu(tests, P)
    print("=== GPU MEA vs CPU MEA ===")
    for sq, gm in zip(tests, gpu_mea):
        cm = cpu.mea(sq, P)
        print(f"{'MATCH' if gm == cm else 'DIFF ':5s} {sq[:36]:36s}")
        print("  GPU:", gm)
        print("  CPU:", cm)

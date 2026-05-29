"""GPU port of the exact CONTRAfold partition function (complementary model).

Span-based parallelization (threads over i at each span s=j-i), one block per
sequence. Exact scoring (terminal_mismatch, dangles, helix_closing, full
ScoreSingle, internal_1x1, bulge_0x1) mirrors contrafold_exact.py. float32 for
throughput; logZ matches CPU/binary within fp tolerance (sampling distribution
unaffected).

Indexing follows CONTRAfold: 1-based s[1..L], FC[i][j] = inside of pair (i,j+1).
Span s = j - i. Per span: FM2 -> FC -> FM1 -> FM. Then F5 exterior.
"""
import numpy as np
import math
from numba import cuda, float32, int32

NEG = np.float32(-1e30)
HALF = np.float32(-5e29)
CMAX = 30  # C_MAX_SINGLE_LENGTH
DH = 30    # D_MAX_HAIRPIN


@cuda.jit(device=True, inline=True)
def lse(a, b):
    if a < HALF:
        return b
    if b < HALF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@cuda.jit(cache=True)
def exact_kernel(seqs, lengths, forced, canon,
                 bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
                 mb, mu, mp, eu, ep,
                 FC, FM, FM1, F5):
    b = cuda.blockIdx.x
    t = cuda.threadIdx.x
    nt = cuda.blockDim.x
    L = lengths[b]
    s = seqs[b]            # 1-based view, length L+2
    fo = forced[b]

    # init (indices 0..L+1)
    idx = t
    tot = (L + 2) * (L + 2)
    while idx < tot:
        i = idx // (L + 2); j = idx % (L + 2)
        FC[b, i, j] = NEG; FM[b, i, j] = NEG; FM1[b, i, j] = NEG
        idx += nt
    if t == 0:
        for j in range(L + 2):
            F5[b, j] = NEG
    cuda.syncthreads()

    for span in range(0, L + 1):
        # ----- FC (FM2 computed inline; not stored) -----
        i = t
        while i + span <= L:
            j = i + span
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and fo[i] == 0 and fo[j + 1] == 0:
                acc = NEG
                # ScoreJunctionB(i,j) — loop-invariant, computed ONCE
                jB_ij = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                # hairpin
                d = j - i
                acc = lse(acc, jB_ij + hp_cum[d if d <= DH else DH])
                # single-branch loops
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
                                snuc = float32(0.0)
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB_ij + jB2 + snuc
                            acc = lse(acc, e + FC[b, p + 1, q - 1])
                # multibranch: FM2 computed inline
                fm2 = NEG
                for k in range(i + 1, j):
                    if FM1[b, i, k] > HALF and FM[b, k, j] > HALF:
                        fm2 = lse(fm2, FM1[b, i, k] + FM[b, k, j])
                if fm2 > HALF:
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    acc = lse(acc, fm2 + jA + mp + mb)
                FC[b, i, j] = acc
            i += nt
        cuda.syncthreads()
        # ----- FM1 -----
        i = t
        while i + span <= L:
            j = i + span
            if 0 < i and i + 2 <= j and j < L:
                acc = NEG
                if canon[s[i + 1], s[j]] == 1 and fo[i + 1] == 0 and fo[j] == 0 and FC[b, i + 1, j - 1] > HALF:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    acc = lse(acc, FC[b, i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]])
                if FM1[b, i + 1, j] > HALF:
                    acc = lse(acc, FM1[b, i + 1, j] + mu)
                FM1[b, i, j] = acc
            i += nt
        cuda.syncthreads()
        # ----- FM (FM2 recomputed inline) -----
        i = t
        while i + span <= L:
            j = i + span
            if 0 < i and i + 2 <= j and j < L:
                acc = NEG
                fm2 = NEG
                for k in range(i + 1, j):
                    if FM1[b, i, k] > HALF and FM[b, k, j] > HALF:
                        fm2 = lse(fm2, FM1[b, i, k] + FM[b, k, j])
                if fm2 > HALF:
                    acc = lse(acc, fm2)
                if FM[b, i, j - 1] > HALF:
                    acc = lse(acc, FM[b, i, j - 1] + mu)
                if FM1[b, i, j] > HALF:
                    acc = lse(acc, FM1[b, i, j])
                FM[b, i, j] = acc
            i += nt
        cuda.syncthreads()

    # ----- F5 exterior (single thread; cheap O(L^2)) -----
    if t == 0:
        F5[b, 0] = float32(0.0)
        for j in range(1, L + 1):
            acc = F5[b, j - 1] + eu
            for k in range(0, j):
                if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0 and FC[b, k + 1, j - 1] > HALF and F5[b, k] > HALF:
                    jA = hc[s[j], s[k + 1]]
                    if j < L:
                        jA += dl[s[j], s[k + 1], s[j + 1]]
                    if k > 0:
                        jA += dr[s[j], s[k + 1], s[k]]
                    acc = lse(acc, F5[b, k] + FC[b, k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA)
            F5[b, j] = acc
    cuda.syncthreads()


MAXSTACK = 320


@cuda.jit(device=True, inline=True)
def jA_dev(a, b, sa, sa1, sb1, L, hc, dl, dr):
    # ScoreJunctionA(a,b): pair (a, b+1); sa=s[a], sb1=s[b+1], sa1=s[a+1], plus dangle right s[b]
    pass  # (inlined manually below for clarity/perf)


@cuda.jit(cache=True)
def sample_kernel(seqs, lengths, forced, canon,
                  bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
                  mb, mu, mp, eu, ep,
                  FC, FM, FM1, F5, rng, n_samples, out):
    g = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    B = lengths.shape[0]
    if g >= B * n_samples:
        return
    bi = g // n_samples
    L = lengths[bi]
    s = seqs[bi]
    fo = forced[bi]

    for t in range(L + 2):
        out[g, t] = -1

    st_type = cuda.local.array(MAXSTACK, int32)
    st_i = cuda.local.array(MAXSTACK, int32)
    st_j = cuda.local.array(MAXSTACK, int32)
    sp = 0
    # start: F5 over region 1..L
    st_type[0] = 0; st_i[0] = 0; st_j[0] = L; sp = 1

    while sp > 0:
        sp -= 1
        typ = st_type[sp]; i = st_i[sp]; j = st_j[sp]

        if typ == 0:                       # ---- F5(j) ----
            if j == 0:
                continue
            # pass A: max
            m = NEG
            wun = NEG
            if F5[bi, j - 1] > HALF:
                wun = F5[bi, j - 1] + eu
                if wun > m:
                    m = wun
            for k in range(0, j):
                if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0 and FC[bi, k + 1, j - 1] > HALF and F5[bi, k] > HALF:
                    jA = hc[s[j], s[k + 1]]
                    if j < L:
                        jA += dl[s[j], s[k + 1], s[j + 1]]
                    if k > 0:
                        jA += dr[s[j], s[k + 1], s[k]]
                    w = F5[bi, k] + FC[bi, k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                    if w > m:
                        m = w
            # pass B: sum
            S = float32(0.0)
            if wun > HALF:
                S += math.exp(wun - m)
            for k in range(0, j):
                if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0 and FC[bi, k + 1, j - 1] > HALF and F5[bi, k] > HALF:
                    jA = hc[s[j], s[k + 1]]
                    if j < L:
                        jA += dl[s[j], s[k + 1], s[j + 1]]
                    if k > 0:
                        jA += dr[s[j], s[k + 1], s[k]]
                    w = F5[bi, k] + FC[bi, k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                    S += math.exp(w - m)
            # pass C: pick
            r = cuda.random.xoroshiro128p_uniform_float32(rng, g) * S
            c = float32(0.0)
            picked = -2
            if wun > HALF:
                c += math.exp(wun - m)
                if r <= c:
                    picked = -1
            if picked == -2:
                for k in range(0, j):
                    if canon[s[k + 1], s[j]] == 1 and fo[k + 1] == 0 and fo[j] == 0 and FC[bi, k + 1, j - 1] > HALF and F5[bi, k] > HALF:
                        jA = hc[s[j], s[k + 1]]
                        if j < L:
                            jA += dl[s[j], s[k + 1], s[j + 1]]
                        if k > 0:
                            jA += dr[s[j], s[k + 1], s[k]]
                        w = F5[bi, k] + FC[bi, k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                        c += math.exp(w - m)
                        if r <= c:
                            picked = k
                            break
            if picked == -1:
                st_type[sp] = 0; st_i[sp] = 0; st_j[sp] = j - 1; sp += 1
            elif picked >= 0:
                k = picked
                out[g, k + 1] = j; out[g, j] = k + 1     # pair (k+1, j)
                st_type[sp] = 1; st_i[sp] = k + 1; st_j[sp] = j - 1; sp += 1   # FC(k+1, j-1)
                st_type[sp] = 0; st_i[sp] = 0; st_j[sp] = k; sp += 1           # F5(k)

        elif typ == 1:                     # ---- FC(i,j): pair (i,j+1) already recorded ----
            jB_ij = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
            d = j - i
            # pass A: max over hairpin, single(p,q), multibranch
            m = jB_ij + hp_cum[d if d <= DH else DH]
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
                    if canon[s[p + 1], s[q]] == 1 and fo[p + 1] == 0 and fo[q] == 0 and FC[bi, p + 1, q - 1] > HALF:
                        if p == i and q == j:
                            e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                        else:
                            jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                            sn = float32(0.0)
                            if l1 == 0 and l2 == 1:
                                sn = b01[s[j]]
                            elif l1 == 1 and l2 == 0:
                                sn = b01[s[i + 1]]
                            elif l1 == 1 and l2 == 1:
                                sn = i11[s[i + 1], s[j]]
                            e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB_ij + jB2 + sn
                        w = e + FC[bi, p + 1, q - 1]
                        if w > m:
                            m = w
            # multibranch fm2
            jA = hc[s[i], s[j + 1]]
            if i < L:
                jA += dl[s[i], s[j + 1], s[i + 1]]
            if j > 0:
                jA += dr[s[i], s[j + 1], s[j]]
            for k in range(i + 1, j):
                if FM1[bi, i, k] > HALF and FM[bi, k, j] > HALF:
                    w = FM1[bi, i, k] + FM[bi, k, j] + jA + mp + mb
                    if w > m:
                        m = w
            # pass B: sum
            S = math.exp((jB_ij + hp_cum[d if d <= DH else DH]) - m)
            for p in range(i, pmax + 1):
                l1 = p - i
                qmin = p + 2
                alt = p - i + j - CMAX
                if alt > qmin:
                    qmin = alt
                for q in range(j, qmin - 1, -1):
                    l2 = j - q
                    if canon[s[p + 1], s[q]] == 1 and fo[p + 1] == 0 and fo[q] == 0 and FC[bi, p + 1, q - 1] > HALF:
                        if p == i and q == j:
                            e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                        else:
                            jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                            sn = float32(0.0)
                            if l1 == 0 and l2 == 1:
                                sn = b01[s[j]]
                            elif l1 == 1 and l2 == 0:
                                sn = b01[s[i + 1]]
                            elif l1 == 1 and l2 == 1:
                                sn = i11[s[i + 1], s[j]]
                            e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB_ij + jB2 + sn
                        S += math.exp((e + FC[bi, p + 1, q - 1]) - m)
            for k in range(i + 1, j):
                if FM1[bi, i, k] > HALF and FM[bi, k, j] > HALF:
                    S += math.exp((FM1[bi, i, k] + FM[bi, k, j] + jA + mp + mb) - m)
            # pass C: pick
            r = cuda.random.xoroshiro128p_uniform_float32(rng, g) * S
            c = math.exp((jB_ij + hp_cum[d if d <= DH else DH]) - m)
            done = False
            if r <= c:
                done = True   # hairpin: interior unpaired, nothing to push
            if not done:
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - CMAX
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and fo[p + 1] == 0 and fo[q] == 0 and FC[bi, p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                sn = float32(0.0)
                                if l1 == 0 and l2 == 1:
                                    sn = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    sn = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    sn = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB_ij + jB2 + sn
                            c += math.exp((e + FC[bi, p + 1, q - 1]) - m)
                            if r <= c:
                                out[g, p + 1] = q; out[g, q] = p + 1   # inner pair (p+1, q)
                                st_type[sp] = 1; st_i[sp] = p + 1; st_j[sp] = q - 1; sp += 1
                                done = True
                                break
                    if done:
                        break
            if not done:
                for k in range(i + 1, j):
                    if FM1[bi, i, k] > HALF and FM[bi, k, j] > HALF:
                        c += math.exp((FM1[bi, i, k] + FM[bi, k, j] + jA + mp + mb) - m)
                        if r <= c:
                            st_type[sp] = 3; st_i[sp] = i; st_j[sp] = k; sp += 1   # FM1(i,k)
                            st_type[sp] = 2; st_i[sp] = k; st_j[sp] = j; sp += 1   # FM(k,j)
                            done = True
                            break

        elif typ == 2:                     # ---- FM(i,j) ----
            m = NEG
            # fm2 splits
            for k in range(i + 1, j):
                if FM1[bi, i, k] > HALF and FM[bi, k, j] > HALF:
                    w = FM1[bi, i, k] + FM[bi, k, j]
                    if w > m:
                        m = w
            wun = NEG
            if FM[bi, i, j - 1] > HALF:
                wun = FM[bi, i, j - 1] + mu
                if wun > m:
                    m = wun
            w1 = NEG
            if FM1[bi, i, j] > HALF:
                w1 = FM1[bi, i, j]
                if w1 > m:
                    m = w1
            S = float32(0.0)
            for k in range(i + 1, j):
                if FM1[bi, i, k] > HALF and FM[bi, k, j] > HALF:
                    S += math.exp((FM1[bi, i, k] + FM[bi, k, j]) - m)
            if wun > HALF:
                S += math.exp(wun - m)
            if w1 > HALF:
                S += math.exp(w1 - m)
            r = cuda.random.xoroshiro128p_uniform_float32(rng, g) * S
            c = float32(0.0)
            done = False
            for k in range(i + 1, j):
                if FM1[bi, i, k] > HALF and FM[bi, k, j] > HALF:
                    c += math.exp((FM1[bi, i, k] + FM[bi, k, j]) - m)
                    if r <= c:
                        st_type[sp] = 3; st_i[sp] = i; st_j[sp] = k; sp += 1
                        st_type[sp] = 2; st_i[sp] = k; st_j[sp] = j; sp += 1
                        done = True
                        break
            if not done and wun > HALF:
                c += math.exp(wun - m)
                if r <= c:
                    st_type[sp] = 2; st_i[sp] = i; st_j[sp] = j - 1; sp += 1
                    done = True
            if not done and w1 > HALF:
                st_type[sp] = 3; st_i[sp] = i; st_j[sp] = j; sp += 1

        else:                              # ---- FM1(i,j) ----
            wfc = NEG
            if canon[s[i + 1], s[j]] == 1 and fo[i + 1] == 0 and fo[j] == 0 and FC[bi, i + 1, j - 1] > HALF:
                jAji = hc[s[j], s[i + 1]]
                if j < L:
                    jAji += dl[s[j], s[i + 1], s[j + 1]]
                if i > 0:
                    jAji += dr[s[j], s[i + 1], s[i]]
                wfc = FC[bi, i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]]
            wun = NEG
            if FM1[bi, i + 1, j] > HALF:
                wun = FM1[bi, i + 1, j] + mu
            m = wfc
            if wun > m:
                m = wun
            S = float32(0.0)
            if wfc > HALF:
                S += math.exp(wfc - m)
            if wun > HALF:
                S += math.exp(wun - m)
            r = cuda.random.xoroshiro128p_uniform_float32(rng, g) * S
            c = float32(0.0)
            done = False
            if wfc > HALF:
                c += math.exp(wfc - m)
                if r <= c:
                    out[g, i + 1] = j; out[g, j] = i + 1     # pair (i+1, j)
                    st_type[sp] = 1; st_i[sp] = i + 1; st_j[sp] = j - 1; sp += 1
                    done = True
            if not done and wun > HALF:
                st_type[sp] = 3; st_i[sp] = i + 1; st_j[sp] = j; sp += 1


def logZ_batch(seqs, P, forced_list=None, threads=128):
    """seqs: list of strings. Returns list of logZ (float)."""
    B = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    n_max = max(len(s) for s in seqs)
    Bn = len(seqs)
    S = np.full((Bn, n_max + 2), 4, np.int32)
    L = np.zeros(Bn, np.int32)
    FO = np.zeros((Bn, n_max + 2), np.int32)
    for bi, sq in enumerate(seqs):
        for k, c in enumerate(sq.upper()):
            S[bi, k + 1] = B.get(c, 4)
        Lb = len(sq)
        L[bi] = Lb
        if forced_list is not None and forced_list[bi] is not None:
            FO[bi, 1:Lb + 1] = np.asarray(forced_list[bi], np.int32)[:Lb]

    canon = np.zeros((5, 5), np.int32)
    for (a, b2) in [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]:
        canon[a, b2] = 1

    d = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    di = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.int32))
    nn = n_max + 2
    FC = cuda.device_array((Bn, nn, nn), np.float32)
    FM = cuda.device_array((Bn, nn, nn), np.float32)
    FM1 = cuda.device_array((Bn, nn, nn), np.float32)
    F5 = cuda.device_array((Bn, nn), np.float32)

    exact_kernel[Bn, threads](
        di(S), di(L), di(FO), di(canon),
        d(P["bp"]), d(P["stack"]), d(P["tm"]), d(P["hc"]), d(P["dl"]), d(P["dr"]),
        d(P["hp_cum"]), d(P["cs"]), d(P["i11"]), d(P["b01"]),
        np.float32(P["mb"]), np.float32(P["mu"]), np.float32(P["mp"]),
        np.float32(P["eu"]), np.float32(P["ep"]),
        FC, FM, FM1, F5)
    cuda.synchronize()
    F5h = F5.copy_to_host()
    return [float(F5h[bi, int(L[bi])]) for bi in range(Bn)]


def sample_batch(seqs, P, n_samples, forced_list=None, threads=128, seed=0):
    """Fold inside on GPU then GPU stochastic traceback. Returns list (per seq)
    of n_samples dot-bracket strings."""
    from numba.cuda.random import create_xoroshiro128p_states
    BASE = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    n_max = max(len(s) for s in seqs)
    Bn = len(seqs)
    S = np.full((Bn, n_max + 2), 4, np.int32)
    L = np.zeros(Bn, np.int32)
    FO = np.zeros((Bn, n_max + 2), np.int32)
    for bi, sq in enumerate(seqs):
        for k, c in enumerate(sq.upper()):
            S[bi, k + 1] = BASE.get(c, 4)
        Lb = len(sq)
        L[bi] = Lb
        if forced_list is not None and forced_list[bi] is not None:
            FO[bi, 1:Lb + 1] = np.asarray(forced_list[bi], np.int32)[:Lb]
    canon = np.zeros((5, 5), np.int32)
    for (a, b2) in [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]:
        canon[a, b2] = 1

    d = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    di = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.int32))
    nn = n_max + 2
    dS, dL, dFO, dcanon = di(S), di(L), di(FO), di(canon)
    dbp, dstack, dtm = d(P["bp"]), d(P["stack"]), d(P["tm"])
    dhc, ddl, ddr = d(P["hc"]), d(P["dl"]), d(P["dr"])
    dhp, dcs, di11, db01 = d(P["hp_cum"]), d(P["cs"]), d(P["i11"]), d(P["b01"])
    mb, mu, mp = np.float32(P["mb"]), np.float32(P["mu"]), np.float32(P["mp"])
    eu, ep = np.float32(P["eu"]), np.float32(P["ep"])

    FC = cuda.device_array((Bn, nn, nn), np.float32)
    FM = cuda.device_array((Bn, nn, nn), np.float32)
    FM1 = cuda.device_array((Bn, nn, nn), np.float32)
    F5 = cuda.device_array((Bn, nn), np.float32)
    exact_kernel[Bn, threads](dS, dL, dFO, dcanon, dbp, dstack, dtm, dhc, ddl, ddr,
                              dhp, dcs, di11, db01, mb, mu, mp, eu, ep, FC, FM, FM1, F5)

    total = Bn * n_samples
    out = cuda.device_array((total, nn), np.int16)
    rng = create_xoroshiro128p_states(total, seed=seed)
    tpb = 64
    blocks = (total + tpb - 1) // tpb
    sample_kernel[blocks, tpb](dS, dL, dFO, dcanon, dbp, dstack, dtm, dhc, ddl, ddr,
                               dhp, dcs, di11, db01, mb, mu, mp, eu, ep,
                               FC, FM, FM1, F5, rng, np.int32(n_samples), out)
    cuda.synchronize()
    outh = out.copy_to_host()

    res = []
    for bi in range(Bn):
        Lb = int(L[bi])
        dbs = []
        for sidx in range(n_samples):
            g = bi * n_samples + sidx
            ch = ['.'] * Lb
            for t in range(1, Lb + 1):
                pj = outh[g, t]
                if pj > t:
                    ch[t - 1] = '('
                    ch[pj - 1] = ')'
            dbs.append(''.join(ch))
        res.append(dbs)
    return res


def fold_tasks_gpu(seqs, forced_masks, P, chunk=4096, seed=0):
    """Pipeline fold primitive: fold a large list of (seq, forced_mask) tasks on
    GPU, 1 Boltzmann sample each. Chunks to bound memory. Returns list of
    dot-bracket strings (one per task). forced_masks[i] may be None.
    Grouped by length internally (sample_batch pads to max len per chunk)."""
    n = len(seqs)
    out = [None] * n
    order = sorted(range(n), key=lambda k: len(seqs[k]))   # group similar lengths
    for c0 in range(0, n, chunk):
        idxs = order[c0:c0 + chunk]
        sub = [seqs[k] for k in idxs]
        fl = [forced_masks[k] for k in idxs] if forced_masks is not None else None
        res = sample_batch(sub, P, 1, forced_list=fl, seed=seed + c0)
        for k, r in zip(idxs, res):
            out[k] = r[0]
    return out


if __name__ == "__main__":
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import cpu
    P = cpu.load()
    tests = ["GGGGAAAACCCC", "GCGCGCAAAAGCGCGCAAAAGCGC",
             "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"]
    g = logZ_batch(tests, P)
    print("=== GPU vs CPU logZ ===")
    for s, gv in zip(tests, g):
        print(f"{s[:36]:36s} GPU={gv:.5f} CPU={cpu.logZ(s, P):.5f}")
    print("\n=== sampling (3 structures of seq[0]) ===")
    for db in sample_batch([tests[0]], P, 3, seed=1)[0]:
        print(" ", db)

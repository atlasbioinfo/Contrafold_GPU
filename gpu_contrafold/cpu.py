"""Faithful CPU port of CONTRAfold's partition function (complementary model).

Replicates InferenceEngine.ipp ComputeInside (simple-FC variant), 1-based
s[1..L], pair (i,j+1) convention, log_base=1.0 (natural log). Goal: logZ
== CONTRAfold binary's --partition output.

Scoring (all enabled terms for the complementary model):
  base_pair, helix_stacking, terminal_mismatch, helix_closing, dangle L/R,
  hairpin_length, bulge_length, internal_length, internal_symmetric,
  internal_asymmetry, internal_explicit (<=4x4), internal_1x1, bulge_0x1.
"""
import os
import numpy as np
import math
from numba import njit

# Bundled trained parameters (CONTRAfold complementary model)
DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "contrafold.params.complementary")

NEG = -1e30
HALF = -5e29
B = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
NBASE = 5   # 0..3 = ACGU, 4 = N/unknown (matches CONTRAfold char_mapping default)
CANON = {(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)}


def code(c):
    return B.get(c, 4)   # non-ACGU -> 4 (N), like CONTRAfold

C_MAX_SINGLE = 30
D_HAIRPIN = 30
D_BULGE = 30
D_INTERNAL = 30
D_ISYM = 15
D_IASYM = 28
D_IEXP = 4


def load(path=None):
    if path is None:
        path = DEFAULT_PARAMS
    raw = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                raw[p[0]] = float(p[1])

    M = NBASE   # size-5 tables; index 4 (N) stays 0, matching CONTRAfold
    bp = np.zeros((M, M))
    for a in 'ACGU':
        for b in 'ACGU':
            v = raw.get(f"base_pair_{a}{b}", raw.get(f"base_pair_{b}{a}", 0.0))
            bp[B[a], B[b]] = v; bp[B[b], B[a]] = v

    stack = np.zeros((M, M, M, M))
    for a in 'ACGU':
        for b in 'ACGU':
            for c in 'ACGU':
                for d in 'ACGU':
                    v = raw.get(f"helix_stacking_{a}{b}{c}{d}",
                                raw.get(f"helix_stacking_{d}{c}{b}{a}", 0.0))
                    stack[B[a], B[b], B[c], B[d]] = v

    tm = np.zeros((M, M, M, M))
    for a in 'ACGU':
        for b in 'ACGU':
            for c in 'ACGU':
                for d in 'ACGU':
                    tm[B[a], B[b], B[c], B[d]] = raw.get(f"terminal_mismatch_{a}{b}{c}{d}", 0.0)

    hc = np.zeros((M, M))
    for a in 'ACGU':
        for b in 'ACGU':
            hc[B[a], B[b]] = raw.get(f"helix_closing_{a}{b}", 0.0)

    dl = np.zeros((M, M, M)); dr = np.zeros((M, M, M))
    for a in 'ACGU':
        for b in 'ACGU':
            for c in 'ACGU':
                dl[B[a], B[b], B[c]] = raw.get(f"dangle_left_{a}{b}{c}", 0.0)
                dr[B[a], B[b], B[c]] = raw.get(f"dangle_right_{a}{b}{c}", 0.0)

    def cum(prefix, dmax, start):
        al = np.zeros(dmax + 1)
        for k in range(start, dmax + 1):
            al[k] = raw.get(f"{prefix}_at_least_{k}", 0.0)
        return np.cumsum(al)

    hp_cum = cum("hairpin_length", D_HAIRPIN, 0)
    bulge_cum = cum("bulge_length", D_BULGE, 1)
    il_cum = cum("internal_length", D_INTERNAL, 2)
    isym_cum = cum("internal_symmetric_length", D_ISYM, 1)
    iasym_cum = cum("internal_asymmetry", D_IASYM, 1)

    iexp = np.zeros((D_IEXP + 1, D_IEXP + 1))
    for l1 in range(1, D_IEXP + 1):
        for l2 in range(1, D_IEXP + 1):
            a, b = min(l1, l2), max(l1, l2)
            iexp[l1, l2] = raw.get(f"internal_explicit_{a}_{b}", 0.0)

    i11 = np.zeros((NBASE, NBASE))
    for a in 'ACGU':
        for b in 'ACGU':
            v = raw.get(f"internal_1x1_nucleotides_{a}{b}",
                        raw.get(f"internal_1x1_nucleotides_{b}{a}", 0.0))
            i11[B[a], B[b]] = v; i11[B[b], B[a]] = v

    b01 = np.zeros(NBASE)
    for a in 'ACGU':
        b01[B[a]] = raw.get(f"bulge_0x1_nucleotides_{a}", 0.0)

    # cache_score_single[l1][l2]
    cs = np.zeros((C_MAX_SINGLE + 1, C_MAX_SINGLE + 1))
    for l1 in range(0, C_MAX_SINGLE + 1):
        for l2 in range(0, C_MAX_SINGLE + 1 - l1):
            if l1 == 0 and l2 == 0:
                continue
            if l1 == 0 or l2 == 0:
                cs[l1, l2] = bulge_cum[min(D_BULGE, l1 + l2)]
            else:
                v = 0.0
                if l1 <= D_IEXP and l2 <= D_IEXP:
                    v += iexp[l1, l2]
                v += il_cum[min(D_INTERNAL, l1 + l2)]
                if l1 == l2:
                    v += isym_cum[min(D_ISYM, l1)]
                v += iasym_cum[min(D_IASYM, abs(l1 - l2))]
                cs[l1, l2] = v

    return dict(bp=bp, stack=stack, tm=tm, hc=hc, dl=dl, dr=dr, hp_cum=hp_cum,
                cs=cs, i11=i11, b01=b01,
                mb=raw["multi_base"], mu=raw["multi_unpaired"], mp=raw["multi_paired"],
                eu=raw["external_unpaired"], ep=raw["external_paired"])


def encode(seq):
    # 1-based: s[1..L]; non-ACGU -> 4 (N). Sentinels s[0],s[L+1]=4 (never pair).
    L = len(seq)
    s = np.full(L + 2, 4, dtype=np.int64)
    for i, c in enumerate(seq.upper()):
        s[i + 1] = code(c)
    return s, L


def canon_mat():
    m = np.zeros((NBASE, NBASE), dtype=np.int64)   # row/col 4 (N) stay 0 -> N never pairs
    for (a, b) in CANON:
        m[a, b] = 1
    return m


@njit(cache=True, inline='always')
def lse(a, b):
    if a < HALF:
        return b
    if b < HALF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def inside(s, L, forced, bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
           mb, mu, mp, eu, ep, canon):
    # allow_paired(a,b): canonical & not forced (1-based a,b in 1..L)
    NEGv = NEG
    FC = np.full((L + 2, L + 2), NEGv)
    FM = np.full((L + 2, L + 2), NEGv)
    FM1 = np.full((L + 2, L + 2), NEGv)
    F5 = np.full(L + 2, NEGv)

    # helper inline score funcs
    # jA(i,j), jB(i,j) use s[i],s[j+1],s[i+1],s[j]
    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            # ---- FM2[i][j] = LSE_{i<k<j} FM1[i][k] + FM[k][j] ----
            FM2 = NEGv
            for k in range(i + 1, j):
                if FM1[i, k] > HALF and FM[k, j] > HALF:
                    FM2 = lse(FM2, FM1[i, k] + FM[k, j])

            # ---- FC[i][j] : pair (i, j+1) ----
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and forced[i] == 0 and forced[j + 1] == 0:
                sum_i = NEGv
                # hairpin: loop length d = j - i (>= C_MIN=0)
                if j - i >= 0:
                    # jB(i,j) = hc[s[i]][s[j+1]] + tm[s[i]][s[j+1]][s[i+1]][s[j]]
                    jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                    sum_i = lse(sum_i, jB + hp_cum[j - i if j - i <= D_HAIRPIN else D_HAIRPIN])
                # single-branch loops (stack/bulge/internal): inner pair (p+1, q)
                pmax = i + C_MAX_SINGLE
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - C_MAX_SINGLE
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        # inner pair (p+1, q)
                        if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                # ScoreSingle
                                jB1 = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = 0.0
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB1 + jB2 + snuc
                            sum_i = lse(sum_i, e + FC[p + 1, q - 1])
                # multibranch
                if FM2 > HALF:
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    sum_i = lse(sum_i, FM2 + jA + mp + mb)
                FC[i, j] = sum_i

            # ---- FM1[i][j] ----
            if 0 < i and i + 2 <= j and j < L:
                sum_i = NEGv
                # FC[i+1][j-1] + jA(j,i) + mp + bp(i+1,j)   [pair (i+1,j)]
                if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    sum_i = lse(sum_i, FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]])
                # FM1[i+1][j] + mu
                if FM1[i + 1, j] > HALF:
                    sum_i = lse(sum_i, FM1[i + 1, j] + mu)
                FM1[i, j] = sum_i

            # ---- FM[i][j] ----
            if 0 < i and i + 2 <= j and j < L:
                sum_i = NEGv
                if FM2 > HALF:
                    sum_i = lse(sum_i, FM2)
                if FM[i, j - 1] > HALF:
                    sum_i = lse(sum_i, FM[i, j - 1] + mu)
                if FM1[i, j] > HALF:
                    sum_i = lse(sum_i, FM1[i, j])
                FM[i, j] = sum_i

    # ---- F5 exterior ----
    F5[0] = 0.0
    for j in range(1, L + 1):
        sum_i = F5[j - 1] + eu       # j unpaired
        for k in range(0, j):
            # branch pair (k+1, j)
            if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF:
                if F5[k] > HALF:
                    # jA(j,k) = hc[s[j]][s[k+1]] + (j<L? dl[s[j]][s[k+1]][s[j+1]]) + (k>0? dr[s[j]][s[k+1]][s[k]])
                    jA = hc[s[j], s[k + 1]]
                    if j < L:
                        jA += dl[s[j], s[k + 1], s[j + 1]]
                    if k > 0:
                        jA += dr[s[j], s[k + 1], s[k]]
                    sum_i = lse(sum_i, F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA)
        F5[j] = sum_i
    return F5[L]


_PARAMS = None
_CANON = canon_mat()


def logZ(seq, P):
    s, L = encode(seq)
    forced = np.zeros(L + 2, dtype=np.int64)
    return inside(s, L, forced, P["bp"], P["stack"], P["tm"], P["hc"], P["dl"], P["dr"],
                  P["hp_cum"], P["cs"], P["i11"], P["b01"],
                  P["mb"], P["mu"], P["mp"], P["eu"], P["ep"], _CANON)


@njit(cache=True)
def viterbi(s, L, forced, bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
            mb, mu, mp, eu, ep, canon):
    """Max-probability (Viterbi/MAP) structure. Same recurrence as `inside` but
    with max instead of log-sum-exp, then an argmax traceback. Returns a partner
    array `pair` (1-based; pair[t] = partner of t, or -1)."""
    NEGv = NEG
    FC = np.full((L + 2, L + 2), NEGv)
    FM = np.full((L + 2, L + 2), NEGv)
    FM1 = np.full((L + 2, L + 2), NEGv)
    F5 = np.full(L + 2, NEGv)

    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            FM2 = NEGv
            for k in range(i + 1, j):
                if FM1[i, k] > HALF and FM[k, j] > HALF:
                    v = FM1[i, k] + FM[k, j]
                    if v > FM2:
                        FM2 = v
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and forced[i] == 0 and forced[j + 1] == 0:
                best = NEGv
                jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                v = jB + hp_cum[j - i if j - i <= D_HAIRPIN else D_HAIRPIN]
                if v > best:
                    best = v
                pmax = i + C_MAX_SINGLE
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - C_MAX_SINGLE
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = 0.0
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB + jB2 + snuc
                            v = e + FC[p + 1, q - 1]
                            if v > best:
                                best = v
                if FM2 > HALF:
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    v = FM2 + jA + mp + mb
                    if v > best:
                        best = v
                FC[i, j] = best
            if 0 < i and i + 2 <= j and j < L:
                best = NEGv
                if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    v = FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]]
                    if v > best:
                        best = v
                if FM1[i + 1, j] > HALF:
                    v = FM1[i + 1, j] + mu
                    if v > best:
                        best = v
                FM1[i, j] = best
            if 0 < i and i + 2 <= j and j < L:
                best = NEGv
                if FM2 > HALF and FM2 > best:
                    best = FM2
                if FM[i, j - 1] > HALF:
                    v = FM[i, j - 1] + mu
                    if v > best:
                        best = v
                if FM1[i, j] > HALF and FM1[i, j] > best:
                    best = FM1[i, j]
                FM[i, j] = best

    F5[0] = 0.0
    for j in range(1, L + 1):
        best = F5[j - 1] + eu
        for k in range(0, j):
            if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF and F5[k] > HALF:
                jA = hc[s[j], s[k + 1]]
                if j < L:
                    jA += dl[s[j], s[k + 1], s[j + 1]]
                if k > 0:
                    jA += dr[s[j], s[k + 1], s[k]]
                v = F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                if v > best:
                    best = v
        F5[j] = best

    # ---- argmax traceback ----
    pair = np.full(L + 2, -1, np.int64)
    EPS = 1e-6
    st_t = np.empty(4 * L + 16, np.int64)
    st_i = np.empty(4 * L + 16, np.int64)
    st_j = np.empty(4 * L + 16, np.int64)
    st_t[0] = 0; st_i[0] = 0; st_j[0] = L
    sp = 1
    while sp > 0:
        sp -= 1
        typ = st_t[sp]; i = st_i[sp]; j = st_j[sp]
        if typ == 0:                       # F5(j)
            if j == 0:
                continue
            val = F5[j]
            done = False
            if F5[j - 1] > HALF and abs(F5[j - 1] + eu - val) <= EPS:
                st_t[sp] = 0; st_i[sp] = 0; st_j[sp] = j - 1; sp += 1; done = True
            if not done:
                for k in range(0, j):
                    if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF and F5[k] > HALF:
                        jA = hc[s[j], s[k + 1]]
                        if j < L:
                            jA += dl[s[j], s[k + 1], s[j + 1]]
                        if k > 0:
                            jA += dr[s[j], s[k + 1], s[k]]
                        v = F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                        if abs(v - val) <= EPS:
                            pair[k + 1] = j; pair[j] = k + 1
                            st_t[sp] = 1; st_i[sp] = k + 1; st_j[sp] = j - 1; sp += 1
                            st_t[sp] = 0; st_i[sp] = 0; st_j[sp] = k; sp += 1
                            done = True; break
        elif typ == 1:                     # FC(i,j): pair (i,j+1) already set
            val = FC[i, j]
            jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
            d = j - i
            done = abs(jB + hp_cum[d if d <= D_HAIRPIN else D_HAIRPIN] - val) <= EPS
            if not done:
                pmax = i + C_MAX_SINGLE
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - C_MAX_SINGLE
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = 0.0
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB + jB2 + snuc
                            if abs(e + FC[p + 1, q - 1] - val) <= EPS:
                                pair[p + 1] = q; pair[q] = p + 1
                                st_t[sp] = 1; st_i[sp] = p + 1; st_j[sp] = q - 1; sp += 1
                                done = True; break
                    if done:
                        break
            if not done:
                jA = hc[s[i], s[j + 1]]
                if i < L:
                    jA += dl[s[i], s[j + 1], s[i + 1]]
                if j > 0:
                    jA += dr[s[i], s[j + 1], s[j]]
                for k in range(i + 1, j):
                    if FM1[i, k] > HALF and FM[k, j] > HALF and abs(FM1[i, k] + FM[k, j] + jA + mp + mb - val) <= EPS:
                        st_t[sp] = 3; st_i[sp] = i; st_j[sp] = k; sp += 1
                        st_t[sp] = 2; st_i[sp] = k; st_j[sp] = j; sp += 1
                        break
        elif typ == 2:                     # FM(i,j)
            val = FM[i, j]
            done = False
            for k in range(i + 1, j):
                if FM1[i, k] > HALF and FM[k, j] > HALF and abs(FM1[i, k] + FM[k, j] - val) <= EPS:
                    st_t[sp] = 3; st_i[sp] = i; st_j[sp] = k; sp += 1
                    st_t[sp] = 2; st_i[sp] = k; st_j[sp] = j; sp += 1
                    done = True; break
            if not done and FM[i, j - 1] > HALF and abs(FM[i, j - 1] + mu - val) <= EPS:
                st_t[sp] = 2; st_i[sp] = i; st_j[sp] = j - 1; sp += 1; done = True
            if not done and FM1[i, j] > HALF and abs(FM1[i, j] - val) <= EPS:
                st_t[sp] = 3; st_i[sp] = i; st_j[sp] = j; sp += 1
        else:                              # FM1(i,j)
            val = FM1[i, j]
            done = False
            if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF:
                jAji = hc[s[j], s[i + 1]]
                if j < L:
                    jAji += dl[s[j], s[i + 1], s[j + 1]]
                if i > 0:
                    jAji += dr[s[j], s[i + 1], s[i]]
                if abs(FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]] - val) <= EPS:
                    pair[i + 1] = j; pair[j] = i + 1
                    st_t[sp] = 1; st_i[sp] = i + 1; st_j[sp] = j - 1; sp += 1
                    done = True
            if not done and FM1[i + 1, j] > HALF and abs(FM1[i + 1, j] + mu - val) <= EPS:
                st_t[sp] = 3; st_i[sp] = i + 1; st_j[sp] = j; sp += 1
    return pair


def mfe(seq, P, forced=None):
    """Maximum-probability (Viterbi) structure as a dot-bracket string."""
    s, L = encode(seq)
    fo = np.zeros(L + 2, dtype=np.int64)
    if forced is not None:
        for k in range(L):
            fo[k + 1] = int(forced[k])
    pair = viterbi(s, L, fo, P["bp"], P["stack"], P["tm"], P["hc"], P["dl"], P["dr"],
                   P["hp_cum"], P["cs"], P["i11"], P["b01"],
                   P["mb"], P["mu"], P["mp"], P["eu"], P["ep"], _CANON)
    ch = ["."] * L
    for t in range(1, L + 1):
        if pair[t] > t:
            ch[t - 1] = "("; ch[pair[t] - 1] = ")"
    return "".join(ch)


F32 = np.float32
NEG32 = F32(-1e30)
HALF32 = F32(-5e29)


@njit(cache=True, inline='always')
def lse32(a, b):
    """log(exp(a)+exp(b)) in float32 — CONTRAfold Fast_LogPlusEquals (RealT=float):
    8-segment Fast_LogExpPlusOne polynomial + hard truncation at 11.8624794162."""
    if a < b:
        t = a; a = b; b = t
    if b < HALF32:
        return a
    d = a - b
    if d >= F32(11.8624794162):
        return a
    if d < F32(3.3792499610):
        if d < F32(1.6320158198):
            if d < F32(0.6615367791):
                r = ((F32(-0.0065591595) * d + F32(0.1276442762)) * d + F32(0.4996554598)) * d + F32(0.6931542306)
            else:
                r = ((F32(-0.0155157557) * d + F32(0.1446775699)) * d + F32(0.4882939746)) * d + F32(0.6958092989)
        elif d < F32(2.4912588184):
            r = ((F32(-0.0128909247) * d + F32(0.1301028251)) * d + F32(0.5150398748)) * d + F32(0.6795585882)
        else:
            r = ((F32(-0.0072142647) * d + F32(0.0877540853)) * d + F32(0.6208708362)) * d + F32(0.5909675829)
    elif d < F32(5.7890710412):
        if d < F32(4.4261691294):
            r = ((F32(-0.0031455354) * d + F32(0.0467229449)) * d + F32(0.7592532310)) * d + F32(0.4348794399)
        else:
            r = ((F32(-0.0010110698) * d + F32(0.0185943421)) * d + F32(0.8831730747)) * d + F32(0.2523695427)
    elif d < F32(7.8162726752):
        r = ((F32(-0.0001962780) * d + F32(0.0046084408)) * d + F32(0.9634431978)) * d + F32(0.0983148903)
    else:
        r = ((F32(-0.0000113994) * d + F32(0.0003734731)) * d + F32(0.9959107193)) * d + F32(0.0149855051)
    return b + r


@njit(cache=True, inline='always')
def fexp32(x):
    """exp(x) in float32 — CONTRAfold Fast_Exp (RealT=float): 6-segment polynomial
    on (-9.91152, 0), 0 below, expf above."""
    if x < F32(-2.4915033807):
        if x < F32(-5.8622823336):
            if x < F32(-9.91152):
                return F32(0.0)
            return ((F32(0.0000803850) * x + F32(0.0021627428)) * x + F32(0.0194708555)) * x + F32(0.0588080014)
        if x < F32(-3.8396630909):
            return ((F32(0.0013889414) * x + F32(0.0244676474)) * x + F32(0.1471290604)) * x + F32(0.3042757740)
        return ((F32(0.0072335607) * x + F32(0.0906002677)) * x + F32(0.3983111356)) * x + F32(0.6245959221)
    if x < F32(-0.6725053211):
        if x < F32(-1.4805375919):
            return ((F32(0.0232410351) * x + F32(0.2085645908)) * x + F32(0.6906367911)) * x + F32(0.8682322329)
        return ((F32(0.0573782771) * x + F32(0.3580258429)) * x + F32(0.9121133217)) * x + F32(0.9793091728)
    if x < F32(0.0):
        return ((F32(0.1199175927) * x + F32(0.4815668234)) * x + F32(0.9975991939)) * x + F32(0.9999505077)
    if x > F32(46.052):
        return F32(1e20)
    return F32(math.exp(x))


@njit(cache=True)
def posterior(s, L, forced, bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
              mb, mu, mp, eu, ep, canon):
    """Base-pair posterior probability matrix via inside + outside.

    Faithful port of CONTRAfold ComputeInside/ComputeOutside/ComputePosterior
    (complementary model: PARAMS_HELIX_LENGTH=0, FAST_SINGLE_BRANCH_LOOPS=1).
    Returns POST[i, j] (1-based, i<j) = P(i pairs j). All arithmetic is in
    float32 with CONTRAfold's exact Fast_LogPlusEquals/Fast_Exp polynomials so
    the result reproduces the RealT=float binary."""
    NEGv = NEG32
    ZERO = F32(0.0)
    # ---------- inside ----------
    FC = np.full((L + 2, L + 2), NEGv)
    FM = np.full((L + 2, L + 2), NEGv)
    FM1 = np.full((L + 2, L + 2), NEGv)
    F5 = np.full(L + 2, NEGv)
    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            FM2 = NEGv
            for k in range(i + 1, j):
                if FM1[i, k] > HALF32 and FM[k, j] > HALF32:
                    FM2 = lse32(FM2, FM1[i, k] + FM[k, j])
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and forced[i] == 0 and forced[j + 1] == 0:
                sum_i = NEGv
                if j - i >= 0:
                    jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                    sum_i = lse32(sum_i, jB + hp_cum[j - i if j - i <= D_HAIRPIN else D_HAIRPIN])
                pmax = i + C_MAX_SINGLE
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - C_MAX_SINGLE
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF32:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB1 = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = ZERO
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB1 + jB2 + snuc
                            sum_i = lse32(sum_i, e + FC[p + 1, q - 1])
                if FM2 > HALF32:
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    sum_i = lse32(sum_i, FM2 + jA + mp + mb)
                FC[i, j] = sum_i
            if 0 < i and i + 2 <= j and j < L:
                sum_i = NEGv
                if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF32:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    sum_i = lse32(sum_i, FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]])
                if FM1[i + 1, j] > HALF32:
                    sum_i = lse32(sum_i, FM1[i + 1, j] + mu)
                FM1[i, j] = sum_i
            if 0 < i and i + 2 <= j and j < L:
                sum_i = NEGv
                if FM2 > HALF32:
                    sum_i = lse32(sum_i, FM2)
                if FM[i, j - 1] > HALF32:
                    sum_i = lse32(sum_i, FM[i, j - 1] + mu)
                if FM1[i, j] > HALF32:
                    sum_i = lse32(sum_i, FM1[i, j])
                FM[i, j] = sum_i
    F5[0] = ZERO
    for j in range(1, L + 1):
        sum_i = F5[j - 1] + eu
        for k in range(0, j):
            if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF32 and F5[k] > HALF32:
                jA = hc[s[j], s[k + 1]]
                if j < L:
                    jA += dl[s[j], s[k + 1], s[j + 1]]
                if k > 0:
                    jA += dr[s[j], s[k + 1], s[k]]
                sum_i = lse32(sum_i, F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA)
        F5[j] = sum_i
    Z = F5[L]

    # ---------- outside ----------
    FCo = np.full((L + 2, L + 2), NEGv)
    FMo = np.full((L + 2, L + 2), NEGv)
    FM1o = np.full((L + 2, L + 2), NEGv)
    F5o = np.full(L + 2, NEGv)
    F5o[L] = ZERO
    for j in range(L, 0, -1):
        F5o[j - 1] = lse32(F5o[j - 1], F5o[j] + eu)
        for k in range(0, j):
            if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0:
                jA = hc[s[j], s[k + 1]]
                if j < L:
                    jA += dl[s[j], s[k + 1], s[j + 1]]
                if k > 0:
                    jA += dr[s[j], s[k + 1], s[k]]
                temp = F5o[j] + ep + bp[s[k + 1], s[j]] + jA
                if FC[k + 1, j - 1] > HALF32:
                    F5o[k] = lse32(F5o[k], temp + FC[k + 1, j - 1])
                if F5[k] > HALF32:
                    FCo[k + 1, j - 1] = lse32(FCo[k + 1, j - 1], temp + F5[k])
    for i in range(0, L + 1):
        for j in range(L, i - 1, -1):
            FM2o = NEGv
            if 0 < i and i + 2 <= j and j < L:
                FM2o = lse32(FM2o, FMo[i, j])
                FMo[i, j - 1] = lse32(FMo[i, j - 1], FMo[i, j] + mu)
                FM1o[i, j] = lse32(FM1o[i, j], FMo[i, j])
            if 0 < i and i + 2 <= j and j < L:
                if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    FCo[i + 1, j - 1] = lse32(FCo[i + 1, j - 1], FM1o[i, j] + jAji + mp + bp[s[i + 1], s[j]])
                FM1o[i + 1, j] = lse32(FM1o[i + 1, j], FM1o[i, j] + mu)
            # FC outer -> inner single-branch + FM2o (complementary #else branch)
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and forced[i] == 0 and forced[j + 1] == 0:
                fco = FCo[i, j]
                if fco > HALF32:
                    jB_ij = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                    pmax = i + C_MAX_SINGLE
                    if pmax > j:
                        pmax = j
                    for p in range(i, pmax + 1):
                        l1 = p - i
                        qmin = p + 2
                        alt = p - i + j - C_MAX_SINGLE
                        if alt > qmin:
                            qmin = alt
                        for q in range(j, qmin - 1, -1):
                            l2 = j - q
                            if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0:
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
                                FCo[p + 1, q - 1] = lse32(FCo[p + 1, q - 1], fco + e)
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    FM2o = lse32(FM2o, fco + jA + mp + mb)
            # distribute FM2o to FM1o[i,k] and FMo[k,j]
            if FM2o > HALF32:
                for k in range(i + 1, j):
                    if FM[k, j] > HALF32:
                        FM1o[i, k] = lse32(FM1o[i, k], FM2o + FM[k, j])
                    if FM1[i, k] > HALF32:
                        FMo[k, j] = lse32(FMo[k, j], FM2o + FM1[i, k])

    # ---------- posterior ----------
    POST = np.zeros((L + 2, L + 2), dtype=np.float32)
    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and forced[i] == 0 and forced[j + 1] == 0:
                outside = FCo[i, j] - Z
                if outside > HALF32:
                    jB_ij = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                    pmax = i + C_MAX_SINGLE
                    if pmax > j:
                        pmax = j
                    for p in range(i, pmax + 1):
                        l1 = p - i
                        qmin = p + 2
                        alt = p - i + j - C_MAX_SINGLE
                        if alt > qmin:
                            qmin = alt
                        for q in range(j, qmin - 1, -1):
                            l2 = j - q
                            if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF32:
                                if p == i and q == j:
                                    e = outside + bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]] + FC[p + 1, q - 1]
                                else:
                                    jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                    snuc = ZERO
                                    if l1 == 0 and l2 == 1:
                                        snuc = b01[s[j]]
                                    elif l1 == 1 and l2 == 0:
                                        snuc = b01[s[i + 1]]
                                    elif l1 == 1 and l2 == 1:
                                        snuc = i11[s[i + 1], s[j]]
                                    e = outside + jB_ij + cs[l1, l2] + FC[p + 1, q - 1] + bp[s[p + 1], s[q]] + jB2 + snuc
                                POST[p + 1, q] += fexp32(e)
            if 0 < i and i + 2 <= j and j < L:
                if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF32 and FM1o[i, j] > HALF32:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    POST[i + 1, j] += fexp32(FM1o[i, j] + FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]] - Z)
    for j in range(1, L + 1):
        outside = F5o[j] - Z
        if outside > HALF32:
            for k in range(0, j):
                if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF32 and F5[k] > HALF32:
                    jA = hc[s[j], s[k + 1]]
                    if j < L:
                        jA += dl[s[j], s[k + 1], s[j + 1]]
                    if k > 0:
                        jA += dr[s[j], s[k + 1], s[k]]
                    POST[k + 1, j] += fexp32(outside + F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA)
    for i in range(1, L + 1):
        for j in range(i + 1, L + 1):
            v = POST[i, j]
            if v < ZERO:
                v = ZERO
            elif v > F32(1.0):
                v = F32(1.0)
            POST[i, j] = v
    return POST


@njit(cache=True)
def mea_decode(POST, L, forced, canon, s, gamma):
    """Maximum-expected-accuracy (posterior) decoding. Faithful port of
    CONTRAfold PredictPairingsPosterior (RealT=float). Returns 1-based partner
    array. `gamma` is the sensitivity/specificity tradeoff."""
    g = F32(gamma)
    two_g = F32(2.0) * g
    # unpaired posterior, scaled by 1/(2*gamma)
    up = np.zeros(L + 1, dtype=np.float32)
    for i in range(1, L + 1):
        u = F32(1.0)
        for j in range(1, i):
            u -= POST[j, i]
        for j in range(i + 1, L + 1):
            u -= POST[i, j]
        up[i] = u / two_g

    score = np.full((L + 1, L + 1), F32(-1.0))
    tb = np.full((L + 1, L + 1), -1, np.int64)
    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            if i == j:
                if F32(0.0) > score[i, j]:
                    score[i, j] = F32(0.0)
                    tb[i, j] = 0
            else:
                # option 1: i+1 unpaired
                v = up[i + 1] + score[i + 1, j]
                if v > score[i, j]:
                    score[i, j] = v
                    tb[i, j] = 1
                # option 2: j unpaired
                v = up[j] + score[i, j - 1]
                if v > score[i, j]:
                    score[i, j] = v
                    tb[i, j] = 2
                if i + 2 <= j:
                    # option 3: pair (i+1, j)
                    if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0:
                        v = POST[i + 1, j] + score[i + 1, j - 1]
                        if v > score[i, j]:
                            score[i, j] = v
                            tb[i, j] = 3
                    # bifurcation
                    for k in range(i + 1, j):
                        v = score[i, k] + score[k, j]
                        if v > score[i, j]:
                            score[i, j] = v
                            tb[i, j] = k + 4

    pair = np.full(L + 2, -1, np.int64)
    # iterative traceback (queue via stack arrays)
    qi = np.empty(2 * L + 4, np.int64)
    qj = np.empty(2 * L + 4, np.int64)
    qi[0] = 0; qj[0] = L
    head = 0; tail = 1
    while head < tail:
        i = qi[head]; j = qj[head]; head += 1
        t = tb[i, j]
        if t <= 0:
            continue
        if t == 1:
            qi[tail] = i + 1; qj[tail] = j; tail += 1
        elif t == 2:
            qi[tail] = i; qj[tail] = j - 1; tail += 1
        elif t == 3:
            pair[i + 1] = j; pair[j] = i + 1
            qi[tail] = i + 1; qj[tail] = j - 1; tail += 1
        else:
            k = t - 4
            qi[tail] = i; qj[tail] = k; tail += 1
            qi[tail] = k; qj[tail] = j; tail += 1
    return pair


def _params_f32(P):
    """Cache float32 copies of the parameter tables (CONTRAfold uses RealT=float;
    the MEA/posterior path matches the binary only in float32)."""
    f = P.get("_f32")
    if f is None:
        f = {k: (np.float32(P[k]) if np.isscalar(P[k]) or P[k].ndim == 0
                 else P[k].astype(np.float32))
             for k in ("bp", "stack", "tm", "hc", "dl", "dr", "hp_cum", "cs",
                       "i11", "b01")}
        for k in ("mb", "mu", "mp", "eu", "ep"):
            f[k] = np.float32(P[k])
        P["_f32"] = f
    return f


def _posterior(seq, P, forced=None):
    s, L = encode(seq)
    fo = np.zeros(L + 2, dtype=np.int64)
    if forced is not None:
        for k in range(L):
            fo[k + 1] = int(forced[k])
    f = _params_f32(P)
    POST = posterior(s, L, fo, f["bp"], f["stack"], f["tm"], f["hc"], f["dl"], f["dr"],
                     f["hp_cum"], f["cs"], f["i11"], f["b01"],
                     f["mb"], f["mu"], f["mp"], f["eu"], f["ep"], _CANON)
    return s, L, fo, POST


def bpp(seq, P, forced=None):
    """Base-pair probability matrix (n x n upper triangle, 0-based) for `seq`."""
    _s, L, _fo, POST = _posterior(seq, P, forced)
    return POST[1:L + 1, 1:L + 1].astype(np.float64)


def mea(seq, P, gamma=6.0, forced=None):
    """Maximum-expected-accuracy structure (CONTRAfold default decoding) as
    dot-bracket. `gamma` is the sensitivity/specificity tradeoff (default 6)."""
    s, L, fo, POST = _posterior(seq, P, forced)
    pair = mea_decode(POST, L, fo, _CANON, s, np.float32(gamma))
    ch = ["."] * L
    for t in range(1, L + 1):
        if pair[t] > t:
            ch[t - 1] = "("; ch[pair[t] - 1] = ")"
    return "".join(ch)


if __name__ == "__main__":
    P = load()
    for s in ["GGGGAAAACCCC", "GCGCGCAAAAGCGCGCAAAAGCGC"]:
        print(f"{s}  logZ={logZ(s, P):.5f}  mea={mea(s, P)}")

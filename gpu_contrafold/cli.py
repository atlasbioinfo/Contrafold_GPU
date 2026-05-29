"""Batch RNA folding CLI for gpu_contrafold.

Fold a single sequence, a JSONL file, or a FASTA file on the GPU.

    gpu-contrafold GGGGAAAACCCC                  # one sequence -> MFE structure (dot-bracket)
    gpu-contrafold GGGGAAAACCCC --sample 10      # 10 Boltzmann samples
    gpu-contrafold GGGGAAAACCCC --logz           # partition function (logZ) instead
    gpu-contrafold seqs.jsonl -o out.jsonl       # batch (JSONL in/out)

By default the maximum-probability (Viterbi/MAP) structure is returned — the same
structure as `contrafold --viterbi`, deterministic. Use --sample N for N Boltzmann
samples, or --logz for the partition function.

The positional argument is a literal RNA sequence if it is not an existing file;
otherwise it is read as JSONL (lines starting with `{`), FASTA (`>`), or a plain
list of sequences (one per line).

JSONL input (one object per line):
    {"id": "rna1", "seq": "GGGGAAAACCCC", "constrain": [0,0,0,1,0,0,0,0,0,0,0,0]}
  - seq        (required) RNA sequence; non-ACGU is treated as N (cannot pair).
  - id         (optional) echoed to output; defaults to the 0-based index.
  - constrain  (optional) per-base 0/1 hard-constraint mask, length == len(seq),
               1 = forced unpaired. Accepts a list [0,0,1,...] or a string "001...".

Output: a single literal sequence prints structures to stdout (one per line);
file input (or any -o) writes JSONL ({"id","structure"}, or {"id","samples":[...]}
for --sample N, or {"id","logZ"} for --logz), input order preserved.
"""
import argparse
import json
import os
import sys

import numpy as np

from . import cpu, gpu

_SEQ_CHARS = set("ACGUTN")


def _parse_jsonl(path):
    recs = []
    with open(path) as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"line {ln}: invalid JSON ({e})")
            if "seq" not in obj or not obj["seq"]:
                raise SystemExit(f"line {ln}: missing required field 'seq'")
            recs.append((obj.get("id", ln), str(obj["seq"]).strip().upper(), obj.get("constrain")))
    return recs


def _parse_fasta(path):
    recs, rid, cur = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur:
                    recs.append((rid, "".join(cur).upper(), None)); cur = []
                rid = line[1:].strip() or len(recs)
            elif line.strip():
                cur.append(line.strip())
    if cur:
        recs.append((rid, "".join(cur).upper(), None))
    return recs


def _parse_seq_lines(path):
    recs = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip().upper()
            if s:
                recs.append((i, s, None))
    return recs


def load_records(src):
    """Return (records, from_file). records: list of (id, seq, constrain)."""
    if os.path.isfile(src):
        head = ""
        with open(src) as fh:
            for line in fh:
                if line.strip():
                    head = line.strip()[0]
                    break
        if head == "{":
            return _parse_jsonl(src), True
        if head == ">":
            return _parse_fasta(src), True
        return _parse_seq_lines(src), True
    seq = src.strip().upper()
    if seq and all(c in _SEQ_CHARS for c in seq):
        return [(0, seq, None)], False
    raise SystemExit(f"'{src}' is not an existing file nor a valid RNA sequence (ACGUTN)")


def build_mask(seq, constrain, where=""):
    """Per-base 0/1 mask (1 = forced unpaired) -> int8 array. None if no constraint.

    Accepts a list of 0/1 ints or a string of '0'/'1'; length must equal len(seq).
    """
    if not constrain:
        return None
    L = len(seq)
    bits = [1 if c == "1" else 0 for c in constrain] if isinstance(constrain, str) else list(constrain)
    if len(bits) != L:
        raise SystemExit(f"{where}constrain length {len(bits)} != seq length {L}")
    m = np.zeros(L, np.int8)
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise SystemExit(f"{where}constrain[{i}] = {b!r}, expected 0 or 1")
        m[i] = b
    return m


def fold(records, P, sample_n=0, with_logz=False, chunk=4096, threads=128, seed=0):
    """Batch-fold records on GPU, grouped by length, results mapped to input order."""
    n = len(records)
    seqs = [r[1] for r in records]
    masks = [build_mask(r[1], r[2], where=f"id={r[0]!r}: ") for r in records]
    order = sorted(range(n), key=lambda k: len(seqs[k]))
    logz = [None] * n
    samples = [None] * n
    for c0 in range(0, n, chunk):
        idx = order[c0:c0 + chunk]
        sub = [seqs[k] for k in idx]
        fl = [masks[k] for k in idx]
        if sample_n:
            for k, r in zip(idx, gpu.sample_batch(sub, P, sample_n, forced_list=fl, threads=threads, seed=seed + c0)):
                samples[k] = r
            if with_logz:
                for k, z in zip(idx, gpu.logZ_batch(sub, P, forced_list=fl, threads=threads)):
                    logz[k] = z
        else:
            for k, z in zip(idx, gpu.logZ_batch(sub, P, forced_list=fl, threads=threads)):
                logz[k] = z
    return logz, samples


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fold":          # back-compat: optional 'fold' subcommand
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="gpu-contrafold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="RNA sequence, or path to a JSONL/FASTA/sequence-per-line file")
    parser.add_argument("-o", "--output", default=None, help="output JSONL (default: stdout)")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="draw N Boltzmann samples per sequence (default: MFE structure)")
    parser.add_argument("--logz", action="store_true",
                        help="emit the partition function (logZ) instead of a structure")
    parser.add_argument("--chunk", type=int, default=4096, help="sequences per GPU launch")
    parser.add_argument("--threads", type=int, default=128, help="GPU threads per block")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    args = parser.parse_args(argv)

    records, from_file = load_records(args.input)
    if not records:
        raise SystemExit("no sequences in input")

    # mode: default = MFE (Viterbi) structure; --sample N = Boltzmann samples; --logz = partition fn
    mode = "logz" if (args.logz and not args.sample) else ("sample" if args.sample else "mfe")

    P = cpu.load()
    logz = samples = structures = None
    if mode == "mfe":
        structures = [cpu.mfe(seq, P, build_mask(seq, con, where=f"id={rid!r}: "))
                      for (rid, seq, con) in records]
    elif mode == "sample":
        logz, samples = fold(records, P, sample_n=args.sample, with_logz=args.logz,
                             chunk=args.chunk, threads=args.threads, seed=args.seed)
    else:  # logz
        logz, _ = fold(records, P, sample_n=0, chunk=args.chunk, threads=args.threads)

    # single literal sequence with no -o: human-readable stdout
    if not from_file and not args.output:
        if mode == "mfe":
            print(structures[0])
        elif mode == "sample":
            if args.logz:
                print(f"# logZ = {logz[0]:.6f}")
            for db in samples[0]:
                print(db)
        else:
            print(f"{logz[0]:.6f}")
        return

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        for k, (rid, _seq, _con) in enumerate(records):
            rec = {"id": rid}
            if mode == "mfe":
                rec["structure"] = structures[k]
            elif mode == "sample":
                if args.logz:
                    rec["logZ"] = logz[k]
                rec["samples"] = samples[k]
            else:
                rec["logZ"] = logz[k]
            out.write(json.dumps(rec) + "\n")
    finally:
        if args.output:
            out.close()
    if args.output:
        print(f"[gpu-contrafold] wrote {len(records)} records -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

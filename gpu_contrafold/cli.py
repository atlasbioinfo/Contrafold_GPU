"""Batch RNA folding CLI: JSONL in, JSONL out.

Input (one JSON object per line):
    {"id": "rna1", "seq": "GGGGAAAACCCC", "constrain": [0,0,0,1,0,0,0,0,0,0,0,0]}
  - seq        (required) RNA sequence; non-ACGU is treated as N (cannot pair).
  - id         (optional) echoed to output; defaults to the 0-based line index.
  - constrain  (optional) per-base 0/1 hard-constraint mask, length == len(seq),
               1 = forced unpaired (e.g. DMS-reactive position). Accepts a list
               [0,0,0,1,...] or a string "000100...". Omit or [] for none.

Output (one JSON object per line, input order preserved):
  - default:          {"id": ..., "logZ": <float>}
  - with --sample N:  {"id": ..., "samples": ["((..))", ...]}
                      (add --logz to also include "logZ")

Usage:
    python -m gpu_contrafold fold in.jsonl -o out.jsonl
    python -m gpu_contrafold fold in.jsonl -o out.jsonl --sample 100 --seed 0
"""
import argparse
import json
import sys

import numpy as np

from . import cpu, gpu


def parse_records(path):
    """Read JSONL -> list of (id, seq, constrain). Raises on malformed lines."""
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
            recs.append((obj.get("id", ln), str(obj["seq"]).strip().upper(),
                         obj.get("constrain")))
    return recs


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
    """Batch-fold records on GPU, grouped by length, results mapped back to input order."""
    n = len(records)
    seqs = [r[1] for r in records]
    masks = [build_mask(r[1], r[2], where=f"id={r[0]!r}: ") for r in records]
    order = sorted(range(n), key=lambda k: len(seqs[k]))   # group similar lengths per chunk
    logz = [None] * n
    samples = [None] * n
    for c0 in range(0, n, chunk):
        idx = order[c0:c0 + chunk]
        sub = [seqs[k] for k in idx]
        fl = [masks[k] for k in idx]
        if sample_n:
            res = gpu.sample_batch(sub, P, sample_n, forced_list=fl, threads=threads, seed=seed + c0)
            for k, r in zip(idx, res):
                samples[k] = r
            if with_logz:
                for k, z in zip(idx, gpu.logZ_batch(sub, P, forced_list=fl, threads=threads)):
                    logz[k] = z
        else:
            for k, z in zip(idx, gpu.logZ_batch(sub, P, forced_list=fl, threads=threads)):
                logz[k] = z
    return logz, samples


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpu-contrafold", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fold", help="batch-fold a JSONL file of RNA sequences",
                        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    pf.add_argument("input", help="input JSONL")
    pf.add_argument("-o", "--output", default=None, help="output JSONL (default: stdout)")
    pf.add_argument("--sample", type=int, default=0, metavar="N",
                    help="draw N Boltzmann samples per sequence (default: 0 = logZ only)")
    pf.add_argument("--logz", action="store_true",
                    help="in --sample mode, also emit logZ (adds a second pass)")
    pf.add_argument("--chunk", type=int, default=4096, help="sequences per GPU launch")
    pf.add_argument("--threads", type=int, default=128, help="GPU threads per block")
    pf.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    args = parser.parse_args(argv)

    records = parse_records(args.input)
    if not records:
        raise SystemExit("no records in input")
    print(f"[gpu-contrafold] folding {len(records)} sequences "
          f"({'sample ' + str(args.sample) if args.sample else 'logZ'}, chunk={args.chunk})",
          file=sys.stderr)

    P = cpu.load()
    logz, samples = fold(records, P, sample_n=args.sample, with_logz=args.logz,
                         chunk=args.chunk, threads=args.threads, seed=args.seed)

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        for k, (rid, _seq, _con) in enumerate(records):
            rec = {"id": rid}
            if args.sample:
                if args.logz:
                    rec["logZ"] = logz[k]
                rec["samples"] = samples[k]
            else:
                rec["logZ"] = logz[k]
            out.write(json.dumps(rec) + "\n")
    finally:
        if args.output:
            out.close()
    print(f"[gpu-contrafold] wrote {len(records)} records"
          + (f" -> {args.output}" if args.output else ""), file=sys.stderr)


if __name__ == "__main__":
    main()

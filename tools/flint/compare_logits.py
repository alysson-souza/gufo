#!/usr/bin/env python3
# /// script
# dependencies = ["numpy"]
# ///
"""Compare per-position Flint logits from a teacher (llama.cpp) and a candidate.

Both dumps are float32 rows of VOCAB logits, one per position of the same
token ids. Reports per position and aggregated:
  KL(teacher || candidate), top-1 agreement, top-5 overlap,
  max |delta log-softmax| over the vocabulary, and the NLL of the next prompt
  token under each side.

  uv run tools/flint/compare_logits.py --teacher llama.f32 --candidate ref.f32 \
      --tokens ids.txt [--json out.json] [--per-position]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

VOCAB = 150144


def log_softmax(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float64)
    peak = rows.max(axis=1, keepdims=True)
    shifted = rows - peak
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def load(path: Path, vocab: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size % vocab:
        raise SystemExit(f"{path}: {data.size} floats is not a multiple of {vocab}")
    return data.reshape(-1, vocab)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--per-position", action="store_true")
    parser.add_argument("--from-position", type=int, default=0,
                        help="aggregate only positions >= N (all are listed)")
    args = parser.parse_args()

    ids = [int(x) for x in re.findall(r"\d+", args.tokens.read_text())]
    teacher = load(args.teacher, args.vocab)
    candidate = load(args.candidate, args.vocab)
    if teacher.shape != candidate.shape or teacher.shape[0] != len(ids):
        raise SystemExit(
            f"shape mismatch: teacher {teacher.shape}, candidate "
            f"{candidate.shape}, {len(ids)} tokens")
    nonfinite = {
        "teacher": int((~np.isfinite(teacher)).sum()),
        "candidate": int((~np.isfinite(candidate)).sum()),
    }

    rows = []
    chunk = 64
    for start in range(0, len(ids), chunk):
        lt = log_softmax(teacher[start:start + chunk])
        lc = log_softmax(candidate[start:start + chunk])
        pt = np.exp(lt)
        kl = (pt * (lt - lc)).sum(axis=1)
        max_delta = np.abs(lt - lc).max(axis=1)
        top_t = np.argsort(-lt, axis=1, kind="stable")[:, :5]
        top_c = np.argsort(-lc, axis=1, kind="stable")[:, :5]
        for i in range(lt.shape[0]):
            p = start + i
            row = {
                "position": p,
                "kl": float(kl[i]),
                "top1_agree": bool(top_t[i, 0] == top_c[i, 0]),
                "top5_overlap": len(set(top_t[i]) & set(top_c[i])) / 5.0,
                "max_abs_dlogp": float(max_delta[i]),
                "teacher_top1": int(top_t[i, 0]),
                "candidate_top1": int(top_c[i, 0]),
            }
            if p + 1 < len(ids):
                row["nll_teacher"] = float(-lt[i, ids[p + 1]])
                row["nll_candidate"] = float(-lc[i, ids[p + 1]])
            rows.append(row)

    summarized = [r for r in rows if r["position"] >= args.from_position]
    kls = np.array([r["kl"] for r in summarized])
    scored = [r for r in summarized if "nll_teacher" in r]
    summary = {
        "positions": len(summarized),
        "from_position": args.from_position,
        "nonfinite": nonfinite,
        "kl_mean": float(kls.mean()),
        "kl_p99": float(np.quantile(kls, 0.99)),
        "kl_max": float(kls.max()),
        "kl_max_position": summarized[int(kls.argmax())]["position"],
        "top1_agreement": float(np.mean([r["top1_agree"] for r in summarized])),
        "top5_overlap": float(np.mean([r["top5_overlap"] for r in summarized])),
        "max_abs_dlogp": float(max(r["max_abs_dlogp"] for r in summarized)),
        "nll_teacher_mean": float(np.mean([r["nll_teacher"] for r in scored])),
        "nll_candidate_mean": float(np.mean([r["nll_candidate"] for r in scored])),
    }
    if args.per_position:
        for r in rows:
            print(json.dumps(r))
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "positions": rows},
                                        indent=1) + "\n")


if __name__ == "__main__":
    main()

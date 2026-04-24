"""
Run the scorer directly over the labeled public eval JSON and report
accuracy. Useful for fast iteration: no HTTP, no uvicorn.

Usage:
    python eval_local.py /path/to/public_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.features import extract_pair_features
from app.scoring import load_scorer
from train import iter_labeled_pairs  # re-use the same loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--label-key", default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="print a few misclassified examples")
    args = ap.parse_args()

    with open(args.data, "r") as f:
        blob = json.load(f)

    scorer = load_scorer()
    print(f"Scorer source: {scorer.source}  threshold={scorer.threshold:.3f}")

    n = 0
    correct = 0
    confusion: Counter[tuple[int, int]] = Counter()
    mistakes: list[dict] = []

    for _case_id, cur_desc, cur_date, pri_desc, pri_date, label in iter_labeled_pairs(blob, args.label_key):
        feats = extract_pair_features(cur_desc, cur_date, pri_desc, pri_date)
        pred, prob = scorer.predict(feats)
        y = 1 if label else 0
        p = 1 if pred else 0
        confusion[(y, p)] += 1
        if p == y:
            correct += 1
        elif args.verbose and len(mistakes) < 20:
            mistakes.append({
                "current": cur_desc, "current_date": cur_date,
                "prior": pri_desc, "prior_date": pri_date,
                "label": bool(label), "pred": pred, "prob": round(prob, 3),
            })
        n += 1

    if n == 0:
        print("No labeled pairs found. Check --label-key.", file=sys.stderr)
        sys.exit(2)

    tp = confusion[(1, 1)]; fn = confusion[(1, 0)]
    fp = confusion[(0, 1)]; tn = confusion[(0, 0)]
    acc = correct / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"Pairs evaluated: {n}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"Confusion  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    if args.verbose:
        print("\nSample mistakes:")
        for m in mistakes:
            print(m)


if __name__ == "__main__":
    main()

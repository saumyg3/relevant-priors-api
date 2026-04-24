"""
Train the linear scorer on the public labeled split.

Usage:
    python train.py /path/to/public_eval.json
    python train.py /path/to/public_eval.json --threshold-search
    python train.py /path/to/public_eval.json --out app/model.json

What this expects the public JSON to look like
-----------------------------------------------
The submission brief doesn't fully document the label field on the public
split, but the portal describes it as "996 cases, 27,614 labeled previous
exams". The likely shapes are one of:

    # Shape A: labels attached to each prior
    {"cases": [
      {"case_id": "...", "current_study": {...},
       "prior_studies": [
         {"study_id": "...", "study_description": "...", "study_date": "...",
          "is_relevant": true},
         ...
       ]}
    ]}

    # Shape B: labels in a parallel array
    {"cases": [...], "labels": [
      {"case_id": "...", "study_id": "...", "is_relevant": true}, ...
    ]}

This script handles both. If your downloaded JSON uses a different key
for the label (e.g. "label", "relevant", "gold"), pass --label-key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from typing import Iterable

# sklearn is only needed at training time, not at serve time, so it
# stays out of the runtime requirements file.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
except ImportError:
    print("Training requires scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

# Make "from app.features import ..." work whether run from repo root or app/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.features import PairFeatures, extract_pair_features


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

LABEL_KEYS_TRIED = ("is_relevant", "label", "relevant", "gold", "y", "target")


def iter_labeled_pairs(blob: dict, label_key: str | None = None):
    """Yield (current_desc, current_date, prior_desc, prior_date, label) tuples.

    Supports both Shape A (label on each prior) and Shape B (parallel
    labels array keyed by (case_id, study_id)).
    """
    cases = blob.get("cases", [])

    # Shape B: build lookup first if a top-level labels array exists.
    # The public split uses key "truth" with field "is_relevant_to_current",
    # so we check several common container and field names.
    labels_index: dict[tuple[str, str], bool] = {}
    labels_array = None
    for container_key in ("labels", "truth", "ground_truth", "gold"):
        if isinstance(blob.get(container_key), list):
            labels_array = blob[container_key]
            break

    if labels_array is not None:
        for row in labels_array:
            case_id = str(row.get("case_id"))
            study_id = str(row.get("study_id"))
            if label_key and label_key in row:
                labels_index[(case_id, study_id)] = bool(row[label_key])
            else:
                # Extended set of possible label field names — covers the
                # public split's "is_relevant_to_current" and common variants.
                for k in LABEL_KEYS_TRIED + ("is_relevant_to_current",):
                    if k in row:
                        labels_index[(case_id, study_id)] = bool(row[k])
                        break

    for case in cases:
        case_id = str(case.get("case_id"))
        cur = case.get("current_study") or {}
        cur_desc = cur.get("study_description", "") or ""
        cur_date = cur.get("study_date")
        for prior in case.get("prior_studies", []):
            study_id = str(prior.get("study_id"))
            pri_desc = prior.get("study_description", "") or ""
            pri_date = prior.get("study_date")

            # Shape A first: label on the prior itself.
            label = None
            if label_key and label_key in prior:
                label = bool(prior[label_key])
            else:
                for k in LABEL_KEYS_TRIED:
                    if k in prior:
                        label = bool(prior[k])
                        break

            # Shape B fallback: parallel labels array.
            if label is None:
                label = labels_index.get((case_id, study_id))

            if label is None:
                continue  # unlabeled — skip for training

            yield cur_desc, cur_date, pri_desc, pri_date, label


def build_dataset(path: str, label_key: str | None):
    with open(path, "r") as f:
        blob = json.load(f)

    X, y = [], []
    for cur_desc, cur_date, pri_desc, pri_date, label in iter_labeled_pairs(blob, label_key):
        feats = extract_pair_features(cur_desc, cur_date, pri_desc, pri_date)
        X.append(feats.to_vector())
        y.append(1 if label else 0)

    if not X:
        print(
            "ERROR: no labeled pairs found. Inspect the JSON and pass "
            "--label-key <field> if your labels live under a different key.",
            file=sys.stderr,
        )
        sys.exit(2)

    return X, y


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def split(X, y, frac: float = 0.2, seed: int = 13):
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - frac))
    tr, va = idx[:cut], idx[cut:]
    return (
        [X[i] for i in tr], [y[i] for i in tr],
        [X[i] for i in va], [y[i] for i in va],
    )


def find_best_threshold(probs, y_true) -> tuple[float, float]:
    """Sweep thresholds and return (threshold, accuracy)."""
    best_t, best_acc = 0.5, 0.0
    for t_int in range(5, 96):
        t = t_int / 100.0
        preds = [1 if p >= t else 0 for p in probs]
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="path to public_eval.json")
    ap.add_argument("--out", default="app/model.json")
    ap.add_argument("--label-key", default=None,
                    help="override label field name if auto-detect fails")
    ap.add_argument("--threshold-search", action="store_true",
                    help="sweep validation thresholds instead of fixing at 0.5")
    ap.add_argument("--C", type=float, default=1.0,
                    help="L2 regularization inverse strength")
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    X, y = build_dataset(args.data, args.label_key)
    pos = sum(y)
    print(f"Loaded {len(y)} labeled pairs. positive rate = {pos/len(y):.3f}")

    X_tr, y_tr, X_va, y_va = split(X, y)
    print(f"Train: {len(y_tr)}  Val: {len(y_va)}")

    clf = LogisticRegression(
        C=args.C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_tr, y_tr)

    val_probs = clf.predict_proba(X_va)[:, 1]
    val_preds_default = (val_probs >= 0.5).astype(int)
    default_acc = accuracy_score(y_va, val_preds_default)

    try:
        auc = roc_auc_score(y_va, val_probs)
    except ValueError:
        auc = float("nan")

    print()
    print(f"Validation accuracy @ 0.5 = {default_acc:.4f}")
    print(f"Validation ROC-AUC       = {auc:.4f}")
    print()
    print("Classification report @ 0.5:")
    print(classification_report(y_va, val_preds_default, digits=4))

    threshold = 0.5
    if args.threshold_search:
        t, acc = find_best_threshold(val_probs.tolist(), y_va)
        print(f"Best threshold by val accuracy: {t:.2f}  (acc={acc:.4f})")
        threshold = t

    feature_names = PairFeatures.feature_names()
    weights = {name: float(w) for name, w in zip(feature_names, clf.coef_[0])}
    intercept = float(clf.intercept_[0])

    print()
    print("Learned weights (sorted by magnitude):")
    for name, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:28s} {w:+.4f}")
    print(f"  {'intercept':28s} {intercept:+.4f}")

    blob = {
        "intercept": intercept,
        "weights": weights,
        "threshold": threshold,
        "trained_on": os.path.basename(args.data),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "validation_accuracy": default_acc,
        "validation_auc": auc if not math.isnan(auc) else None,
        "n_train": len(y_tr),
        "n_val": len(y_va),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

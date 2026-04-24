"""
Scoring for prior-relevance.

Two modes:

1. Trained: if `model.json` exists next to this file, load logistic-regression
   weights fit on the public 27K-prior labeled split. This is what you want
   for the final submission.

2. Zero-shot fallback: if no model file is present, score with a hand-tuned
   linear combination of the same features. This makes the endpoint return
   reasonable predictions immediately (useful for the portal's quick API
   check, for CI smoke tests, and as a safety net).

The model format is plain JSON so you can eyeball the learned weights:

    {
      "intercept": -1.23,
      "weights": {"same_modality": 0.8, "any_region_match": 1.4, ...},
      "threshold": 0.5,
      "trained_on": "public_eval.json",
      "trained_at": "2026-04-24T..."
    }
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Mapping

from .features import PairFeatures, extract_pair_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")


# -----------------------------------------------------------------------------
# Zero-shot weights (reasonable priors, not fit to data)
# -----------------------------------------------------------------------------
# Intuition:
#   - Region overlap is the single biggest signal. A prior on the wrong body
#     part is almost never relevant.
#   - Same modality matters but less than region. A prior CT head is often
#     relevant for reading an MRI head.
#   - Recency helps but radiology priors stay useful for years — esp. for
#     oncology and chronic conditions — so we don't decay aggressively.
#   - Exact description match is a very strong signal.
#   - Modality-both-unknown is a mild boost: if we can't parse either, we
#     fall back to "probably relevant" rather than rejecting everything.

ZERO_SHOT_WEIGHTS: Mapping[str, float] = {
    "exact_desc_match":       2.2,
    "same_modality":          0.9,
    "modality_both_unknown":  0.3,
    "region_overlap":         0.6,
    "region_jaccard":         1.4,
    "any_region_match":       1.8,
    "context_overlap":        0.5,
    "contrast_conflict":     -0.4,
    "years_delta":           -0.08,   # mild decay per year
    "is_recent_1y":           0.3,
    "is_very_recent_90d":     0.2,
    "is_stale_5y":           -0.2,
    "missing_date":           0.0,
    "prior_desc_len_scaled":  0.0,
    "laterality_conflict":   -1.2,
    "laterality_match":       0.4,
}
ZERO_SHOT_INTERCEPT = -0.8
ZERO_SHOT_THRESHOLD = 0.5


@dataclass
class Scorer:
    intercept: float
    weights: Mapping[str, float]
    threshold: float
    source: str  # "trained" or "zero_shot"

    def score(self, feats: PairFeatures) -> float:
        """Return probability in [0, 1] that the prior is relevant."""
        names = PairFeatures.feature_names()
        values = feats.to_vector()
        z = self.intercept
        for name, val in zip(names, values):
            z += self.weights.get(name, 0.0) * val
        # Clamp to avoid math.exp overflow on wild inputs.
        z = max(-40.0, min(40.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, feats: PairFeatures) -> tuple[bool, float]:
        p = self.score(feats)
        return (p >= self.threshold, p)


def load_scorer() -> Scorer:
    """Load trained weights from disk if available, else fall back."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r") as f:
            blob = json.load(f)
        return Scorer(
            intercept=float(blob["intercept"]),
            weights={k: float(v) for k, v in blob["weights"].items()},
            threshold=float(blob.get("threshold", 0.5)),
            source="trained",
        )
    return Scorer(
        intercept=ZERO_SHOT_INTERCEPT,
        weights=dict(ZERO_SHOT_WEIGHTS),
        threshold=ZERO_SHOT_THRESHOLD,
        source="zero_shot",
    )


def score_pair(
    scorer: Scorer,
    current_desc: str,
    current_date: str | None,
    prior_desc: str,
    prior_date: str | None,
) -> tuple[bool, float]:
    feats = extract_pair_features(current_desc, current_date, prior_desc, prior_date)
    return scorer.predict(feats)

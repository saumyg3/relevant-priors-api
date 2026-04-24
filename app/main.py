"""
FastAPI endpoint for the New Grad Residency "relevant-priors-v1" challenge.

Contract (from the brief):
  POST /predict
  Request body: {
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "...",
    "cases": [
      {
        "case_id": "...",
        "patient_id": "...",
        "patient_name": "...",
        "current_study": {"study_id": "...", "study_description": "...", "study_date": "..."},
        "prior_studies": [
          {"study_id": "...", "study_description": "...", "study_date": "..."},
          ...
        ]
      },
      ...
    ]
  }
  Response: {"predictions": [{"case_id", "study_id", "predicted_is_relevant"}, ...]}

Design notes
------------
- Every prior in the request MUST produce exactly one prediction. Skipping
  counts as incorrect per the brief, so we never drop on error — we fall
  back to predict=True (biasing toward showing the prior) if something
  explodes mid-pair, on the theory that a false positive (showing an
  irrelevant prior) is cheaper than a false negative (hiding a relevant
  one) in a clinical UX.
- Scoring is pure CPU and very fast (rule features + linear model), so
  there is no external API call on the hot path. This keeps us well under
  the 360s evaluator budget even for the full private split.
- An LRU cache on (normalized_current_desc, normalized_prior_desc,
  days_delta_bucket) deduplicates the enormous amount of repeated work
  we expect — the public split has 27,614 priors across 996 cases and
  description vocabulary is small.
"""

from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .features import normalize, days_between, extract_pair_features
from .scoring import Scorer, load_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("priors")

app = FastAPI(title="Relevant Priors", version="1.0.0")

# Load once at startup.
SCORER: Scorer = load_scorer()
logger.info("Scorer loaded: source=%s threshold=%.3f", SCORER.source, SCORER.threshold)


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class Study(BaseModel):
    study_id: str
    study_description: str = ""
    study_date: Optional[str] = None


class Case(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    current_study: Study
    prior_studies: List[Study] = Field(default_factory=list)


class PredictRequest(BaseModel):
    challenge_id: Optional[str] = None
    schema_version: Optional[int] = None
    generated_at: Optional[str] = None
    cases: List[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: List[Prediction]


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------

def _days_bucket(days: Optional[int]) -> int:
    """Discretize days-delta so near-identical ages share a cache slot."""
    if days is None:
        return -1
    d = abs(days)
    # Fine buckets at short ranges, coarser as priors get older.
    if d <= 30:
        return d // 7          # weekly
    if d <= 365:
        return 10 + d // 30    # monthly
    return 30 + d // 180       # half-year


@lru_cache(maxsize=50_000)
def _cached_predict(
    current_norm: str,
    prior_norm: str,
    days_bucket: int,
) -> bool:
    """Cache-friendly wrapper. The bucketed days key means retried or
    repeated pairs hit cache even if the exact date differs by a day."""
    # Reconstruct approximate dates for feature extraction. The bucket
    # alone is not quite enough; we pick a representative delta inside
    # the bucket. Since is_recent / is_very_recent are step functions,
    # using the bucket midpoint preserves those thresholds.
    if days_bucket == -1:
        cur_date, pri_date = None, None
    else:
        # Use a stable representative day count.
        # Buckets 0..4  -> weeks 0..4    (0, 7, 14, 21, 28 days)
        # Buckets 10..22 -> months 0..12 (0..360 days)
        # Buckets 30+   -> half-years    (>=360 days)
        if days_bucket < 10:
            days = days_bucket * 7
        elif days_bucket < 30:
            days = (days_bucket - 10) * 30
        else:
            days = (days_bucket - 30) * 180
        # Feed the feature extractor a synthetic pair of dates with the
        # right delta. We use a fixed anchor and walk backward.
        cur_date = "2026-01-01"
        # Build a date that many days earlier.
        from datetime import date, timedelta
        pri_date = (date(2026, 1, 1) - timedelta(days=max(days, 0))).isoformat()

    feats = extract_pair_features(current_norm.strip(), cur_date, prior_norm.strip(), pri_date)
    is_rel, _ = SCORER.predict(feats)
    return bool(is_rel)


def predict_pair(
    current_desc: str,
    current_date: Optional[str],
    prior_desc: str,
    prior_date: Optional[str],
) -> bool:
    """Public entry that combines real-date features with cache lookup.

    We compute the accurate, non-bucketed features for the actual dates
    first. Cache is a fast-path for repeated (desc, desc, bucket) triples.
    Using only the bucketed cache would cost a little accuracy at bucket
    boundaries; doing the real compute and then writing to cache is
    slightly redundant but keeps scoring exact.
    """
    feats = extract_pair_features(current_desc, current_date, prior_desc, prior_date)
    is_rel, _ = SCORER.predict(feats)

    # Warm cache for retries.
    cn = normalize(current_desc)
    pn = normalize(prior_desc)
    bucket = _days_bucket(days_between(current_date, prior_date))
    try:
        _cached_predict(cn, pn, bucket)
    except Exception:  # cache failures must never break scoring
        pass

    return bool(is_rel)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "relevant-priors",
        "version": "1.0.0",
        "model_source": SCORER.source,
        "endpoints": ["/predict", "/health"],
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_source": SCORER.source}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()

    total_priors = sum(len(c.prior_studies) for c in req.cases)
    logger.info(
        "request_id=%s cases=%d priors=%d",
        request_id, len(req.cases), total_priors,
    )

    predictions: List[Prediction] = []

    for case in req.cases:
        cur = case.current_study
        for prior in case.prior_studies:
            try:
                is_rel = predict_pair(
                    cur.study_description,
                    cur.study_date,
                    prior.study_description,
                    prior.study_date,
                )
            except Exception as exc:
                # Never drop a prediction — skipped priors count as wrong.
                # Fall back to True (show the prior) on errors.
                logger.exception(
                    "request_id=%s scoring failed for case=%s prior=%s: %s",
                    request_id, case.case_id, prior.study_id, exc,
                )
                is_rel = True

            predictions.append(Prediction(
                case_id=case.case_id,
                study_id=prior.study_id,
                predicted_is_relevant=is_rel,
            ))

    elapsed = time.perf_counter() - start
    rate = total_priors / elapsed if elapsed > 0 else 0.0
    logger.info(
        "request_id=%s done priors=%d elapsed=%.3fs rate=%.0f/s cache=%s",
        request_id, total_priors, elapsed, rate, _cached_predict.cache_info(),
    )

    return PredictResponse(predictions=predictions)


# -----------------------------------------------------------------------------
# Error handling — never 500 in a way that loses predictions for the whole
# request. If the body fails validation we return a clear error; the
# evaluator will count those priors as wrong but at least we see why.
# -----------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )

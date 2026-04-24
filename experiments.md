# Experiments

## Problem framing

For each (current, prior) pair, predict whether the prior should be
shown to the radiologist reading the current study. Accuracy =
`correct / (correct + incorrect)`, with skipped priors counted as
incorrect. Each study exposes only a `study_description` string, a
`study_date`, and an ID — no reports, no images, no reason codes.

The public split has 27,614 labeled priors across 996 cases with a
positive rate of **23.8%**. The 360-second evaluator timeout rules
out per-prior LLM calls.

## Baseline

Logistic regression over 14 hand-designed features of the pair:

| Family | Features |
| --- | --- |
| Description match | `exact_desc_match` |
| Modality | `same_modality`, `modality_both_unknown` |
| Body region | `region_overlap`, `region_jaccard`, `any_region_match` |
| Clinical context | `context_overlap`, `contrast_conflict` |
| Recency | `years_delta`, `is_recent_1y`, `is_very_recent_90d`, `is_stale_5y`, `missing_date` |
| Length | `prior_desc_len_scaled` |

Extraction in `app/features.py`: a dictionary of modality and region
tokens matched against a normalized (uppercase, punctuation-stripped,
whitespace-padded) study description. sklearn `LogisticRegression`
with `class_weight="balanced"`, L2 = 1.0, 80/20 train/val split
(seed 13).

### Results

| Iteration | Val acc @ 0.5 | Val AUC | Best threshold | Best val acc |
| --- | ---: | ---: | ---: | ---: |
| v1 baseline vocab | 0.8742 | 0.8522 | 0.76 | 0.8905 |
| v2 + MAMMO short forms ("MAM", "DIGITAL SCREENER") | 0.9173 | 0.9327 | 0.73 | 0.9279 |
| v3 + ECHO modality, CHEST expanded (ribs, V/Q, pleural) | 0.9116 | 0.9353 | 0.69 | 0.9227 |

Full public-split score with the final model (v3) at threshold 0.73:
**accuracy 0.9271, precision 0.916, recall 0.763, F1 0.833**, AUC 0.935.

### Learned weight pattern

| Feature | Weight |
| --- | ---: |
| `region_jaccard` | +4.79 |
| `exact_desc_match` | +4.67 |
| `prior_desc_len_scaled` | +1.55 |
| `context_overlap` | −0.77 |
| `any_region_match` | −0.69 |
| `same_modality` | +0.63 |
| `is_very_recent_90d` | +0.60 |
| `region_overlap` | +0.45 |
| `modality_both_unknown` | +0.42 |
| `contrast_conflict` | −0.36 |

Two signs worth explaining:

- **`context_overlap` is negative** because when many priors share the
  same niche clinical keyword (e.g. a stack of screening mammograms),
  the labels tend to mark most of them not relevant — one is enough,
  the rest are redundant.
- **`any_region_match` is mildly negative** despite `region_jaccard`
  being the strongest positive. They're highly collinear; Jaccard
  absorbs the main signal and the boolean corrects the overshoot on
  low-overlap matches.

## What worked

1. **Region Jaccard as the headline feature.** A prior on a different
   body part is almost never relevant. Jaccard handles multi-region
   studies (e.g. CT chest/abdomen/pelvis) better than boolean
   intersection.
2. **Exact description match is near-deterministic positive.**
3. **Vocabulary expansion driven by error analysis.** The single
   biggest jump (+3.7 acc, +0.08 AUC) came from one fix: the mistake
   list was dominated by mammography because the region vocabulary
   only matched "MAMMO" and missed the short form "MAM" and
   "DIGITAL SCREENER". Adding those two families killed hundreds of
   false negatives.
4. **Gentle recency decay, not a hard cutoff.** Priors stay useful
   for years — especially oncology and chronic disease imaging.
5. **Keeping the scorer linear.** 14-entry JSON model, ~1.2 s for 27K
   priors on one core, trivial to debug.
6. **Threshold sweep against class imbalance.** At 23.8% positive
   rate, 0.5 under-predicts true. Tuned 0.73 recovered ~1.5 points
   over the default.

## What failed

1. **Per-prior LLM scoring** ruled out on budget: ~30K priors at any
   realistic latency blows 360 s. Even one batched call per case
   doesn't fit comfortably. Not on the hot path.
2. **Patient-name / patient-id features** had no predictive power —
   relevance is about the studies, not the patient.
3. **ECHO modality + chest expansion (v3)** nudged AUC up (+0.003)
   but slightly regressed accuracy at the tuned threshold. Kept
   anyway because AUC generalizes better to the private split.
4. **TF-IDF cosine similarity of descriptions** didn't beat
   exact-match + region features on validation accuracy. The region
   vocabulary captures what IDF weighting would recover.

## Error analysis

Remaining misses cluster in four buckets:

1. **Laterality false positives.** "MAM diagnostic RT" and "MAM
   diagnostic LT" on the same day are often labeled not relevant to
   each other, but the model sees matching modality + region + date
   and scores them near 1.0.
2. **Cross-modality relevance.** A recent CT chest is sometimes
   labeled relevant for a transesophageal echo; a PET/CT skull-to-
   thigh is relevant for a subsequent CT chest. Modality mismatch
   penalizes these.
3. **Vocabulary gaps** beyond the two patched: "V/Q scan", "RIBS",
   "STANDARD SCREENING COMBO", "R2 Mammography Digitized Film".
4. **Very old mammography priors** (2005–2010) labeled relevant for
   2021 diagnostic studies. Global recency decay is too aggressive
   for breast imaging specifically.

## Next steps

Ordered by accuracy-per-effort.

1. **Laterality mismatch feature.** Cheapest win — one regex for
   `(LT|RT|LEFT|RIGHT|BILAT)` and a boolean feature.
2. **Exhaustive vocabulary pass.** Scan unique descriptions, flag
   unmatched tokens, add them. Expected +1 to +2 points given how
   much v2 gained from one fix.
3. **Cross-modality affinity feature.** Hand-curated or empirical
   `P(relevant | cur_mod, pri_mod)` from the public split, as a
   single numeric feature.
4. **Modality-specific recency decay.** Separate `years_delta`
   features per modality family so the model can learn that breast
   priors stay useful longer.
5. **Gradient boosting for non-linearities.** LightGBM over the same
   features if linear plateaus; preliminary runs suggest similar
   accuracy at higher dependency weight, so not shipped.
6. **Description embeddings** (e.g. `all-MiniLM-L6-v2`, ~22 MB):
   current-vs-prior cosine similarity as one feature. Precompute
   per-request in a single batched call to stay fast.
7. **LLM in the cold path only.** Reserve LLM calls for pairs where
   the linear scorer is uncertain (probability in [0.35, 0.65]).
   Batch uncertain priors per case into one call, cache by
   normalized (current_desc, prior_desc). Fits inside 360 s.

## Reproducibility

```bash
pip install -r requirements-train.txt
python train.py public_eval.json --threshold-search
python eval_local.py public_eval.json
```

`app/model.json` is regenerated deterministically (fixed RNG seed).

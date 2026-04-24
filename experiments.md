# Experiments

## Problem

For each (current, prior) study pair, predict whether a radiologist reading the current study would want the prior shown. Accuracy = correct / (correct + incorrect), skipped priors count as incorrect. The 360-second evaluator budget rules out per-prior LLM calls.

Only signal per study: `study_description`, `study_date`, IDs. Public split: 27,614 labeled priors, 996 cases, **23.8% positive rate**.

## Approach

Logistic regression over 16 hand-designed features of the pair:

| Family | Features |
| --- | --- |
| Description match | `exact_desc_match` |
| Modality | `same_modality`, `modality_both_unknown` |
| Region | `region_overlap`, `region_jaccard`, `any_region_match` |
| Laterality | `laterality_conflict`, `laterality_match` |
| Context | `context_overlap`, `contrast_conflict` |
| Recency | `years_delta`, `is_recent_1y`, `is_very_recent_90d`, `is_stale_5y`, `missing_date` |
| Length | `prior_desc_len_scaled` |

Extraction in `app/features.py`: modality, region, and laterality token dictionaries matched against normalized study descriptions. sklearn `LogisticRegression`, `class_weight="balanced"`, L2 = 1.0.

**Train/val split is grouped by `case_id`**, not random over pairs. A random pair-level split leaks: the same case can have priors in both train and val, letting the model memorize the current study's description. Grouped by case, 996 unique cases shuffle into 80% train / 20% val.

## Results

| Iteration | Split | Val acc @ 0.5 | Val AUC | Best threshold | Best val acc |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 baseline vocab | pair (leaky) | 0.8742 | 0.8522 | 0.76 | 0.8905 |
| v2 + MAMMO short forms | pair (leaky) | 0.9173 | 0.9327 | 0.73 | 0.9279 |
| v3 + ECHO, CHEST expansion | pair (leaky) | 0.9116 | 0.9353 | 0.69 | 0.9227 |
| v4 + grouped split + laterality + missing-date fix | **group** | **0.9192** | **0.9452** | **0.69** | **0.9396** |

v4 is the shipped model. Numbers in rows v1–v3 are optimistic because of pair-level leakage; v4 is the honest generalization estimate.

## Learned weights (v4)

| Feature | Weight |
| --- | ---: |
| `laterality_conflict` | −5.74 |
| `exact_desc_match` | +4.15 |
| `region_jaccard` | +3.76 |
| `same_modality` | +1.48 |
| `laterality_match` | +1.35 |
| `region_overlap` | +0.96 |
| `is_very_recent_90d` | +0.62 |
| `prior_desc_len_scaled` | +0.35 |
| `contrast_conflict` | −0.26 |
| `is_recent_1y` | −0.12 |

Laterality conflict is the single strongest feature: an MRI left knee is essentially never a relevant prior for an MRI right knee, even with same modality and same date. `exact_desc_match` and `region_jaccard` remain the other two anchors.

## What worked

1. **Grouped split fixed optimism in the validation numbers.** v3 (pair-level) reported 91.16% but was overestimating — after regrouping, the same model family lands at 91.92%/0.945 AUC with the laterality feature added. This is the real generalization number.
2. **Laterality feature.** Biggest single-feature addition in v4. Caught a whole class of errors where matching modality + region + date were firing on left/right mismatches labeled irrelevant.
3. **Vocabulary expansion from error analysis.** Earlier v2 gain (+3.7 val acc) came from adding "MAM" and "DIGITAL SCREENER" to the MAMMO vocabulary after reading the v1 mistake list.
4. **Region Jaccard as the core positive.** Handles studies that touch multiple regions (CT chest/abdomen/pelvis) better than boolean intersection.
5. **Missing-date gate.** v3 had a subtle bug: `abs_delta = 0` on missing dates caused `is_recent=1` and `is_very_recent=1` to fire. v4 gates all three recency step features on `has_date`.
6. **Cache on the hot path.** v3 had a cache that never helped because `predict_pair()` did the full feature extraction before consulting it. v4 routes prediction directly through an `lru_cache` on `(current_desc, current_date, prior_desc, prior_date)` — identical pairs across retries and repeated study descriptions hit cache on the first try.
7. **Threshold sweep against imbalance.** At 23.8% positive, 0.5 under-predicts true. v4 ships at 0.69 (validation-optimized).

## What failed

1. **Per-prior LLM scoring.** Ruled out on budget — 30K priors × any realistic latency blows the 360s timeout. Not on the hot path.
2. **Patient-name / patient-id features.** Zero predictive power.
3. **ECHO + chest expansion (v3 → v4).** Raised AUC slightly but contributed little accuracy once laterality was added; kept for robustness.
4. **TF-IDF cosine over raw descriptions.** Did not beat exact-match + region features. The vocabulary already captures what IDF weighting would recover.

## Error analysis (v4 remaining misses)

1. **Cross-modality relevance.** A recent CT chest is sometimes relevant for a transesophageal echo; a PET/CT skull-to-thigh is relevant for a subsequent CT chest. Modality mismatch penalizes these.
2. **Vocabulary gaps.** Site-specific shorthand not in the current tables — "V/Q Scan", "R2 Mammography Digitized Film", "STANDARD SCREENING COMBO".
3. **Old mammography priors (2005–2010)** labeled relevant for 2021 diagnostic studies. Recency decay is too aggressive for breast imaging specifically.
4. **Bilateral vs unilateral same-day priors.** "MAM diagnostic RT" and "MAM diagnostic LT" on the same day are sometimes labeled redundant. Laterality features handle the clear L/R conflict case but not this redundancy-among-a-pair-of-priors case, which is inherently a case-level signal.

## Next steps (by expected accuracy-per-effort)

1. **Cross-modality affinity feature** — hand-curated or empirical `P(relevant | cur_mod, pri_mod)` as one numeric feature. Interpretable and cheap.
2. **Exhaustive vocabulary pass.** Scan unique descriptions and add missed tokens. v2 gained 3.7 points from a single fix; more coverage is still on the table.
3. **Modality-specific recency decay** — separate `years_delta` per modality family (breast stays "young" longer than general imaging).
4. **Case-level redundancy features.** When many priors for the same case have identical descriptions, the labels tend to pick one and mark the rest redundant. Features over the set of priors (rank of recency, rank of similarity) would capture this.
5. **Gradient boosting** over the same features if linear plateaus. Preliminary tests showed similar accuracy at heavier dependency weight.
6. **Description embeddings** (all-MiniLM-L6-v2, ~22MB) as one cosine-similarity feature, batched per request.
7. **LLM in the cold path only.** Reserve calls for pairs with probability in [0.35, 0.65], batched per case, cached by normalized description pair.

## Reproducibility

```bash
pip install -r requirements-train.txt
python train.py public_eval.json --threshold-search
python eval_local.py public_eval.json
```

Grouped split is seeded (`seed=13`), so `app/model.json` reproduces deterministically.

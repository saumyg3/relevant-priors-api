# Relevant Priors Endpoint

Submission for the New Grad Residency `relevant-priors-v1` challenge.

Given one current radiology examination and a list of prior examinations
for the same patient, the endpoint returns a relevance prediction
(`true`/`false`) for every prior. Skipping a prior counts as incorrect,
so the service always returns exactly one prediction per input prior.

## Approach in one paragraph

Each (current, prior) pair is reduced to a small vector of
interpretable features — modality match, body-region overlap,
contrast-protocol conflict, years since the prior, exact description
match, and a few others — and fed to a logistic regression trained on
the 27,614 labeled pairs in the public split. No LLM is on the hot
path, so there is no timeout risk; the full public split scores in a
few seconds on a single CPU core. See `experiments.md` for ablations,
error analysis, and next steps.

## Repo layout

```
app/
  __init__.py
  features.py      # feature extraction
  scoring.py       # loads model.json (or zero-shot fallback)
  main.py          # FastAPI app, POST /predict
  model.json       # written by train.py (gitignored if you prefer)
train.py           # fit logistic regression on public eval JSON
eval_local.py      # score public JSON without HTTP
smoke_test.py      # hit a running endpoint with the brief's sample
requirements.txt
requirements-train.txt
Dockerfile
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Smoke test in another shell:

```bash
python smoke_test.py http://localhost:8080/predict
```

## Train on the public eval split

```bash
pip install -r requirements-train.txt
python train.py ~/Downloads/public_eval.json --threshold-search
# -> writes app/model.json; uvicorn picks it up on next start
```

If the script complains that no labeled pairs were found, inspect the
JSON with `jq 'keys' ...` and pass `--label-key <field>`.

Check accuracy without HTTP:

```bash
python eval_local.py ~/Downloads/public_eval.json
```

## Deploy

### Render (simplest)

1. Push the repo to GitHub.
2. New Web Service -> pick the repo -> Docker runtime (auto-detected).
3. Environment: nothing required. The Dockerfile binds `$PORT`.
4. Region: choose the closest US region to the evaluator.

### Fly.io

```bash
fly launch --dockerfile Dockerfile --no-deploy
fly deploy
```

Use a `min_machines_running = 1` in `fly.toml` so the evaluator doesn't
catch a cold start inside its 360-second budget.

### Railway

"Deploy from GitHub", select the repo, it picks up the Dockerfile.

After deploy, paste the resulting URL (`https://yourapp.example.com/predict`)
into the portal's API field and run the quick API check.

## Contract

`POST /predict` with the request body documented in the brief. The
service accepts extra fields (forward-compatible) and returns:

```json
{"predictions": [
  {"case_id": "...", "study_id": "...", "predicted_is_relevant": true},
  ...
]}
```

Exactly one prediction per input prior, same `case_id` and `study_id`.

Also exposed:

- `GET /` — basic service metadata
- `GET /health` — liveness probe

## Operational notes

- Logs every request with `request_id`, case count, prior count, and
  elapsed time, per the brief's hint.
- Scoring is pure CPU. For the public split's 27,614 priors the full
  response is well under a second on a single worker.
- If `model.json` is missing, the service falls back to a hand-tuned
  zero-shot scorer so it still returns sensible predictions (useful
  for the portal's browser quick check before training finishes).
- Errors during scoring of a single pair are caught and fall back to
  `predicted_is_relevant = true` for that pair — predictions are never
  dropped, because skipped priors count as incorrect.

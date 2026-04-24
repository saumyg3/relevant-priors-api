"""Send the sample payload from the challenge brief to a running endpoint."""

import argparse
import json
import sys
import urllib.request

SAMPLE = {
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "2026-04-16T12:00:00.000Z",
    "cases": [
        {
            "case_id": "1001016",
            "patient_id": "606707",
            "patient_name": "Andrews, Micheal",
            "current_study": {
                "study_id": "3100042",
                "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                "study_date": "2026-03-08",
            },
            "prior_studies": [
                {
                    "study_id": "2453245",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2020-03-08",
                },
                {
                    "study_id": "992654",
                    "study_description": "CT HEAD WITHOUT CNTRST",
                    "study_date": "2021-03-08",
                },
                {
                    "study_id": "777777",
                    "study_description": "XR FOOT 3V LEFT",
                    "study_date": "2019-06-15",
                },
            ],
        }
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="endpoint URL, e.g. http://localhost:8080/predict")
    args = ap.parse_args()

    body = json.dumps(SAMPLE).encode("utf-8")
    req = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    print(json.dumps(out, indent=2))

    preds = out.get("predictions", [])
    expected_ids = {p["study_id"] for p in SAMPLE["cases"][0]["prior_studies"]}
    returned_ids = {p["study_id"] for p in preds}
    missing = expected_ids - returned_ids
    if missing:
        print(f"\nWARNING: missing predictions for prior study_ids: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"\nOK: returned {len(preds)} predictions for {len(expected_ids)} priors.")


if __name__ == "__main__":
    main()

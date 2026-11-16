# Temporalis MAM Model Provenance and Release Gate

**Status:** Enforced research-to-production artifact process  
**Scope:** `BruxismMAM_Temporalis` only. The four classes remain device-inferred wellness phenotypes; this process does not establish diagnostic validity.

## Why the gate exists

A Core ML package alone cannot show which signals, people, labels, algorithm settings, or validation split produced it. A model can also be a placeholder used only to test the integration path. Neither condition is suitable for a release inference route.

Every new MAM artifact therefore needs a JSON provenance record that links:

1. the exact Keras input (`keras_sha256`);
2. the exact `algorithm_spec.yaml` fingerprint and input shape;
3. the source data fingerprint and cohort identity process;
4. a participant-level holdout strategy and its metrics; and
5. explicit release approval after review.

## Training behavior

`scripts/generate_mam_model.py` now requires a `user_id` column and creates a deterministic user-level holdout by default. It rejects a single-user corpus because a random split of overlapping windows would create invalid validation evidence.

```bash
.venv/bin/python scripts/generate_mam_model.py \
  --gold data/processed/temporalis_multi_user.csv \
  --user-column user_id \
  --out artifacts/mam_temporalis.h5
```

This writes `artifacts/mam_temporalis.provenance.json`. The generated record is still **not release eligible**. It documents the run and its holdout metrics; it does not claim the required independent review, cohort sufficiency, or product approval.

The only exception is explicitly research-only experimentation:

```bash
.venv/bin/python scripts/generate_mam_model.py \
  --gold data/processed/single_subject_pilot.csv \
  --allow-window-random-research-only \
  --out artifacts/pilot_only.h5
```

That record is labelled `research_only`; it cannot be converted into the default bundled Core resource.

## Core ML conversion

The converter has no stub-model fallback. It always requires a Keras artifact and matching provenance:

```bash
.venv/bin/python scripts/convert_temporalis_mam.py \
  --keras artifacts/mam_temporalis.h5 \
  --provenance artifacts/mam_temporalis.provenance.json \
  --output artifacts/BruxismMAM_Temporalis.mlpackage
```

A write to the default `OralableCore` resource path is blocked unless the reviewed provenance explicitly sets both:

```json
{
  "lifecycle": {
    "release_eligible": true,
    "stub_model": false
  },
  "validation": {
    "strategy": "user_holdout",
    "metrics": { "validation_accuracy": 0.0 }
  }
}
```

The release approver must update that record only after validating the source evidence; do not set these values to bypass the gate.

## Bundled artifact policy

`OralableCore/Sources/OralableCore/Resources/BruxismMAM_Temporalis.provenance.json` is the sidecar for the packaged model. `CoreMLTemporalisClassifier` fails closed unless the sidecar says the model is release-eligible and affirmatively non-stub.

The currently checked-in model is marked `unverified_legacy_research_artifact`. It has an artifact fingerprint but lacks a linked Keras source, immutable training run, user-level metrics, and approval. It is intentionally not eligible for production inference.

Run the offline integrity check from the research hub:

```bash
.venv/bin/python scripts/verify_model_provenance.py
.venv/bin/python scripts/verify_model_provenance.py --release  # expected to fail until approval evidence exists
```

## Required evidence before enabling a release model

- Frozen Keras source and SHA-256.
- Versioned algorithm-spec fingerprint.
- Cohort manifest with stable, privacy-preserving participant IDs and placement/side metadata.
- User-level holdout results, with class-level metrics and error review.
- An independent test panel never used in model selection.
- Human release approval that records model version, application build, and validation packet location.

The Tier 1 / Tier 2 cohort targets and labeling constraints remain in [CORE_ML_TRAINING_COHORT.md](./CORE_ML_TRAINING_COHORT.md).

# Data Boundary

The public code separates local dataset inspection from any external LLM request. Raw SEM images,
absolute filesystem paths, internal identifiers, credentials, and proprietary results are not LLM
inputs and must not appear in logs or committed artifacts.

## Public distribution

This repository contains synthetic SEM-like metadata for reproducible offline execution. The
sample profile records `real_company_evaluation=false`. The repository does not distribute
proprietary images, private paths, internal configurations, internal identifiers, or internal
evaluation results.

This codebase has been run against proprietary SEM data in an internal environment. That statement
does not imply that any private input or result is present in the public tree.

## Dataset profile boundary

`VRO_DATASET_ROOT` is blank in `.env.example` and is reserved for local dataset-profile
integration. The committed sample does not require it. A private-data integration must enforce the
following flow:

```text
local dataset root -> deterministic local profiling -> sanitized DatasetProfile -> workflow/LLM
```

An external LLM may receive only allowlisted profile fields such as task type, modality, channel
count, dimensions, dtype, anonymized label vocabulary, class counts, split unit, imbalance
summary, and a non-reversible content fingerprint. It must not receive image bytes, absolute paths,
company or customer names, equipment identifiers, wafer or lot identifiers, credentials, or
internal endpoints.

A dataset profile is versioned evidence, not a permanent constant. Image, label, split, or
preprocessing changes require a new fingerprint and regenerated profile.

## Other boundaries

- DashScope credentials are loaded from the environment at the CLI boundary and are redacted from
  structured failures and call records.
- Repository snapshots downloaded at runtime remain local and retain their upstream licensing.
- Workflow state stores relative references; generated data is confined to the Git-ignored `var/`
  tree.
- Public sample approvals are explicitly marked as scripted. They do not impersonate a human
  decision or private-data evaluation.

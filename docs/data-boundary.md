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

`VRO_DATASET_ROOT` is blank in `.env.example` and is used only by the explicit local profiler.
`VRO_DATASET_PROFILE` selects its generated JSON for conversational Code advice. The committed
sample does not require either. The local integration enforces the
following flow:

```text
local dataset root -> deterministic local profiling -> sanitized DatasetProfile -> workflow/LLM
```

The Code LLM receives only an explicit typed projection: task, modality, channels, dimensions,
class count, anonymous sorted count values, aggregate split counts, split-strategy enum,
proportions, numeric resize and known normalization enum. Dataset IDs, label names/IDs, group keys,
holdout values, source authorization/configuration and original content/location references are
excluded even if they occur in a valid profile. It must not receive image bytes, absolute paths,
company or customer names, equipment identifiers, wafer or lot identifiers, credentials, or
internal endpoints.

A dataset profile is versioned evidence, not a permanent constant. Image, label, split, or
preprocessing changes require a new fingerprint and regenerated profile.

Code analysis never opens images or profiles a root. Missing/invalid explicit JSON fails without
abstract fallback or parser/path details. Context stores only the safe facts snapshot, not the
original path; restart without explicit configuration restores frozen facts, not a refreshed
dataset. Explicit configuration always revalidates. A changed projection invalidates old Code
results; changes only in excluded metadata do not change the safe facts fingerprint.

## Local profiling contract

Profiling starts only from `VRO_DATASET_ROOT` or an explicit `--dataset-root` value; there is no
directory discovery or fallback. A root contains exactly these inputs:

```text
dataset.json
samples.csv              # relative_path,source_label,split,group_id
images/**                # PNG or PGM
```

The manifest fixes the dataset ID, version, creation time, source kind, de-identified label map,
channel count, split policy, and preprocessing contract. The local profiler verifies every image
listed by the CSV, rejects extra files, symlinks, unsafe relative paths, unknown labels, malformed
headers, and groups that cross splits. It derives counts and image shape from the files rather
than accepting them as profile facts.

The generated `DatasetProfile` contains opaque location and content references only. It never
contains the supplied root, sample paths, source labels, group identifiers, or image bytes. A
different content hash for an existing dataset ID/version is rejected instead of overwriting the
stored profile. The bundled sample follows this same profiling path for adaptation; local training
and evaluation remain bound to their committed synthetic inputs.

## Other boundaries

- DashScope credentials are loaded from the environment at the CLI boundary and are redacted from
  structured failures and call records.
- Repository snapshots downloaded at runtime remain local and retain their upstream licensing.
- Workflow state stores relative references; generated data is confined to the Git-ignored `var/`
  tree.
- Public sample approvals are explicitly marked as scripted. They do not impersonate a human
  decision or private-data evaluation.

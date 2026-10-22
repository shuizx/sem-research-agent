# Validation results and limits

Recorded on 2026-10-22. These checks describe the tested configuration and scenarios, not a
guarantee for every provider, repository or request. Reproduction commands are in
[Development](development.md); runtime components and trust boundaries are in
[Architecture](architecture.md).

## Offline checks

The recorded Python 3.12 offline suite passed **797 tests**, with one existing Windows skip where
symlink creation was unavailable. The repository-insight subset passed 25 tests. Ruff formatting
and lint passed, and strict mypy passed for 121 source files. Offline tests use injected model
responses and do not require real credentials, private data, network access or a GPU.

## Live provider checks

The tested OpenAI-compatible configuration was the SHU gateway with the exact model ID
`qwen3.8-max`, `temperature=0`, structured output and tool calling. Credentials stayed in local
configuration and were not committed.

- Supervisor greeting and out-of-scope routing completed without worker delegation.
- A direct arXiv request completed a typed paper assessment without unrequested Code delegation.
- Research executed real searches. The bounded latest-paper case returned `NO_CANDIDATES`;
  this verifies the empty-result path, not successful candidate discovery in every time window.
- Repository routing reached its human approval gate before code analysis.
- Restart restored the completed public working context without repeating provider calls.

## Read-only Code analysis

One check used the previously approved source archive for
[`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar), fixed at commit
`49b7aa97b0c12fe0d4054e670403a16b6b834ddd`. Three public metadata requests verified source
provenance; the archive was reused, not downloaded again.

The real model invoked summary and target-profile inspection, then read six source files:
`main.py`, `utils.py`, `models/__init__.py`, `models/dla_simple.py`, `models/resnet.py` and
`README.md`. Reads totaled 18,195 bytes without truncation.

The first advice submission exceeded the schema limits for `risks` and `items_to_verify`.
The ToolNode returned one sanitized observation containing only field names, error types and
limits. A second submission passed the unchanged schema and actual-read-path checks.
The graph finished after **10 model calls**, including that single correction, with zero SDK
retries. The observed duration was 190.94 seconds; it is not a latency benchmark.

The output contained typed adaptation advice and a hash-only tool trace. All cited code paths
belonged to the actual read set. No field was silently truncated and no budget was increased.

## Scope limits

- The successful Code check covered the planner with approved cached source; the complete
  conversation, approval and fresh-download workflow was not rerun end to end after the repair.
- The target was the abstract SEM profile, not private images or a locally supplied dataset.
- No upstream repository was installed, imported, executed, patched, smoke-tested or trained.
- These checks do not establish model-accuracy improvements, private-data results, performance
  guarantees or production readiness.
- The controlled synthetic training/evaluation sample is a separate workflow, not evidence of
  training the public GitHub repository.

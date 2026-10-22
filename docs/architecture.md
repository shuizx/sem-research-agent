# Architecture

SEM Research Agent is organized as a set of dependency-injected LangGraph workflows. Each graph
keeps orchestration separate from network access, model calls, repository handling, execution,
and persistence.

## Workflow stages

1. **Research** retrieves normalized arXiv records, requests a schema-constrained applicability
   assessment, and pauses for candidate selection.
2. **Repository** validates a GitHub target, resolves an immutable commit, checks licensing, builds
   a static repository profile, and pauses before ingestion.
3. **Repository insight** reads a hash-verified source archive through bounded tools. A LangGraph
   `ToolNode` exposes only manifest, configuration, data-pipeline, and training-entrypoint reads.
4. **Adaptation** turns observed repository evidence and a `DatasetProfile` into a typed plan,
   applies allowlisted edits, runs a bounded smoke check, and pauses before patch acceptance.
5. **Training** freezes an approved experiment specification, pauses before execution, and invokes
   the configured local runner under a command and resource policy.
6. **Evaluation** verifies baseline/candidate comparability, computes deterministic metrics, and
   writes a report without an LLM decision.

The top-level pipeline composes the research-through-evaluation graphs serially. Failure or human
rejection stops downstream execution and produces an explicit terminal record.

## Layers

The default conversational entry is a supervisor `StateGraph` that observes compact typed worker
results and delegates serially to Research or Code (three delegations maximum). Research uses
model → ToolNode → model with search/inspect/submit/finish tools (two searches, five papers).
Code uses its existing four read-only tools; public repository conversation never enters patch,
smoke or training. The controlled synthetic five-stage sample remains separate.
The deterministic boundary rejects duplicate completed worker/target pairs, unrequested Code
expansion from a paper-only request and ordinary explicit negatives such as `no code` or
`不分析代码`. Direct repository-only requests cannot expand into unrelated Research. These are
small scope guards, not a general natural-language authorization parser.

One default-24 request budget counts every live supervisor/worker invocation before execution,
including direct-paper schema corrections and `/continue`. Child graphs reuse inherited
interrupt/resume checkpoints; human approval never comes from a model. Successful small working
context is persisted, not pending gates or transcripts. Local traces contain delegation outcomes,
call counts and separate bounded tool event hashes, not raw model messages.

`VRO_DATASET_PROFILE`/`--dataset-profile` selects an existing generated DatasetProfile. Local
validation projects numeric/enum facts into `SafeTargetProfile`; the Code tool and model observe
that actual snapshot. Context embeds only this safe snapshot; content-addressed copies under
`target-profiles/` support provenance, not automatic dataset refresh. An explicit file always
revalidates and fails closed. Changing safe target facts or selecting a new paper clears old Code
context. Without explicit configuration, restored safe facts remain a frozen snapshot; otherwise
Code marks its abstract SEM default explicitly.

- `domain/` contains strict versioned records, value objects, enums, and stable failures.
- `ports/` defines narrow provider-neutral interfaces for side effects.
- `adapters/` implements DashScope, arXiv, GitHub, source-archive, and local-execution boundaries.
- `application/services/` contains deterministic validation and transformation logic.
- `application/workflows/` assembles LangGraph state graphs, conditional edges, and interrupts.
- `cli/` is the configuration and user-interaction boundary.
- `fixtures/` supplies offline sample inputs; `tests/` verifies contracts and failure paths.

The installed distribution is named `sem-research-agent`. The existing
`vision_research_ops` import namespace is retained for compatibility.

## Trust and execution boundaries

Paper text, web content, repository files, README instructions, and model outputs are untrusted
input. They cannot add tools or change system policy. Graph nodes do not execute arbitrary shell
commands; execution occurs through injected, policy-checking interfaces. Repository source is
pinned and hash verified before analysis.

Human gates bind a typed decision to the exact evidence presented at the interrupt. Scripted
decisions are available only in explicit sample mode and carry distinct provenance.

## State and artifacts

Checkpoint state is JSON-safe and contains only small records plus canonical relative references.
Patches, logs, predictions, metrics, reports, downloaded source snapshots, and other generated
artifacts live below `var/`, which is excluded from version control. Stores use stable identities
and reject conflicting rewrites.

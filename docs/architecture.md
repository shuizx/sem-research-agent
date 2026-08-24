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

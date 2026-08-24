# SEM Research Agent

SEM Research Agent is a local, human-gated workflow for evaluating recent computer-vision
research against SEM defect-classification requirements. It combines structured LLM decisions
with deterministic repository inspection, controlled code adaptation, bounded training, and
reproducible baseline-versus-candidate evaluation.

The distribution name and product name are `sem-research-agent` and **SEM Research Agent**. The
Python import namespace remains `vision_research_ops` to preserve stable module paths and stored
workflow references.

## Capabilities

- Retrieves arXiv candidates and asks a schema-constrained LLM to assess applicability.
- Resolves a selected public GitHub repository to an immutable commit, checks its license, and
  builds a bounded static profile.
- Uses a LangGraph `ToolNode` with four read-only code-inspection tools before producing a typed
  adaptation plan.
- Applies only supported PyTorch image-classification changes, runs a controlled smoke check, and
  records the accepted patch as a local artifact.
- Requires human decisions before candidate selection, repository ingestion, patch acceptance,
  and training submission.
- Runs short local training under an explicit command and resource policy.
- Compares baseline and candidate outputs with deterministic metrics and a generated report; the
  LLM does not determine the reported result.
- Offers a bounded conversational interface with structured intent routing and small local
  working context.

All LLM responses that cross an execution boundary are validated with Pydantic models. Invalid
schemas and provider failures stop explicitly; they are not converted into silent fallback
results.

## Architecture

```text
arXiv retrieval -> structured applicability -> human gate
                                             |
                                             v
fixed repository snapshot -> static profile -> human gate
                                             |
                                             v
read-only ToolNode inspection -> typed adaptation plan -> bounded patch/smoke -> human gate
                                                                            |
                                                                            v
                                                        approved local training -> evaluation
```

LangGraph owns state transitions, conditional edges, interrupts, and resume behavior. Narrow
ports isolate network, GitHub, LLM, filesystem, patch, smoke-test, and training effects. Workflow
state contains small JSON-safe values and relative artifact references; generated files are
written below the Git-ignored `var/` directory.

See [docs/architecture.md](docs/architecture.md) for component details and
[docs/data-boundary.md](docs/data-boundary.md) for the data-handling contract.

## Quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --all-groups --locked
uv run python -m vision_research_ops.cli.pipeline sample `
  --mode fixture `
  --adaptation-planner scripted `
  --workspace var/sample-run `
  --workflow-id sem-sample `
  --auto-approve-sample
```

This command is offline and uses the committed synthetic SEM profile, a small fixture repository,
scripted structured model responses, and deterministic CPU training. Remove
`--auto-approve-sample` to enter each decision interactively. Scripted sample approvals still
resume the same LangGraph interrupts used by interactive decisions.

### DashScope configuration

Copy `.env.example` to `.env`, replace the API-key placeholder, and keep `.env` untracked:

```powershell
Copy-Item .env.example .env
```

```dotenv
DASHSCOPE_API_KEY=replace-with-your-dashscope-api-key
VRO_LLM_MODEL=qwen-plus
VRO_DATASET_ROOT=
```

Start the conversational interface:

```powershell
.\scripts\run-agent.ps1
```

The live path uses `langchain_openai.ChatOpenAI` with DashScope's OpenAI-compatible endpoint,
`temperature=0`, and structured output. The central settings layer reads configuration at the CLI
boundary. Missing credentials or invalid structured responses fail explicitly.

To inspect the interface without contacting external services:

```powershell
.\scripts\run-agent.ps1 -SampleMode
```

`VRO_DATASET_ROOT` is reserved for local dataset-profile integration and is blank by default. The
bundled sample does not require a private dataset path.

## Sample mode and evidence

Sample inputs live under `fixtures/` and are intentionally small. They exercise the same graphs,
typed state, gates, artifact stores, patch restrictions, smoke checks, training policy, and
evaluation engine as the corresponding local workflow. Generated evidence is written under the
chosen workspace and is not committed.

The sample dataset metadata explicitly records that it is synthetic and that it is not a real
company evaluation. No bundled metric should be interpreted as a proprietary or production
result.

## Data privacy

This codebase has been run against proprietary SEM data in an internal environment. The public
repository excludes proprietary data, paths, identifiers, configurations, and results. Its
bundled reproducible example uses synthetic data.

Raw images and local absolute paths must remain on the local side of the dataset-profile boundary.
Only a sanitized, versioned `DatasetProfile` may be supplied to an external LLM. Secrets are read
from environment configuration, redacted at port boundaries, and never committed. More details
are in [docs/data-boundary.md](docs/data-boundary.md).

## Limitations

- The supported adaptation path is intentionally narrow: Python, PyTorch,
  `image_classification`, and known repository layouts.
- The system is local and single-user; it is not a hosted service, multi-tenant platform, or job
  scheduler.
- Public-repository inspection is bounded and does not establish that upstream code is safe to
  execute. Training requires an approved, policy-compatible snapshot.
- Applicability and adaptation outputs are proposals for human review. They are not scientific
  conclusions or deployment approvals.
- The bundled sample demonstrates behavior and reproducibility, not operational scale or model
  quality on private data.

## Tests

The default suite is offline and uses no real API key, public network, GPU, or proprietary data.

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Development conventions and test boundaries are documented in
[docs/development.md](docs/development.md).

## License

Released under the [MIT License](LICENSE). See [NOTICE](NOTICE) for the public-distribution and
sample-asset notices.

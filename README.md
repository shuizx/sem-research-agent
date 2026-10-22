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

### LLM configuration (DashScope or an explicit OpenAI-compatible gateway)

If `.env` does not already exist, copy `.env.example` to `.env`. Edit it locally and keep it
untracked; never replace an existing secret file or put a real key in `.env.example`:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```dotenv
DASHSCOPE_API_KEY=replace-with-your-dashscope-api-key
VRO_LLM_MODEL=qwen-plus
VRO_DATASET_ROOT=
```

For the user-selected SHU gateway, use these settings in your local `.env` instead:

```dotenv
VRO_LLM_BASE_URL=https://claw.shu.edu.cn/api/v1
VRO_LLM_MODEL=qwen3.8-max
VRO_LLM_API_KEY=replace-with-your-shu-api-key
```

Do not append `/chat/completions` to the base URL. The model ID is passed exactly as configured.
The generic key takes precedence; a custom endpoint requires its own `VRO_LLM_API_KEY` and never
falls back to `DASHSCOPE_API_KEY`. Process variables override `.env` for the same name. Recheck
old process variables when switching gateways. HTTPS is required; credentials, query parameters
and fragments are rejected in the base URL. `.gitignore` prevents ordinary accidental commits,
not local file access or encryption. Do not paste keys into chats, screenshots or shell history.

Configuration support is not a live capability guarantee: this gateway/model must support
OpenAI-compatible Chat Completions, function/tool calls and schema-bound output. A bounded SHU
check using the exact model ID above verified authentication, Supervisor routing, paper analysis,
and research retrieval. A separate Code check against previously approved, fixed-commit cached
source read six files and produced validated adaptation advice after one sanitized list-length
correction, within the unchanged ten-call budget. This verifies the Code planner, not a new full
approval/download workflow. See [P6-T15 evidence](docs/handoffs/P6-T15.md) and
[P6-T16 correction evidence](docs/handoffs/P6-T16.md). Supervisor schema failures and the Code
planner's declared final-list length failures each permit at most one bounded correction.
Provider errors remain explicit; there is no automatic retry against a different gateway/model
and no unstructured-output fallback.
Legacy code/CLI names such as `dashscope` and `DASHSCOPE_TOOL_CALLING` identify the existing adapter
path, not a vendor claim; recorded provider IDs are `openai-compatible` for custom endpoints.

Start the conversational interface:

```powershell
.\scripts\run-agent.cmd
```

The live path uses `langchain_openai.ChatOpenAI` with the selected OpenAI-compatible endpoint,
`temperature=0`, and structured output. The central settings layer reads configuration at the CLI
boundary. Missing credentials or invalid structured responses fail explicitly.

To inspect the interface without contacting external services:

```powershell
.\scripts\run-agent.cmd -SampleMode
```

`VRO_DATASET_ROOT` is used only by the explicit local profiler. The conversation never opens images.
The bundled sample does not require a private dataset path. The `.cmd` wrapper works with normal
PowerShell execution-policy defaults; it does not change the machine policy.

The default conversation uses one LangGraph supervisor and two serial workers: Research searches,
reads and assesses papers through tools; Code reads a fixed public GitHub snapshot and proposes
adaptation advice. Public GitHub conversation ends at read-only advice: it does not patch, smoke
test or train upstream code. The five-stage controlled fixture sample above is a separate command.
Each main request has at most three delegations and 24 live model calls shared by supervisor,
Research, Code and schema correction. Help/status/clear/exit are local deterministic commands.

### Local dataset profile

Inspect an explicitly supplied local image directory without sending image data or local paths to
the workflow:

```powershell
uv run python -m vision_research_ops.cli.dataset_profile `
  --dataset-root <approved-dataset-root>
```

The root must contain `dataset.json`, `samples.csv`, and `images/`. The profiler accepts only
standard PNG or PGM images, rejects symlinks and non-canonical paths, and writes a sanitized
profile below `var/dataset-profiles/`. The command emits only the dataset ID, version, content
hash, sample count, and relative profile reference.

Select that generated JSON explicitly for conversational Code analysis:

```powershell
uv run python -m vision_research_ops.cli.agent --dataset-profile var/dataset-profiles/<id>/<version>/profile.json
```

Alternatively set `VRO_DATASET_PROFILE` in the ignored `.env`. Invalid or missing explicit files
fail before model calls, without fallback. Without configuration, the tool clearly reports an
`abstract_default`, not observed dataset facts. Code receives only numeric/enum safe facts, never
dataset IDs, labels, groups, image paths, source configuration or image bytes.

Successful paper/repository context and the safe target snapshot survive restart in bounded local
JSON, not a chat transcript. With no explicit file on restart, this is a frozen statistics snapshot,
not a refreshed dataset view. Explicit files are always revalidated and replace that snapshot;
changed safe facts clear stale Code context. New paper selection also clears old Code context.
`/continue` uses the current paper's unique canonical GitHub URL through the same supervisor and
human Gate; absent/ambiguous links ask for clarification. `分析当前仓库` reuses the current repository.
`/clear` removes working context but not artifacts; an explicit startup configuration remains active
and is revalidated on the next request. Compact delegation/count traces are local under `var/`.

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

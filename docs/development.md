# Development

## Environment

Use Python 3.12 and the lockfile-managed environment:

```powershell
uv sync --all-groups --locked
```

Do not commit `.env`, generated workspaces, downloaded repositories, model outputs, or local
dataset paths. Runtime artifacts belong below `var/`.

## Quality checks

Run the offline quality gate before committing:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

The default test suite must not require an external service, API key, network connection, GPU, or
private dataset. Live-provider behavior is tested through injected transports and strict schemas.

## Design conventions

- Keep public functions, graph nodes, protocols, and exceptions typed.
- Validate data at LLM, network, Git, filesystem, patch, and execution boundaries.
- Use timezone-aware UTC timestamps and stable relative artifact references.
- Keep LangGraph responsible for orchestration, interrupts, and recovery; keep metric calculation
  and execution deterministic.
- Add tools only through narrow injected interfaces. Repository content and model responses never
  define executable policy.
- Preserve explicit human decisions for candidate selection, repository ingestion, patch
  acceptance, and training submission.

The distribution is `sem-research-agent`; imports use `vision_research_ops` for compatibility.

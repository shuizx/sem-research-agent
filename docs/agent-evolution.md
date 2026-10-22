# Lightweight agent evolution

Decision date: 2026-10-22. Baseline: public `main` at `7c4024c`.

The conversational application evolves into one supervisor and two fixed, serial workers:
Research and Code. LangGraph remains the runtime. Research owns paper evidence; Code owns
repository reading and adaptation advice. The controlled training/evaluation sample remains a
separate explicit workflow. This document is the fixed scope and acceptance record for the next
three tasks. Only one implementation task may be active at a time.

| Task | Status | Dependency |
| --- | --- | --- |
| P6-T11 Supervisor and delegation | DONE | Reviewed `1e113ad`; A1–A8 PASS |
| P6-T12 Research tool loop | DONE | Reviewed `754f05d`; B1–B7 PASS |
| P6-T13 Profile and context integration | DONE | Reviewed `d3dd8b3`; C1–C7 PASS |
| P6-T14 Explicit provider configuration | DONE | `2684829`; D1–D5 offline PASS |
| P6-T15 Bounded conversation live repair | DONE | `d24a820`; two review findings closed; separate Code advice failure recorded |
| P6-T16 Bounded Code advice correction | DONE | Reviewed `138c13f`; K1–K6 PASS; actual cached-source live correction succeeded |

The original three tasks completed independent fixed-matrix reviews and local integration.
P6-T15 closes the URL and Supervisor defects after one review and one bounded repair, with
783 offline tests passing and one existing Windows symlink skip. Real paper analysis, research
retrieval and repository approval routing passed; Code advice submission exposed a separate
oversized-list validation failure. That failed evidence remains in the P6-T15 handoff. P6-T16
repairs this boundary without relaxing schemas or budgets: independent K1–K6 review accepted,
797 offline tests passed with one existing skip, and one real cached-source Code planner check
completed after the permitted correction. This was not a new full approval/download workflow.
No further implementation task is READY in this plan. Remote publication is separate.

## Shared decisions

- Use the existing StateGraph, ToolNode, Pydantic and DashScope stack. No new agent framework,
  database, service, queue, vector store or runtime worker process is required.
- The supervisor selects one action at a time from typed delegation tools and observes the
  result before its next decision. At most three delegations are allowed per user request.
  Help, status, clear and exit remain deterministic local commands.
- Worker tasks have independent contexts. Pass a goal, validated public targets and necessary
  context references; return a compact typed status, evidence, next-step information and artifact
  references. Do not forward complete worker transcripts to the supervisor.
- The completed design has a shared request model-call budget (default 24), counting supervisor,
  worker and schema-correction invocations before they execute. Provider errors and exhaustion
  are explicit. Existing source file/byte limits remain in force. No parallel or recursive
  delegation and no unrestricted tool discovery.
- Candidate and repository approvals remain actual LangGraph interrupts. The CLI presents
  approval requests and resumes the pending work; a model cannot supply human approval.
  Ordinary resume must not repeat completed retrieval or source analysis. Persistence across
  application restarts remains limited to successful working-context summaries.
- Child results are evidence, not instructions. Repository links used for handoff come from
  explicit user input or collected source evidence. No invented repository or success result.
- Public GitHub work ends at read-only analysis/advice. Real third-party patching/training,
  private-data execution and cancelled production infrastructure are outside these tasks.
- Keep public APIs in domain/ports, dependencies and CI unchanged unless the coordinator accepts
  a concrete necessary contract request. New task-local types belong in application services.

## P6-T11: Supervisor and delegation

Input: a conversational request, existing successful context, existing research/code workflows.
Output: a real supervisor graph, typed worker results, concise CLI output and a local call trace.

Implement a bounded observation/decision/delegation loop. Wrap and reuse existing paper and code
capabilities instead of duplicating providers. A compound request can select a paper and then
request code advice; a paper-only request must not automatically expand to repository access.
Existing direct commands and single-target requests remain usable. An explicit sample command
continues to invoke the controlled five-stage workflow.

Owned paths: new `application/agents/**`, `application/workflows/supervisor.py`, task-local
`application/services/supervisor*` or `agent_*`, `prompts/supervisor/**`, `adapters/llm/supervisor.py`,
`cli/agent.py`, `cli/conversation.py`, `application/services/conversation_intent.py`,
`application/workflows/research.py`, `application/workflows/repository_insight.py`,
`tests/supervisor/**`, directly affected `tests/conversation/**`, and `docs/handoffs/P6-T11.md`.
Narrow necessary changes to existing LLM adapters and runtime dependency composition are allowed
for typed results, shared budget injection and interrupt propagation; identify them in the handoff.
Coordinator-owned shared files: this document, README, architecture, settings, dependencies,
domain, ports, CI and the original coordination roadmap.

Fixed acceptance matrix:

1. A1: a real StateGraph supervisor observes typed worker results and chooses the next action;
   it is not a renamed one-shot intent classifier. Live and explicit scripted paths are provided.
2. A2: direct paper and repository requests select only the intended capability; a normal compound
   paper-to-code request performs two serial delegations using the returned paper evidence.
3. A3: candidate/repository reject prevents the corresponding downstream effects; genuine
   interrupt/resume continues correctly and ordinary resume does not replay completed work.
4. A4: at most three delegations, known tool names and typed inputs/results are enforced; invalid
   model output, provider failure and budget exhaustion terminate with a clear result.
5. A5: worker context and returned summaries are bounded and exclude raw images, local paths,
   credentials and unnecessary full transcripts; source text cannot grant execution permission.
6. A6: default conversational entry reaches the supervisor, while help/status/clear/exit and
   explicit sample execution continue working. No raw event flood unless debug was selected.
7. A7: new ordinary routing/gate/error regressions and existing conversation/repository-insight
   tests pass, as do formatting, lint, strict typing and the offline suite.
8. A8: no new production infrastructure, dependency/domain/port drift, real third-party execution,
   fabricated evidence or unrelated changes; handoff records actual commands and limitations.

## P6-T12: Research tool loop

Input: a paper goal/time window or explicit arXiv target. Output: evidence-based assessments,
human-selected paper context and provenance-backed code links.

The worker uses a real model/ToolNode loop to search, inspect collected paper evidence and submit
a typed result. It can refine its query once after inspecting initial results (two search calls
total), with at most five analyzed candidates. Reuse arXiv normalization, windowing and deduplication.
One loop should produce the typed assessment; avoid nesting redundant LLM assessment calls for
the same evidence. Missing Python/PyTorch metadata and missing code are uncertainty/adaptation
notes, not automatic exclusion; classification/SEM relevance is assessed from available evidence.
Explicit user paper analysis works without source-code links. The final candidate gate is reused.

Owned paths: Research worker and its task-local models/runtime/prompts, `adapters/llm/**`,
`application/services/paper_*`, `application/nodes/research.py`, `application/workflows/research.py`,
`cli/research.py`, relevant supervisor/CLI integration, `tests/research/**`, `tests/supervisor/**`,
directly affected conversation tests and `docs/handoffs/P6-T12.md`.

Fixed acceptance matrix:

1. B1: actual tool observations control whether to refine a search or finish; initial no-result
   followed by one successful refinement is demonstrated offline.
2. B2: at most two searches/five assessed candidates, bounded UTC windows, deduplication and a
   shared request model-call limit hold for normal calls including schema correction.
3. B3: a relevant classification paper without Python/PyTorch/code metadata can be assessed;
   missing evidence is reported, not invented. Non-relevant papers receive a reason.
4. B4: submitted paper IDs, claims/citations and code handoffs refer to retrieved evidence;
   unsupported IDs or schema/provider failures cannot become successful results.
5. B5: candidate approve/edit/reject and resume preserve the selected evidence and do not repeat
   completed provider/model calls; only approved selection reaches a compound code request.
6. B6: direct arXiv analysis, CLI output and paper reports retain summary, rationale, code status
   and provenance. No-results is a valid explicit outcome.
7. B7: task regressions, existing affected tests and ordinary offline quality checks pass without
   new infrastructure or protected-interface drift.

## P6-T13: Profile and context integration

Input: an explicit generated local DatasetProfile or the abstract SEM target; successful current
paper/repository context. Output: Code advice tied to the actual sanitized target and a usable
end-to-end supervisor CLI.

Use a generated profile file/reference, not raw image access during code analysis. Revalidate
profile data locally and send only an explicit safe field projection. If an explicitly selected
profile is missing/invalid, fail clearly; do not silently substitute the abstract profile.
Keep root paths, sample paths, raw labels, identifiers and images local. Reuse the existing small
context JSON and add only the minimal target-profile reference/provenance needed. No chat archive
or per-worker persistent memory. Surface concise task/tool progress and save small local traces.

Owned paths: supervisor/worker integration, repository-insight models/planner/runtime, profile
projection service, conversation-context service and CLI, relevant adapters, directly related
tests, `docs/handoffs/P6-T13.md`. Settings and user docs are coordinator-owned shared edits and
must be agreed before editing; dependencies/domain/ports/CI remain protected.

Fixed acceptance matrix:

1. C1: Code's target tool receives safe facts from the explicitly selected real local profile;
   changing valid channel/label/split facts changes the tool evidence supplied to the model.
2. C2: abstract versus configured target provenance is visible; invalid explicit profile fails
   without fallback or raw-path leakage, and raw image input is not sent to an LLM.
3. C3: successful current paper/repository/profile context survives a CLI restart; clear works;
   stale repository evidence is not silently attributed to a newly selected paper/profile.
4. C4: paper-only, repository-only and approved paper-to-code compound requests work end-to-end;
   normal follow-up requests use the correct current context and ask when it is absent/ambiguous.
5. C5: a shared request budget covers all live model and correction calls; traces record concise
   delegation/tool outcomes without secrets, raw prompts or full worker messages.
6. C6: concise startup/progress/results, documented commands and an ordinary scripted demonstration
   agree with actual behavior; a small authorized real-provider smoke is attempted if configured.
7. C7: existing controlled training/evaluation sample and offline quality checks pass; no unrelated
   infrastructure, scope expansion or claims of third-party training.

## Verification and delivery

Run locked dependency validation, Ruff format/check, strict mypy, targeted task tests and the
existing offline suite. Use pytest's no-cache provider and a unique temporary base directory.
If Windows sandbox ACLs prevent the existing interpreter or temporary directories from working,
use the existing Python 3.12 environment with this checkout's `src` on PYTHONPATH and an approved
ordinary offline run. Do not modify directory ACLs, delete historical caches, upgrade packages
or add stress/fuzz/concurrency tests to fix an environment issue.

Each implementation receives one independent review against its fixed matrix and, if necessary,
one bounded repair. Complete and integrate it locally before the next task. Record actual commit,
checks and unrun live checks in the handoff; never substitute test totals for functional evidence.
Local branches/commits and fast-forward integration are part of delivery. Publishing to a remote
repository is separate from these acceptance gates.

## P6-T15: Bounded conversation live repair

Authorized 2026-10-22; implementation baseline `26848297ca21886c58cb19bfc351daa4fd1ef269`.
Use GPT-6 Astra / xhigh for the serial implementer and independent reviewer. One review and at
most one bounded repair; no new architecture or production hardening task.

Inputs: ordinary Chinese input containing an arXiv URL followed immediately by punctuation/text;
observed HTTP-200 `SUPERVISOR_SCHEMA_FAILED` results; existing strict supervisor contracts; the
currently configured SHU model; public paper evidence and the previously authorized read-only
`kuangliu/pytorch-cifar` snapshot at `49b7aa97b0c12fe0d4054e670403a16b6b834ddd`.
The previous Code diagnostic deadline is not evidence of a product timeout defect.
Outputs: minimal repair, focused ordinary regressions, safe failure evidence and a truthful handoff.

Owned implementation paths: `application/services/conversation_intent.py`,
`adapters/llm/supervisor.py`, narrowly necessary `application/services/supervisor_models.py`,
`application/workflows/supervisor.py`, `cli/conversation.py`, directly related tests under
`tests/conversation/` and `tests/supervisor/`, and `docs/handoffs/P6-T15.md`. Diagnostic scripts
and generated evidence belong in a new ignored `var/validation/p6-t15-*` directory.
Coordinator owns README, this plan and parent coordination docs. Changes to shared LLM transport,
Research/Code planner or another path require a concrete observed need and coordinator approval.
Domain/ports, dependencies/lock, CI and actual `.env` are protected. No training/image access.

Fixed acceptance F1–F7:

1. F1: ordinary arXiv/GitHub URLs next to Chinese punctuation or explanatory text resolve to the
   intended canonical target. Existing multiple-target ambiguity, canonical-host/path checks and
   plain commands remain intact. Invalid ordinary inputs return a safe response, not an uncaught
   Pydantic exception or a raw input stuffed into a constrained target field.
2. F2: diagnose Supervisor parsing/response-shape failure using only safe categories, known field
   locations/types and tool-presence metadata. Repair the actual bounded structured-proposal path;
   preserve the closed action schema, no free-text JSON extraction, no invented success/action,
   and no switch to a different provider/model. Do not assume every HTTP 200 is a valid proposal.
3. F3: if schema correction is needed, allow at most one correction per decision, using the same
   schema and sanitized feedback; count both attempts in the shared budget before invocation.
   Provider/auth/timeout failure is not a schema retry. Final failure remains explicit, safe and
   cannot dispatch a worker or overwrite successful working context.
4. F4: ordinary in-scope requests still use the real supervisor; unsupported requests terminate
   without arbitrary tools/execution. Existing delegation bounds, paper-only/repository-only scope,
   candidate/repository Gates, reject, ordinary resume and source/file budgets do not regress.
5. F5: add only direct ordinary regressions for F1–F4. Ruff format/lint, strict mypy, locked
   dependency validation, affected tests and the existing full offline suite pass with no new
   dependency, raw-image/company-data access, transcript persistence or cancelled infrastructure.
6. F6: one small authorized live regression group checks the configured model's supervisor,
   explicit public paper, latest research and Code advice. Reuse the already authorized pinned
   snapshot for Code when practical and label cached-source evidence honestly; do not pretend
   this is a new GitHub download/Gate. A new live download/approval requires explicit human choice.
   No-candidates and user rejection are legitimate outcomes. External unavailability or a
   diagnostic deadline is recorded separately from a reproducible product defect; never replace
   a failed live run with scripted success or repeatedly rerun until passing.
7. F7: independent review and handoff identify actual commands, failures, schema/correction counts,
   live attempts versus unrun checks, protected-path diff and clean Git provenance. No secrets,
   raw provider body, full model response or private paths in tracked files or diagnostics.

Verification uses the existing development Python with public `src` on PYTHONPATH. Run Ruff
format/check, strict mypy, affected pytest and full `-m "not external and not gpu and not slurm"`
pytest with no-cache provider/unique temporary directories; run `uv --offline --no-cache lock
--check` if the locked environment permits it, and Git diff/protected-path checks. Keep real
dotenv disabled only at its default path for offline tests, not for synthetic dotenv fixtures.
Known sandbox ACL issues may use approved ordinary execution; do not edit ACLs or old caches.

External effects are limited to authorized SHU/public metadata calls, read-only analysis of the
already approved snapshot, ignored local artifacts, and local Git integration. Only central
Settings may read the key; disable external tracing for live probes. Use isolated context and
the abstract SEM target, not user dataset configuration. Live diagnosis has at most four calls;
post-repair verification is one bounded group, retains the per-request 24-call budget, and records
application-call versus SDK-retry counts honestly. A slow Code check may run up to six minutes;
do not turn the earlier two-minute diagnostic limit into a product acceptance requirement.

## P6-T16: Bounded Code advice correction

Authorized follow-up under the user's repair/development delegation; base
`c45260c5c0f8cbecd22c1941918ec4ccef8dc972`. Use GPT-6 Astra / xhigh, serial implementation,
one independent fixed-matrix review and at most one bounded repair. No architecture expansion.

Input: the newly approved live Code attempt reached `submit_adaptation_advice` after reading
six approved files, then failed validation with `advice.risks:too_long` and
`advice.items_to_verify:too_long` (existing maximum six each). Output: explicit prompt limits and
at most one sanitized final-advice length correction inside the existing ToolNode loop.
Do not truncate/discard model fields to manufacture success, weaken the schema, or increase limits.

Owned paths: `application/services/repository_insight_planner.py`, the existing
`prompts/repository_insight/` prompt, direct `tests/repository_insight/` tests and
`docs/handoffs/P6-T16.md`. Coordinator permits only a backward-compatible prompt-version update
in `application/services/repository_insight_models.py` if the prompt version changes; all advice
fields/constraints remain unchanged. README/this plan/parent coordination docs remain shared.
Do not change domain/ports, dependencies/lock, CI, Settings/transport, real dotenv, other workflows,
user context, source files in the approved snapshot, or add a general retry framework.

Fixed acceptance K1–K6:

1. K1: valid first submissions keep the actual four-tool Code path and typed output, with no
   unconditional extra call. Prompt explicitly describes the existing collection limits and
   asks for concise non-executable advice; stored prompt provenance stays accurate.
2. K2: an ordinary oversized final-advice list can yield one allowlisted schema-length feedback
   observation to the same model/tool schema, followed by one corrected submission. Feedback
   contains only declared field names, stable error types and schema limits, no original values,
   raw errors/model responses or source/prompt transcripts. No free-text JSON recovery or truncation.
3. K3: at most one correction per Code run, inside unchanged 10 model-turn / 10 tool-call,
   six-file / 48 KiB and shared 24-model-call limits. Count the correction before invocation.
   Exhaustion remains explicit; no second Gate, repository fetch or repeated source read is
   required just to correct final advice. Provider/auth/timeout and policy/path/execution errors
   are not treated as correctable list-length errors.
4. K4: repeated invalid output or a non-correctable failure cannot create a successful result,
   artifact or working context. Final accepted advice still passes all current schema, safety
   and actual-read-path validation. Human Gate and no-execution boundaries remain unchanged.
5. K5: direct ordinary tests cover first-valid, oversized-to-valid, repeated invalid, unchanged
   budgets and non-correctable errors; affected/full offline suites, Ruff/mypy and protected Git
   diffs pass. No pressure/fuzz, new dependency, provider switch or infrastructure.
6. K6: truthful handoff and one independent review. After freezing source, one bounded live
   planner check may reuse only the already approved fixed cached snapshot and real current model
   with abstract target/isolated artifacts. At most 20 model HTTP sends, existing application
   budgets and a six-minute diagnostic deadline; public metadata only if needed. No new download,
   human-approval replay, full-workflow success claim, or retry-until-pass. Record actual result,
   correction/budget counts and limits; a valid first attempt does not prove live correction.

Verification uses the established dev interpreter/public PYTHONPATH and isolated offline settings,
temporary test/cache directories, affected and full offline pytest, Ruff format/lint and strict
mypy. Lock already passed unchanged under P6-T15; verify no dependency diff rather than reinstall.
External effects: authorized SHU/public metadata checks, read-only approved cached source, ignored
artifacts, local commits and reviewed fast-forward integration only. No push/tag, raw images,
third-party execution, patch/smoke/training or changes to the real API configuration.

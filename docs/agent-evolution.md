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

All three tasks have completed their independent fixed-matrix reviews and local integration.
No further implementation task is ready under this plan. Remote publication is separate.
The final offline suite passed 746 tests with one existing Windows symlink skip. Authorized
live attempts produced mixed results; complete final live paper-to-Code execution and provider
stability are not claimed as verified. See the P6-T13 handoff for exact evidence and limits.

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

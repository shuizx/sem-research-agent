"""Deterministic patch tool for the sole controlled PLAIN_PYTORCH fixture."""

from __future__ import annotations

import difflib
import json
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from vision_research_ops.application.services.adaptation_models import (
    REQUIRED_PATCH_FIELDS,
    CompiledAdaptationPlan,
    PatchArtifactRecord,
    PatchChangeRecord,
    PatchField,
)
from vision_research_ops.application.services.adaptation_patch import (
    ADAPTATION_CONFIG_PATH,
    PatchPolicyError,
    canonical_json_bytes,
    compile_adaptation_config,
    content_hash,
    validate_patch_fields,
    validate_patch_path,
)
from vision_research_ops.application.services.adaptation_planning import (
    ALLOWED_FIXTURE_REPOSITORY_URL,
)
from vision_research_ops.application.services.adaptation_store import LocalAdaptationStore
from vision_research_ops.ports import OperationContext

_FIXTURE_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "config.yaml",
        "data.py",
        "fixture_probe.py",
        "model.py",
        "train.py",
        "sem_adaptation.json",
        "sem_adapter.py",
    }
)
_EXPECTED_FIXTURE_HASHES = {
    "LICENSE": "bd3678da1f9613a2bee3c58ca013e8c339f10f7c52ec3cea231b948c336aa3b7",
    "README.md": "69197c48f65f91cafb4ccd56ec32c0d310d14f54e73c0fc605954369e4ae635d",
    "config.yaml": "49452cea7b09e7e063f58a8ead5f590664b72ec4cfc7a0a31227af3968362045",
    "data.py": "8435d2153ab8ab04d1c8ff4d033d149c2fc25c2057b2ae9006cb64d7586760a3",
    "fixture_probe.py": "3ebea2917edd173797ce2746d712dd81f97b90ed5bb7462438bf9a4e838dca63",
    "model.py": "991897f3414caff7612e677292d78d04040e1c1ed306511a57d2321de95c1518",
    "train.py": "30cafdbcf196995f31fb51fb824da396727a690037c5f2bfa326a1051041d03f",
    "sem_adaptation.json": "128dd54e8ddef1061e85bce1004f28a87933e043b4713e20fc6fe3b80c86ee6e",
    "sem_adapter.py": "1e0e5301db77ab2c6a25e061edc112c085e42d527cd7fa7f3c266b25d563927f",
}
_ORDERED_FIELDS: list[PatchField] = [
    "/input/channels",
    "/model/num_classes",
    "/data/label_mapping",
    "/data/group_split_key",
    "/metrics/names",
    "/metrics/output_file",
]


class FixturePatchTool:
    """Copy one allowlisted fixture and apply one reproducible JSON config diff."""

    def __init__(
        self,
        *,
        fixture_root: Path,
        store: LocalAdaptationStore,
        clock: Callable[[], datetime],
    ) -> None:
        self._fixture_root = fixture_root
        self._store = store
        self._clock = clock
        self.call_count = 0

    def _validate_fixture_source(self) -> None:
        if not self._fixture_root.is_dir():
            raise PatchPolicyError("controlled fixture root is unavailable")
        observed: set[str] = set()
        for path in self._fixture_root.rglob("*"):
            if path.is_symlink():
                raise PatchPolicyError("fixture source cannot contain symlinks")
            if path.is_file():
                relative = path.relative_to(self._fixture_root).as_posix()
                if relative == ADAPTATION_CONFIG_PATH:
                    validate_patch_path(relative)
                observed.add(relative)
                text = path.read_text(encoding="utf-8")
                normalized = text.replace("\r\n", "\n").encode("utf-8")
                expected = _EXPECTED_FIXTURE_HASHES.get(relative)
                if expected is None or content_hash(normalized) != f"sha256:{expected}":
                    raise PatchPolicyError("fixture source content is not allowlisted")
        if observed != _FIXTURE_FILES:
            raise PatchPolicyError("fixture source does not match its exact file allowlist")

    def _workspace_ref(self, workflow_id: str, revision: int) -> str:
        return f"workspaces/{workflow_id}/r{revision}"

    def _patch_ref(self, workflow_id: str, revision: int) -> str:
        return f"patches/{workflow_id}/r{revision}/change.patch"

    async def apply(
        self,
        plan: CompiledAdaptationPlan,
        *,
        ctx: OperationContext,
    ) -> PatchArtifactRecord:
        """Materialize a fixed workspace, apply config, and export the actual diff."""
        if not ctx.idempotency_key:
            raise PatchPolicyError("patch application requires an idempotency key")
        if (
            plan.repository_url != ALLOWED_FIXTURE_REPOSITORY_URL
            or plan.repository_kind != "CONTROLLED_PLAIN_PYTORCH_FIXTURE"
        ):
            raise PatchPolicyError("patch tool only accepts the controlled fixture repository")
        validate_patch_path(ADAPTATION_CONFIG_PATH)
        validate_patch_fields(_ORDERED_FIELDS)
        if {change.target_field for change in plan.proposal.changes} != REQUIRED_PATCH_FIELDS:
            raise PatchPolicyError("plan change targets do not match the fixed template")

        manifest_ref = self._store.patch_manifest_ref(plan.workflow_id, plan.revision)
        manifest_path = self._store.resolve_ref(manifest_ref)
        workspace_ref = self._workspace_ref(plan.workflow_id, plan.revision)
        workspace = self._store.resolve_ref(workspace_ref)
        patch_ref = self._patch_ref(plan.workflow_id, plan.revision)
        patch_path = self._store.resolve_ref(patch_ref)
        if manifest_path.exists():
            existing = self._store.load_patch_record(manifest_ref)
            if (
                existing.base_commit_sha != plan.base_commit_sha
                or existing.plan_id != plan.plan_id
                or existing.plan_revision != plan.revision
                or not workspace.is_dir()
                or not patch_path.is_file()
                or content_hash(patch_path.read_bytes()) != existing.patch_hash
            ):
                raise PatchPolicyError("existing patch revision conflicts with the requested plan")
            return existing

        self._validate_fixture_source()
        if workspace.exists():
            raise PatchPolicyError("untracked workspace already exists for this patch revision")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir()
        for relative in sorted(_FIXTURE_FILES):
            source = self._fixture_root / Path(*relative.split("/"))
            destination = workspace / Path(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        config_path = workspace / ADAPTATION_CONFIG_PATH
        before = config_path.read_bytes()
        try:
            parsed_before = json.loads(before.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PatchPolicyError("fixture adaptation config must be UTF-8 JSON") from error
        if not isinstance(parsed_before, dict):
            raise PatchPolicyError("fixture adaptation config must be a JSON object")
        after = canonical_json_bytes(compile_adaptation_config(plan))
        config_path.write_bytes(after)
        diff = "".join(
            difflib.unified_diff(
                before.decode("utf-8").splitlines(keepends=True),
                after.decode("utf-8").splitlines(keepends=True),
                fromfile=f"a/{ADAPTATION_CONFIG_PATH}",
                tofile=f"b/{ADAPTATION_CONFIG_PATH}",
                lineterm="\n",
            )
        ).encode("utf-8")
        if not diff:
            raise PatchPolicyError("compiled patch did not change the fixture config")
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_bytes(diff)
        now = self._clock()
        record = PatchArtifactRecord(
            workflow_id=plan.workflow_id,
            attempt_id=f"adaptation-attempt-{plan.workflow_id}-r{plan.revision}",
            attempt_number=plan.revision,
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            repository_id=plan.repository_id,
            base_commit_sha=plan.base_commit_sha,
            dataset_version=plan.dataset_version,
            patch_hash=content_hash(diff),
            workspace_ref=workspace_ref,
            patch_ref=patch_ref,
            manifest_ref=manifest_ref,
            changes=[
                PatchChangeRecord(
                    path=ADAPTATION_CONFIG_PATH,
                    operation="MODIFY",
                    field_paths=_ORDERED_FIELDS,
                    before_hash=content_hash(before),
                    after_hash=content_hash(after),
                )
            ],
            created_at=now,
        )
        self._store.write_patch_record(record)
        self.call_count += 1
        return record


__all__ = ["FixturePatchTool"]

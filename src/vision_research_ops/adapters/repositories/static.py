"""Bounded zip inspection for two small PyTorch classification layouts."""

from __future__ import annotations

import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from vision_research_ops.domain import (
    ArtifactRef,
    Reason,
    RiskFinding,
    SeverityLevel,
)
from vision_research_ops.ports import (
    OperationContext,
    ProviderError,
    RepositoryAnalysis,
    RepositoryFileSummary,
    RepositoryPolicy,
    RepositoryResolution,
    make_failure,
)

_MAX_FILES = 500
_MAX_TOTAL_BYTES = 8 * 1024 * 1024
_MAX_TEXT_FILE_BYTES = 256 * 1024
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}
_DEPENDENCY_NAMES = {
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}
_ENTRYPOINT_NAMES = {"main.py", "train.py", "training.py"}
_CONFIG_SUFFIXES = {".json", ".toml", ".yaml", ".yml"}
_DANGEROUS_PATTERNS = {
    "DYNAMIC_EXEC": re.compile(r"\b(?:eval|exec)\s*\("),
    "OS_SYSTEM": re.compile(r"\bos\.system\s*\("),
    "SUBPROCESS": re.compile(r"\bsubprocess\.(?:call|run|Popen)\s*\("),
}


def _safe_member_path(name: str) -> PurePosixPath:
    if "\\" in name or "%" in name:
        raise ValueError("archive member path is not canonical POSIX")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("archive member path escapes the repository")
    return path


def _is_symlink(info: ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


class ZipStaticRepositoryAnalyzer:
    """Inspect source text and file names without extraction, import, or execution."""

    def __init__(self, *, artifact_root: Path) -> None:
        self._artifact_root = artifact_root.resolve()

    def _artifact_path(self, artifact: ArtifactRef) -> Path:
        relative = Path(artifact.uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("repository archive URI must stay below the artifact root")
        resolved = (self._artifact_root / relative).resolve()
        if not resolved.is_relative_to(self._artifact_root):
            raise ValueError("repository archive URI escapes the artifact root")
        return resolved

    @staticmethod
    def _resolution(artifact: ArtifactRef) -> RepositoryResolution:
        return RepositoryResolution.model_validate(artifact.metadata)

    @staticmethod
    def _repository_paths(infos: list[ZipInfo]) -> dict[ZipInfo, str]:
        safe = {info: _safe_member_path(info.filename) for info in infos}
        roots = {path.parts[0] for path in safe.values() if path.parts}
        strip_root = len(roots) == 1
        result: dict[ZipInfo, str] = {}
        for info, path in safe.items():
            parts = path.parts[1:] if strip_root else path.parts
            if parts:
                result[info] = PurePosixPath(*parts).as_posix()
        return result

    @staticmethod
    def _license_from_text(files: dict[str, str]) -> str | None:
        license_text = "\n".join(
            text
            for path, text in files.items()
            if PurePosixPath(path).name.casefold().startswith(("license", "copying"))
        ).casefold()
        if "mit license" in license_text:
            return "MIT"
        if "apache license" in license_text and "version 2.0" in license_text:
            return "Apache-2.0"
        if "gnu general public license" in license_text:
            return "GPL-3.0-only"
        if "redistribution and use in source and binary forms" in license_text:
            return "BSD-3-Clause"
        return None

    async def analyze(
        self,
        repository_archive: ArtifactRef,
        policy: RepositoryPolicy,
        *,
        ctx: OperationContext,
    ) -> RepositoryAnalysis:
        """Return deterministic evidence for ordinary small classification repositories."""
        del ctx
        try:
            archive_path = self._artifact_path(repository_archive)
            resolution = self._resolution(repository_archive)
            with ZipFile(archive_path) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_FILES:
                    raise ValueError("repository archive contains too many files")
                if sum(info.file_size for info in infos) > _MAX_TOTAL_BYTES:
                    raise ValueError("repository archive is too large for pipeline inspection")
                mapped = self._repository_paths(infos)
                text_files: dict[str, str] = {}
                summaries: list[RepositoryFileSummary] = []
                for info, path in sorted(mapped.items(), key=lambda item: item[1].casefold()):
                    kind: Literal["FILE", "DIRECTORY", "SYMLINK"] = (
                        "DIRECTORY" if info.is_dir() else "SYMLINK" if _is_symlink(info) else "FILE"
                    )
                    summaries.append(
                        RepositoryFileSummary(
                            schema_version="1",
                            path=path,
                            size_bytes=info.file_size,
                            kind=kind,
                        )
                    )
                    suffix = PurePosixPath(path).suffix.casefold()
                    basename = PurePosixPath(path).name.casefold()
                    if (
                        kind == "FILE"
                        and (
                            suffix in _TEXT_SUFFIXES or basename.startswith(("license", "copying"))
                        )
                        and info.file_size <= _MAX_TEXT_FILE_BYTES
                    ):
                        text_files[path] = archive.read(info).decode("utf-8", errors="replace")
        except (BadZipFile, KeyError, OSError, UnicodeError, ValueError):
            raise ProviderError(
                make_failure(
                    code="REPOSITORY_STATIC_ANALYSIS_FAILED",
                    category="STATIC_ANALYSIS",
                    message="The bounded repository archive could not be inspected safely.",
                    retryable=False,
                    ctx=None,
                )
            ) from None

        python_files = {
            path: text for path, text in text_files.items() if path.casefold().endswith(".py")
        }
        entrypoints = sorted(
            path
            for path in python_files
            if PurePosixPath(path).name.casefold() in _ENTRYPOINT_NAMES
        )
        data_loaders = sorted(
            path
            for path, text in python_files.items()
            if re.search(r"\b(?:DataLoader|Dataset|ImageFolder)\b", text)
        )
        dependencies = sorted(
            path for path in text_files if PurePosixPath(path).name.casefold() in _DEPENDENCY_NAMES
        )
        configurations = sorted(
            path
            for path in text_files
            if PurePosixPath(path).suffix.casefold() in _CONFIG_SUFFIXES
            and PurePosixPath(path).name.casefold() not in _DEPENDENCY_NAMES
        )

        framework_evidence: list[str] = []
        model_head_evidence: list[str] = []
        has_torchvision_or_timm = False
        for path, text in sorted(python_files.items()):
            if re.search(r"(?:^|\n)\s*(?:import torch|from torch\b)", text):
                framework_evidence.append(f"PYTORCH_IMPORT:{path}")
            if re.search(r"\b(?:torchvision|timm)\b", text):
                framework_evidence.append(f"TORCHVISION_OR_TIMM:{path}")
                has_torchvision_or_timm = True
            if re.search(r"\b(?:CrossEntropyLoss|cross_entropy|NLLLoss|nll_loss)\b", text):
                framework_evidence.append(f"CLASSIFICATION_LOSS:{path}")
            if re.search(r"\b(?:classifier|num_classes|n_classes|class_count)\b", text):
                model_head_evidence.append(path)
                framework_evidence.append(f"MODEL_HEAD:{path}")

        findings: list[RiskFinding] = []
        for path, text in sorted(python_files.items()):
            for rule_id, pattern in _DANGEROUS_PATTERNS.items():
                if pattern.search(text):
                    findings.append(
                        RiskFinding(
                            schema_version="1",
                            finding_id=f"risk-{len(findings) + 1}",
                            rule_id=rule_id,
                            category="UNSAFE_EXECUTION_PATTERN",
                            severity=SeverityLevel.HIGH,
                            description="Static source contains a disallowed execution primitive.",
                            location_ref=path,
                        )
                    )

        has_pytorch = any(item.startswith("PYTORCH_IMPORT:") for item in framework_evidence)
        has_classification_loss = any(
            item.startswith("CLASSIFICATION_LOSS:") for item in framework_evidence
        )
        has_classification_head = any(item.startswith("MODEL_HEAD:") for item in framework_evidence)
        supported = bool(
            python_files
            and has_pytorch
            and has_classification_loss
            and has_classification_head
            and entrypoints
            and data_loaders
            and not findings
        )
        reasons = [
            Reason(
                schema_version="1",
                code="REPOSITORY_STRUCTURE_SUPPORTED"
                if supported
                else "REPOSITORY_STRUCTURE_UNSUPPORTED",
                message=(
                    "Static evidence matches a supported PyTorch classification layout."
                    if supported
                    else "Required PyTorch classification structure is missing or unsafe."
                ),
            )
        ]
        if configurations:
            framework_evidence.append(f"CONFIGURATION_FILES:{len(configurations)}")
        framework_evidence.append(
            "STRUCTURE:TORCHVISION_TIMM"
            if supported and has_torchvision_or_timm
            else "STRUCTURE:PLAIN_PYTORCH"
            if supported
            else "STRUCTURE:UNSUPPORTED"
        )
        return RepositoryAnalysis(
            schema_version="1",
            repository=resolution,
            policy=policy,
            file_tree_summary=summaries,
            dependency_files=dependencies,
            framework_evidence=framework_evidence,
            entrypoint_candidates=entrypoints,
            data_loader_candidates=data_loaders,
            command_candidates=[],
            license_spdx=self._license_from_text(text_files),
            dangerous_patterns=findings,
            supported=supported,
            support_reasons=reasons,
        )


__all__ = ["ZipStaticRepositoryAnalyzer"]

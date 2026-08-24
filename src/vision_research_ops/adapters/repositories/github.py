"""Read-only GitHub adapter that pins and snapshots public repositories."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.parse import quote
from urllib.request import Request, urlopen

from vision_research_ops.application.services.repository_models import (
    normalize_github_repository_url,
)
from vision_research_ops.domain import ArtifactKind, ArtifactRef, JsonObject
from vision_research_ops.ports import (
    OperationContext,
    ProviderError,
    RepositoryMetadata,
    RepositoryResolution,
    make_failure,
)

GitHubTransport = Callable[[str, Mapping[str, str], int], bytes]

_API_ROOT = "https://api.github.com"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 25 * 1024 * 1024


def _utc_now() -> datetime:
    return datetime.now(UTC)


class GitHubRepositoryProvider:
    """Resolve metadata and bounded zip snapshots without running repository code."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        api_token: str | None = None,
        timeout_seconds: int = 20,
        transport: GitHubTransport | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("GitHub timeout_seconds must be positive")
        if api_token is not None and not api_token.strip():
            raise ValueError("GitHub api_token must be non-blank when supplied")
        self._artifact_root = artifact_root
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds
        self._transport = transport or self._fetch
        self._clock = clock

    @staticmethod
    def _fetch(url: str, headers: Mapping[str, str], timeout_seconds: int) -> bytes:
        request = Request(url, headers=dict(headers))
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = cast(bytes, response.read(_MAX_ARCHIVE_BYTES + 1))
        if len(payload) > _MAX_ARCHIVE_BYTES:
            raise ValueError("GitHub response exceeds the bounded response size")
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "SEM Research Agent/0.1 pipeline-repository-agent",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._api_token is not None:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    async def _request(self, url: str, *, ctx: OperationContext, archive: bool = False) -> bytes:
        if ctx.deadline_exceeded(now=self._clock()):
            raise ProviderError(
                make_failure(
                    code="GITHUB_REQUEST_DEADLINE_EXCEEDED",
                    category="TIMEOUT",
                    message="The GitHub request deadline elapsed before retrieval.",
                    retryable=True,
                    ctx=ctx,
                )
            )
        try:
            payload = await asyncio.to_thread(
                self._transport,
                url,
                self._headers(),
                self._timeout_seconds,
            )
            limit = _MAX_ARCHIVE_BYTES if archive else _MAX_JSON_BYTES
            if len(payload) > limit:
                raise ValueError("GitHub response exceeds its bounded size")
            return payload
        except (OSError, TimeoutError, UnicodeError, ValueError):
            raise ProviderError(
                make_failure(
                    code="GITHUB_PROVIDER_REQUEST_FAILED",
                    category="PROVIDER",
                    message="The bounded GitHub request failed.",
                    retryable=True,
                    ctx=ctx,
                )
            ) from None

    async def _request_json(self, url: str, *, ctx: OperationContext) -> JsonObject:
        payload = await self._request(url, ctx=ctx)
        try:
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("GitHub JSON response must be an object")
            return cast(JsonObject, value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ProviderError(
                make_failure(
                    code="GITHUB_PROVIDER_RESPONSE_INVALID",
                    category="PROVIDER_SCHEMA",
                    message="GitHub returned invalid repository metadata.",
                    retryable=True,
                    ctx=ctx,
                )
            ) from None

    async def resolve(
        self,
        repository_url: str,
        revision: str | None,
        *,
        ctx: OperationContext,
    ) -> RepositoryResolution:
        """Resolve a canonical GitHub URL and ref to a complete lowercase SHA."""
        try:
            locator = normalize_github_repository_url(repository_url)
        except ValueError:
            raise ProviderError(
                make_failure(
                    code="GITHUB_REPOSITORY_URL_INVALID",
                    category="INPUT",
                    message="The repository URL is not an allowed public GitHub repository.",
                    retryable=False,
                    ctx=ctx,
                )
            ) from None
        requested_revision = "HEAD" if revision is None else revision
        if not requested_revision.strip() or len(requested_revision) > 255:
            raise ProviderError(
                make_failure(
                    code="GITHUB_REVISION_INVALID",
                    category="INPUT",
                    message="The requested GitHub revision is invalid.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        encoded_revision = quote(requested_revision, safe="")
        url = f"{_API_ROOT}/repos/{locator.owner}/{locator.name}/commits/{encoded_revision}"
        payload = await self._request_json(url, ctx=ctx)
        commit_sha = payload.get("sha")
        if not isinstance(commit_sha, str) or _COMMIT_RE.fullmatch(commit_sha) is None:
            raise ProviderError(
                make_failure(
                    code="GITHUB_COMMIT_SHA_INVALID",
                    category="PROVIDER_SCHEMA",
                    message="GitHub did not return a complete lowercase commit SHA.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        return RepositoryResolution(
            schema_version="1",
            canonical_url=locator.canonical_url,
            provider="GITHUB",
            owner=locator.owner,
            name=locator.name,
            commit_sha=commit_sha,
        )

    async def fetch_metadata(
        self,
        repository: RepositoryResolution,
        *,
        ctx: OperationContext,
    ) -> RepositoryMetadata:
        """Read bounded repository and language metadata from fixed API endpoints."""
        base = f"{_API_ROOT}/repos/{repository.owner}/{repository.name}"
        repo_payload, language_payload = await asyncio.gather(
            self._request_json(base, ctx=ctx),
            self._request_json(f"{base}/languages", ctx=ctx),
        )
        default_branch = repo_payload.get("default_branch")
        license_value = repo_payload.get("license")
        license_spdx = None
        if isinstance(license_value, dict):
            candidate = license_value.get("spdx_id")
            if isinstance(candidate, str) and candidate.strip() and candidate != "NOASSERTION":
                license_spdx = candidate
        languages: dict[str, int] = {}
        for key, value in language_payload.items():
            if (
                isinstance(key, str)
                and key.strip()
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            ):
                languages[key] = value
        return RepositoryMetadata(
            schema_version="1",
            repository=repository,
            license_spdx=license_spdx,
            languages=languages,
            default_branch=default_branch if isinstance(default_branch, str) else None,
            raw_fields=cast(
                JsonObject,
                {
                    "archived": bool(repo_payload.get("archived", False)),
                    "fork": bool(repo_payload.get("fork", False)),
                    "html_url": repository.canonical_url,
                },
            ),
        )

    async def snapshot(
        self,
        repository: RepositoryResolution,
        *,
        ctx: OperationContext,
    ) -> ArtifactRef:
        """Download one immutable commit archive and expose only a relative artifact URI."""
        filename = f"{repository.owner}-{repository.name}-{repository.commit_sha}.zip"
        relative = Path("snapshots") / filename
        destination = self._artifact_root / relative
        url = (
            f"{_API_ROOT}/repos/{repository.owner}/{repository.name}/zipball/"
            f"{repository.commit_sha}"
        )
        payload = await self._request(url, ctx=ctx, archive=True)
        digest_hex = sha256(payload).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_bytes()
            if sha256(existing).hexdigest() != digest_hex:
                raise ProviderError(
                    make_failure(
                        code="GITHUB_SNAPSHOT_CONFLICT",
                        category="INTEGRITY",
                        message="An existing snapshot conflicts with the resolved commit archive.",
                        retryable=False,
                        ctx=ctx,
                    )
                )
        else:
            temporary = destination.with_suffix(".zip.tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        return ArtifactRef(
            schema_version="1",
            artifact_id=f"archive-{repository.commit_sha}",
            kind=ArtifactKind.REPOSITORY_ARCHIVE,
            uri=relative.as_posix(),
            sha256=f"sha256:{digest_hex}",
            size_bytes=len(payload),
            media_type="application/zip",
            created_at=self._clock().astimezone(UTC),
            producer="github-repository-provider-v1",
            sensitivity="PUBLIC",
            metadata=cast(
                JsonObject,
                {
                    "schema_version": "1",
                    "canonical_url": repository.canonical_url,
                    "provider": repository.provider,
                    "owner": repository.owner,
                    "name": repository.name,
                    "commit_sha": repository.commit_sha,
                },
            ),
        )


__all__ = ["GitHubRepositoryProvider", "GitHubTransport"]

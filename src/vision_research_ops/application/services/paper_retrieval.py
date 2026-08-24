"""Deterministic retrieval-window, normalization, and deduplication services."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from pydantic import TypeAdapter

from vision_research_ops.domain import ProvenanceRef, QuerySpec, UTCDateTime
from vision_research_ops.ports import (
    OperationContext,
    PaperProvider,
    PaperQuery,
    RawPaperRecord,
)

from .paper_models import ResearchPaper, RetrievalWindow

_UTC_ADAPTER: TypeAdapter[datetime] = TypeAdapter(UTCDateTime)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_NON_ID_CHARACTER_RE = re.compile(r"[^a-z0-9._-]+")


def stable_unique[T](values: Iterable[T]) -> list[T]:
    """Return first occurrences without requiring values to be sortable."""
    result: list[T] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def compute_retrieval_window(
    *,
    now: datetime,
    last_successful_run_at: datetime | None,
    overlap: timedelta,
    initial_lookback: timedelta,
) -> RetrievalWindow:
    """Build an aware UTC window with overlap after a previous successful run."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if overlap < timedelta(0):
        raise ValueError("overlap must not be negative")
    if initial_lookback <= timedelta(0):
        raise ValueError("initial_lookback must be positive")
    end_at = now.astimezone(UTC)
    if last_successful_run_at is None:
        start_at = end_at - initial_lookback
    else:
        if last_successful_run_at.tzinfo is None or last_successful_run_at.utcoffset() is None:
            raise ValueError("last_successful_run_at must be timezone-aware")
        start_at = last_successful_run_at.astimezone(UTC) - overlap
    if start_at >= end_at:
        raise ValueError("computed retrieval window must not be empty")
    return RetrievalWindow(start_at=start_at, end_at=end_at)


def query_for_window(base: QuerySpec, window: RetrievalWindow) -> QuerySpec:
    """Copy a query with inclusive provider calendar dates for the exact UTC window."""
    return base.model_copy(
        update={
            "date_from": window.start_at.date(),
            "date_to": window.end_at.date(),
        }
    )


async def collect_provider_records(
    provider: PaperProvider,
    *,
    query_id: str,
    query_spec: QuerySpec,
    max_pages: int,
    max_records: int,
    page_size: int,
    ctx: OperationContext,
) -> tuple[list[RawPaperRecord], int]:
    """Collect bounded provider pages and reject cursor cycles or provider mismatch."""
    if min(max_pages, max_records) <= 0:
        return [], 0
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    paper_query = PaperQuery(
        schema_version="1",
        query_id=query_id,
        query_spec=query_spec,
        page_size=min(page_size, max_records),
    )
    cursor: str | None = None
    seen_cursors: set[str | None] = set()
    records: list[RawPaperRecord] = []
    pages_used = 0

    for _ in range(max_pages):
        if cursor in seen_cursors:
            raise ValueError("paper provider returned a cursor cycle")
        seen_cursors.add(cursor)
        page = await provider.search(paper_query, cursor=cursor, ctx=ctx)
        pages_used += 1
        if page.provider_name != provider.provider_name:
            raise ValueError("paper page provider_name does not match the injected provider")
        remaining = max_records - len(records)
        records.extend(page.records[:remaining])
        if len(records) >= max_records or page.next_cursor is None:
            break
        cursor = page.next_cursor
    return records, pages_used


def _required_text(raw: Mapping[str, object], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"raw paper field {name} must be a non-blank string")
    return " ".join(value.split())


def _optional_text(raw: Mapping[str, object], name: str) -> str | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"raw paper field {name} must be null or a non-blank string")
    return " ".join(value.split())


def _text_list(raw: Mapping[str, object], name: str) -> list[str]:
    value = raw.get(name, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"raw paper field {name} must be a list of non-blank strings")
    return stable_unique(" ".join(item.split()) for item in value)


def normalize_arxiv_id(value: str) -> str:
    """Normalize arXiv URL/prefix/version spellings to one base identifier."""
    normalized = value.strip()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "arXiv:",
        "arxiv:",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = _ARXIV_VERSION_RE.sub("", normalized)
    if not normalized:
        raise ValueError("arXiv identifier must not be empty")
    return normalized


def normalize_doi(value: str) -> str:
    """Normalize DOI URL/prefix and case while retaining the DOI payload."""
    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if not normalized:
        raise ValueError("DOI must not be empty")
    return normalized


def normalize_title_identity(value: str) -> str:
    """Create a Unicode-aware comparison key without changing the displayed title."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).split())


def _content_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _paper_id(provider_name: str, record_id: str, arxiv_id: str | None) -> str:
    if arxiv_id is not None:
        safe = _NON_ID_CHARACTER_RE.sub("-", arxiv_id.casefold()).strip("-")
        return f"paper-arxiv-{safe}"
    digest = sha256(f"{provider_name}:{record_id}".encode()).hexdigest()[:20]
    return f"paper-{provider_name.casefold()}-{digest}"


def normalize_raw_paper(record: RawPaperRecord) -> ResearchPaper:
    """Normalize one provider record into the public Research Agent representation."""
    raw = dict(record.raw_fields)
    external_ids = {item.provider_name.casefold(): item.value for item in record.external_ids}
    raw_arxiv = external_ids.get("arxiv") or _optional_text(raw, "arxiv_id")
    raw_doi = external_ids.get("doi") or _optional_text(raw, "doi")
    arxiv_id = None if raw_arxiv is None else normalize_arxiv_id(raw_arxiv)
    doi = None if raw_doi is None else normalize_doi(raw_doi)
    title = _required_text(raw, "title")[:1024]
    abstract = _required_text(raw, "abstract")[:8000]
    published_at = _UTC_ADAPTER.validate_python(_required_text(raw, "published_at"))
    updated_at = _UTC_ADAPTER.validate_python(
        _optional_text(raw, "updated_at") or _required_text(raw, "published_at")
    )
    entry_url = _optional_text(raw, "entry_url")
    if entry_url is None:
        if arxiv_id is None:
            raise ValueError("raw paper requires entry_url when no arXiv identifier exists")
        entry_url = f"https://arxiv.org/abs/{arxiv_id}"
    comment = _optional_text(raw, "comment")
    if comment is not None:
        comment = comment[:1024]

    return ResearchPaper(
        paper_id=_paper_id(record.provider_name, record.provider_record_id, arxiv_id),
        provider_name=record.provider_name,
        provider_record_ids=[record.provider_record_id],
        arxiv_id=arxiv_id,
        doi=doi,
        title=title,
        abstract=abstract,
        authors=_text_list(raw, "authors"),
        categories=_text_list(raw, "categories"),
        published_at=published_at,
        updated_at=updated_at,
        entry_url=entry_url,
        pdf_url=_optional_text(raw, "pdf_url"),
        comment=comment,
        code_urls=_text_list(raw, "code_urls"),
        provenance=[
            ProvenanceRef(
                schema_version="1",
                source_type="provider",
                source_id=record.provider_record_id,
                source_url=entry_url,
                retrieved_at=record.retrieved_at,
                content_hash=_content_hash(raw),
            )
        ],
    )


def _identity_keys(paper: ResearchPaper) -> list[str]:
    keys = [f"title:{normalize_title_identity(paper.title)}"]
    if paper.arxiv_id is not None:
        keys.append(f"arxiv:{paper.arxiv_id.casefold()}")
    if paper.doi is not None:
        keys.append(f"doi:{paper.doi.casefold()}")
    return keys


def _merge_group(group: list[ResearchPaper]) -> ResearchPaper:
    primary = max(
        group,
        key=lambda paper: (paper.updated_at, paper.published_at, paper.paper_id),
    )
    return primary.model_copy(
        update={
            "provider_record_ids": stable_unique(
                record_id for paper in group for record_id in paper.provider_record_ids
            ),
            "authors": stable_unique(author for paper in group for author in paper.authors),
            "categories": stable_unique(
                category for paper in group for category in paper.categories
            ),
            "code_urls": stable_unique(url for paper in group for url in paper.code_urls),
            "provenance": stable_unique(
                provenance for paper in group for provenance in paper.provenance
            ),
        }
    )


def normalize_and_deduplicate(records: Iterable[RawPaperRecord]) -> list[ResearchPaper]:
    """Normalize records and merge any arXiv, DOI, or title-connected duplicates."""
    papers = [normalize_raw_paper(record) for record in records]
    parents = list(range(len(papers)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    identity_owner: dict[str, int] = {}
    for index, paper in enumerate(papers):
        for identity in _identity_keys(paper):
            previous = identity_owner.setdefault(identity, index)
            union(index, previous)

    groups: dict[int, list[ResearchPaper]] = {}
    for index, paper in enumerate(papers):
        groups.setdefault(find(index), []).append(paper)
    merged = [_merge_group(group) for group in groups.values()]
    return sorted(
        merged,
        key=lambda paper: (
            -paper.published_at.timestamp(),
            normalize_title_identity(paper.title),
            paper.paper_id,
        ),
    )


def within_window(paper: ResearchPaper, window: RetrievalWindow) -> bool:
    """Return whether publication or revision activity lies inside the exact interval."""
    activity_at = max(paper.published_at, paper.updated_at)
    return window.start_at <= activity_at <= window.end_at


__all__ = [
    "collect_provider_records",
    "compute_retrieval_window",
    "normalize_and_deduplicate",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_raw_paper",
    "normalize_title_identity",
    "query_for_window",
    "stable_unique",
    "within_window",
]

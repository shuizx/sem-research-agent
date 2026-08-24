"""Small, read-only arXiv Atom adapter for the research Research Agent."""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from vision_research_ops.domain import JsonObject
from vision_research_ops.ports import (
    ExternalPaperId,
    OperationContext,
    PaperQuery,
    PaperSearchPage,
    ProviderError,
    RawPaperRecord,
    make_failure,
)
from vision_research_ops.settings import ARXIV_API_URL

_ATOM = "http://www.w3.org/2005/Atom"
_ARXIV = "http://arxiv.org/schemas/atom"
_OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
_CODE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
)
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


class ArxivPaperProvider:
    """Retrieve bounded public arXiv metadata without downloading PDFs or code."""

    provider_name = "arxiv"

    def __init__(
        self,
        *,
        timeout_seconds: int = 20,
        transport: Callable[[str, int], bytes] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("arXiv timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._transport = transport or self._fetch
        self._clock = clock

    @staticmethod
    def _fetch(url: str, timeout_seconds: int) -> bytes:
        request = Request(
            url,
            headers={"User-Agent": "SEM Research Agent/0.1 pipeline-research-agent"},
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = cast(bytes, response.read(_MAX_RESPONSE_BYTES + 1))
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise ValueError("arXiv response exceeds the bounded response size")
        return payload

    @staticmethod
    def _quoted(value: str) -> str:
        return value.replace('"', " ").strip()

    @classmethod
    def _search_expression(cls, query: PaperQuery) -> str:
        spec = query.query_spec
        keyword_parts = [f'all:"{cls._quoted(term)}"' for term in spec.keywords]
        domain_parts = [f"cat:{cls._quoted(domain)}" for domain in spec.domains]
        positive_groups = []
        if keyword_parts:
            positive_groups.append(f"({' OR '.join(keyword_parts)})")
        if domain_parts:
            positive_groups.append(f"({' OR '.join(domain_parts)})")
        expression = " AND ".join(positive_groups)
        for excluded in spec.excluded_terms:
            expression += f' ANDNOT all:"{cls._quoted(excluded)}"'
        if spec.date_from is not None and spec.date_to is not None:
            lower = spec.date_from.strftime("%Y%m%d0000")
            upper = spec.date_to.strftime("%Y%m%d2359")
            expression += f" AND submittedDate:[{lower} TO {upper}]"
        return expression

    @classmethod
    def _search_url(cls, query: PaperQuery, start: int) -> str:
        params = {
            "search_query": cls._search_expression(query),
            "start": str(start),
            "max_results": str(query.page_size),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return f"{ARXIV_API_URL}?{urlencode(params)}"

    @staticmethod
    def _id_url(value: str) -> str:
        return f"{ARXIV_API_URL}?{urlencode({'id_list': value, 'max_results': '1'})}"

    async def _request(self, url: str, *, ctx: OperationContext) -> bytes:
        if ctx.deadline_exceeded(now=self._clock()):
            raise ProviderError(
                make_failure(
                    code="ARXIV_REQUEST_DEADLINE_EXCEEDED",
                    category="TIMEOUT",
                    message="The arXiv request deadline elapsed before retrieval.",
                    retryable=True,
                    ctx=ctx,
                )
            )
        try:
            return await asyncio.to_thread(self._transport, url, self._timeout_seconds)
        except (ET.ParseError, OSError, TimeoutError, UnicodeError, ValueError):
            raise ProviderError(
                make_failure(
                    code="ARXIV_PROVIDER_REQUEST_FAILED",
                    category="PROVIDER",
                    message="The bounded arXiv metadata request failed.",
                    retryable=True,
                    ctx=ctx,
                )
            ) from None

    @staticmethod
    def _text(parent: ET.Element, path: str, *, required: bool = False) -> str | None:
        element = parent.find(path)
        value = None if element is None or element.text is None else _normalize_space(element.text)
        if required and not value:
            raise ValueError(f"arXiv Atom field {path} is required")
        return value or None

    def _entry_record(self, entry: ET.Element, retrieved_at: datetime) -> RawPaperRecord:
        entry_url = self._text(entry, f"{{{_ATOM}}}id", required=True)
        assert entry_url is not None
        arxiv_id = entry_url.rsplit("/", maxsplit=1)[-1]
        title = self._text(entry, f"{{{_ATOM}}}title", required=True)
        abstract = self._text(entry, f"{{{_ATOM}}}summary", required=True)
        published = self._text(entry, f"{{{_ATOM}}}published", required=True)
        updated = self._text(entry, f"{{{_ATOM}}}updated", required=True)
        assert title is not None and abstract is not None
        assert published is not None and updated is not None

        authors = [
            _normalize_space(name.text)
            for author in entry.findall(f"{{{_ATOM}}}author")
            if (name := author.find(f"{{{_ATOM}}}name")) is not None and name.text
        ]
        categories = [
            term
            for category in entry.findall(f"{{{_ATOM}}}category")
            if (term := category.attrib.get("term"))
        ]
        doi = self._text(entry, f"{{{_ARXIV}}}doi")
        comment = self._text(entry, f"{{{_ARXIV}}}comment")
        pdf_url: str | None = None
        link_text: list[str] = []
        for link in entry.findall(f"{{{_ATOM}}}link"):
            href = link.attrib.get("href")
            if not href:
                continue
            link_text.append(href)
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = href

        evidence_text = " ".join([title, abstract, comment or "", *link_text])
        code_urls = list(
            dict.fromkeys(
                match.rstrip(".,;:!?)]}") for match in _CODE_URL_RE.findall(evidence_text)
            )
        )
        external_ids = [ExternalPaperId(schema_version="1", provider_name="arxiv", value=arxiv_id)]
        if doi is not None:
            external_ids.append(ExternalPaperId(schema_version="1", provider_name="doi", value=doi))

        return RawPaperRecord(
            schema_version="1",
            provider_name=self.provider_name,
            provider_record_id=arxiv_id,
            external_ids=external_ids,
            raw_fields=cast(
                JsonObject,
                {
                    "arxiv_id": arxiv_id,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": categories,
                    "published_at": published,
                    "updated_at": updated,
                    "entry_url": entry_url,
                    "pdf_url": pdf_url,
                    "comment": comment,
                    "code_urls": code_urls,
                },
            ),
            retrieved_at=retrieved_at,
        )

    def _parse_page(
        self,
        payload: bytes,
        *,
        start: int,
        page_size: int,
        request_url: str,
    ) -> PaperSearchPage:
        try:
            root = ET.fromstring(payload)
            retrieved_at = self._clock().astimezone(UTC)
            records = [
                self._entry_record(entry, retrieved_at)
                for entry in root.findall(f"{{{_ATOM}}}entry")
            ]
            total_text = self._text(root, f"{{{_OPENSEARCH}}}totalResults")
            total = len(records) if total_text is None else int(total_text)
        except (AssertionError, ET.ParseError, TypeError, ValueError):
            raise ValueError("arXiv returned invalid Atom metadata") from None
        next_start = start + len(records)
        next_cursor = str(next_start) if records and next_start < total else None
        request_hash = sha256(request_url.encode("utf-8")).hexdigest()[:16]
        return PaperSearchPage(
            schema_version="1",
            provider_name=self.provider_name,
            records=records,
            next_cursor=next_cursor,
            provider_request_id=f"arxiv-{request_hash}",
            retrieved_at=retrieved_at,
        )

    async def search(
        self,
        query: PaperQuery,
        *,
        cursor: str | None,
        ctx: OperationContext,
    ) -> PaperSearchPage:
        """Return one bounded Atom page using an opaque decimal start cursor."""
        try:
            start = 0 if cursor is None else int(cursor)
        except ValueError:
            raise ProviderError(
                make_failure(
                    code="ARXIV_CURSOR_INVALID",
                    category="INPUT",
                    message="The arXiv continuation cursor is invalid.",
                    retryable=False,
                    ctx=ctx,
                )
            ) from None
        if start < 0:
            raise ProviderError(
                make_failure(
                    code="ARXIV_CURSOR_INVALID",
                    category="INPUT",
                    message="The arXiv continuation cursor is invalid.",
                    retryable=False,
                    ctx=ctx,
                )
            )
        url = self._search_url(query, start)
        payload = await self._request(url, ctx=ctx)
        try:
            return self._parse_page(
                payload,
                start=start,
                page_size=query.page_size,
                request_url=url,
            )
        except ValueError:
            raise ProviderError(
                make_failure(
                    code="ARXIV_PROVIDER_RESPONSE_INVALID",
                    category="PROVIDER_SCHEMA",
                    message="The arXiv metadata response failed validation.",
                    retryable=True,
                    ctx=ctx,
                )
            ) from None

    async def get_by_external_id(
        self,
        external_id: ExternalPaperId,
        *,
        ctx: OperationContext,
    ) -> RawPaperRecord | None:
        """Look up one arXiv identifier without accepting an arbitrary URL."""
        if external_id.provider_name.casefold() != "arxiv":
            return None
        url = self._id_url(external_id.value)
        payload = await self._request(url, ctx=ctx)
        try:
            page = self._parse_page(payload, start=0, page_size=1, request_url=url)
        except ValueError:
            raise ProviderError(
                make_failure(
                    code="ARXIV_PROVIDER_RESPONSE_INVALID",
                    category="PROVIDER_SCHEMA",
                    message="The arXiv metadata response failed validation.",
                    retryable=True,
                    ctx=ctx,
                )
            ) from None
        return page.records[0] if page.records else None


__all__ = ["ArxivPaperProvider"]

"""Result types for :meth:`WorksQuery.collect`

A collect run pulls works for many source IDs (members, funders, journals,
prefixes, or types) in input order, deep-paging each one with a cursor.
Every source gets its own :class:`SourceResult`, including sources that
returned zero works or failed on the first page.
"""

from typing import Any

#: every page available was retrieved (a source with zero works is complete)
STATUS_COMPLETE = "complete"
#: paging stopped because ``cursor_max`` was reached while more works existed
STATUS_CURSOR_MAX = "cursor_max"
#: a request failed; any pages retrieved before the failure are retained
STATUS_FAILED = "failed"

STATUSES = (STATUS_COMPLETE, STATUS_CURSOR_MAX, STATUS_FAILED)


class SourceResult:
    """Outcome of collecting works for one source ID.

    :ivar source_index: zero-based position of this ID in the collect input
    :ivar endpoint: source route, e.g. ``"members"``
    :ivar source_id: the ID as given in the collect input
    :ivar url: URL of the first (page 1) request
    :ivar pages: raw Crossref response envelopes retrieved successfully, in
        page order; empty only when the first page already failed. A source
        with zero works still keeps its one successful (empty-items) page
    :ivar status: one of :data:`STATUSES`
    :ivar failed_page: 1-based page number that failed, or ``None``
    :ivar failed_url: URL of the request that failed, or ``None``
    :ivar error_type: category of the failure (exception class name, or
        ``"HTTPStatusError"`` for a ``warn=True`` HTTP failure), or ``None``
    :ivar error_message: detail message for the failure, or ``None``
    """

    def __init__(
        self,
        source_index: int,
        endpoint: str,
        source_id: Any,
        url: str,
    ) -> None:
        self.source_index = source_index
        self.endpoint = endpoint
        self.source_id = source_id
        self.url = url
        self.pages: list[dict[str, Any]] = []
        self.status: str | None = None
        self.failed_page: int | None = None
        self.failed_url: str | None = None
        self.error_type: str | None = None
        self.error_message: str | None = None

    @property
    def works(self) -> list[dict[str, Any]]:
        """All work records retrieved for this source, in page/item order."""
        return [item for page in self.pages for item in page["message"]["items"]]

    @property
    def count(self) -> int:
        """Number of work records retrieved for this source."""
        return sum(len(page["message"]["items"]) for page in self.pages)

    @property
    def ok(self) -> bool:
        """``True`` when the source finished without failure."""
        return self.status == STATUS_COMPLETE

    @property
    def failed(self) -> bool:
        """``True`` when a request failed for this source."""
        return self.status == STATUS_FAILED

    @property
    def capped(self) -> bool:
        """``True`` when paging stopped at ``cursor_max``."""
        return self.status == STATUS_CURSOR_MAX

    def provenance(self) -> list[dict[str, Any]]:
        """One provenance dict per work in :attr:`works`, in the same order."""
        records = []
        for page_number, page in enumerate(self.pages, start=1):
            for item_index, _item in enumerate(page["message"]["items"]):
                records.append(
                    {
                        "source_index": self.source_index,
                        "endpoint": self.endpoint,
                        "source_id": self.source_id,
                        "page": page_number,
                        "item_index": item_index,
                    }
                )
        return records

    def __repr__(self) -> str:
        return (
            f"SourceResult(endpoint={self.endpoint!r}, source_id={self.source_id!r}, "
            f"source_index={self.source_index}, status={self.status!r}, "
            f"works={self.count})"
        )


class CollectResult:
    """The result of a :meth:`WorksQuery.collect` run.

    Iterating yields the :class:`SourceResult` objects in input order, so a
    source that appears twice in the input appears twice as two independent
    results. Works are never merged by DOI across sources.
    """

    def __init__(self, endpoint: str, sources: list[SourceResult]) -> None:
        self.endpoint = endpoint
        self.sources = sources

    @property
    def works(self) -> list[dict[str, Any]]:
        """All works across sources, ordered by input position, page, item."""
        return [work for source in self.sources for work in source.works]

    @property
    def provenance(self) -> list[dict[str, Any]]:
        """Provenance per work in :attr:`works`, in the same order."""
        return [record for source in self.sources for record in source.provenance()]

    @property
    def statuses(self) -> list[str]:
        """Status of every source, in input order."""
        return [source.status for source in self.sources]

    @property
    def complete(self) -> list[SourceResult]:
        """Sources that finished completely (including empty ones)."""
        return [s for s in self.sources if s.status == STATUS_COMPLETE]

    @property
    def capped(self) -> list[SourceResult]:
        """Sources that stopped because ``cursor_max`` was reached."""
        return [s for s in self.sources if s.status == STATUS_CURSOR_MAX]

    @property
    def failed(self) -> list[SourceResult]:
        """Sources whose requests failed (successful pages are retained)."""
        return [s for s in self.sources if s.status == STATUS_FAILED]

    @property
    def ok(self) -> bool:
        """``True`` only when no source failed or hit the cursor cap."""
        return all(s.status == STATUS_COMPLETE for s in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self):
        return iter(self.sources)

    def __getitem__(self, index: int) -> SourceResult:
        return self.sources[index]

    def __repr__(self) -> str:
        counts = {status: self.statuses.count(status) for status in STATUSES}
        return (
            f"CollectResult(endpoint={self.endpoint!r}, sources={len(self.sources)}, "
            f"complete={counts[STATUS_COMPLETE]}, cursor_max={counts[STATUS_CURSOR_MAX]}, "
            f"failed={counts[STATUS_FAILED]}, works={len(self.works)})"
        )

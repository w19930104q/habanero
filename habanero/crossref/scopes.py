"""Composable multi-scope collection views for :class:`WorksQuery`.

A :class:`ScopeCollection` runs several independent :class:`WorksQuery`
branches (e.g. one journal, one member and one funder scope) and merges
their work records into one literature list while keeping track of which
scope(s) every record was found in.

Merging rules:

* branches are processed in construction order, and within a branch items
  are processed by page (0-based) and then by their index in the page
* DOIs are compared case-insensitively; the first record seen for a DOI
  supplies the primary metadata, later hits only add provenance
* when a later hit differs from the primary record on a top-level field,
  the differing value is kept verbatim (arrays stay whole, they are never
  split or element-wise compared) together with its source
* records without a DOI are never merged with anything and are kept
  independently, in processing order
* empty branches are legal; a branch that returns zero items simply
  contributes nothing
* a branch that raises is reported explicitly in
  :attr:`MergedWorks.failures` (when ``on_error="collect"``) instead of
  being silently turned into a zero-hit branch
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .worksquery import WorksQuery


class _Missing:
    """Sentinel marking a field absent from another version of a record."""

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


#: Passed as :attr:`FieldVariant.value` when the other version of a record
#: does not include the field at all (as opposed to including it as ``None``).
MISSING = _Missing()


@dataclass(frozen=True)
class Source:
    """Where one occurrence of a work record came from.

    :param endpoint: the Crossref endpoint the branch queried, e.g.
        ``"works"``, ``"journals"``, ``"members"`` or ``"funders"``
    :param scope_id: the endpoint id the branch was scoped to (e.g. an
        ISSN, member id or funder id), or ``None`` for the plain
        ``/works`` endpoint
    :param page: 0-based page number within the branch's response
    :param index: 0-based index of the item within that page
    :param branch: label of the branch in the :class:`ScopeCollection`
    """

    endpoint: str
    scope_id: Any
    page: int
    index: int
    branch: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "scope_id": self.scope_id,
            "page": self.page,
            "index": self.index,
            "branch": self.branch,
        }


@dataclass(frozen=True)
class FieldVariant:
    """A top-level field on which two versions of a DOI disagree.

    :param field: name of the top-level field
    :param value: the other version's value, kept verbatim. Arrays are kept
        whole; :data:`MISSING` means the other version lacks the field
    :param source: the :class:`Source` of the other version
    """

    field: str
    value: Any
    source: Source


@dataclass(frozen=True)
class BranchFailure:
    """A branch whose query could not be completed.

    :param branch: label of the failing branch
    :param endpoint: the endpoint it queried
    :param ids: the ids it was constructed with (before normalization)
    :param error_type: type name of the exception raised
    :param message: message of the exception raised
    """

    branch: str
    endpoint: str
    ids: Any
    error_type: str
    message: str


@dataclass
class MergedWork:
    """One merged work record.

    :param record: the primary metadata, taken verbatim from the first
        occurrence of the DOI (case-insensitive). Never mutated by merging
    :param sources: every occurrence of the record, in processing order
    :param variants: top-level fields on which later occurrences differed
        from the primary record, each with the original value and source
    """

    record: dict[str, Any]
    sources: list[Source] = field(default_factory=list)
    variants: list[FieldVariant] = field(default_factory=list)

    @property
    def doi(self) -> str | None:
        """The DOI as recorded on the primary record, or ``None``."""
        return self.record.get("DOI")

    @property
    def scopes(self) -> list[str]:
        """Labels of the branches the record was found in, first hit first."""
        seen: list[str] = []
        for source in self.sources:
            if source.branch not in seen:
                seen.append(source.branch)
        return seen

    def __repr__(self) -> str:
        doi = self.doi
        head = f"DOI={doi!r}" if doi is not None else "no DOI"
        return (
            f"<MergedWork {head}, sources={len(self.sources)}, "
            f"variants={len(self.variants)}>"
        )


@dataclass
class MergedWorks:
    """The merged literature list produced by :meth:`ScopeCollection.execute`.

    :param works: merged records in first-seen order; records without a DOI
        are kept independently
    :param failures: branches that could not be completed; empty when every
        branch succeeded
    :param branches: the branch labels, in construction order
    """

    works: list[MergedWork] = field(default_factory=list)
    failures: list[BranchFailure] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[MergedWork]:
        return iter(self.works)

    def __len__(self) -> int:
        return len(self.works)

    def __getitem__(self, key: int | slice) -> MergedWork | list[MergedWork]:
        return self.works[key]

    @property
    def scopes(self) -> list[str]:
        """Labels of the branches that contributed at least one record."""
        seen: list[str] = []
        for merged in self.works:
            for label in merged.scopes:
                if label not in seen:
                    seen.append(label)
        return seen

    @property
    def ok(self) -> bool:
        """``True`` when no branch failed."""
        return not self.failures

    def __repr__(self) -> str:
        return (
            f"<MergedWorks works={len(self.works)}, "
            f"failures={len(self.failures)}, branches={self.branches}>"
        )


def _normalize_ids(ids: Any) -> list[Any]:
    """Mirror the id handling in :func:`habanero.request.request`."""
    if ids is None:
        return []
    if isinstance(ids, str):
        return ids.split()
    if isinstance(ids, int) and not isinstance(ids, bool):
        return [ids]
    return list(ids)


def _iter_items(
    result: Any, scope_ids: list[Any]
) -> Iterable[tuple[Any, int, int, dict[str, Any]]]:
    """Yield ``(scope_id, page, index, item)`` tuples from a raw result.

    A branch without ids returns a dict or (with cursor paging) a list of
    page dicts. A branch with several ids returns a list with one entry per
    id, each itself a dict or a list of page dicts, hence the nesting.
    """
    pages: list[tuple[Any, int, Any]]
    if scope_ids and len(scope_ids) > 1:
        # a multi-id branch returns one entry per id, each itself a dict or
        # (with cursor paging) a list of page dicts
        if not isinstance(result, list):
            result = [result]
        pages = []
        for scope_id, entry in zip(scope_ids, result):
            if entry is None:
                # warn=True request failure inside a multi-id branch
                continue
            entry_pages = entry if isinstance(entry, list) else [entry]
            # page numbers are per scope id: each id's own walk starts at 0
            pages.extend(
                (scope_id, page_number, page)
                for page_number, page in enumerate(entry_pages)
            )
    elif scope_ids:
        # single id: the whole result (dict or list of pages) belongs to it
        entry_pages = result if isinstance(result, list) else [result]
        pages = [
            (scope_ids[0], page_number, page)
            for page_number, page in enumerate(entry_pages)
        ]
    else:
        entry_pages = result if isinstance(result, list) else [result]
        pages = [
            (None, page_number, page) for page_number, page in enumerate(entry_pages)
        ]

    for scope_id, page_number, page in pages:
        items = page["message"]["items"]
        for index, item in enumerate(items):
            yield scope_id, page_number, index, item


class ScopeCollection:
    """A composable, merged view over several :class:`WorksQuery` branches.

    Each branch keeps its own query conditions and its own cursor; branches
    never share parameters or results. Construct from a mapping of labels
    to queries, or an iterable of ``(label, query)`` pairs::

        journal = WorksQuery(cr).journals("2167-8359").limit(100).cursor("*")
        member = WorksQuery(cr).members(98).limit(100).cursor("*")
        funder = WorksQuery(cr).funders("10.13039/100000001").limit(100)

        collection = ScopeCollection(
            {"journal": journal, "member": member, "funder": funder}
        )
        merged = collection.execute()
        for work in merged:
            print(work.doi, work.scopes)

    :param branches: label/query pairs. Labels must be unique, non-empty
        strings; at least one branch is required; every query must be a
        :class:`WorksQuery`
    :param on_error: ``"collect"`` (the default) records a failing branch
        in :attr:`MergedWorks.failures` and keeps going; ``"raise"`` lets
        the exception propagate
    """

    def __init__(
        self,
        branches: Mapping[str, WorksQuery] | Iterable[tuple[str, WorksQuery]],
        on_error: str = "collect",
    ) -> None:
        if isinstance(branches, Mapping):
            pairs = list(branches.items())
        else:
            pairs = list(branches)
        if not pairs:
            raise ValueError("ScopeCollection requires at least one branch")
        labels: list[str] = []
        normalized: list[tuple[str, WorksQuery]] = []
        for label, query in pairs:
            if not isinstance(label, str) or not label:
                raise TypeError("branch labels must be non-empty strings")
            if not isinstance(query, WorksQuery):
                raise TypeError(
                    f"branch {label!r} must be a WorksQuery, got {type(query).__name__}"
                )
            if label in labels:
                raise ValueError(f"duplicate branch label: {label!r}")
            labels.append(label)
            normalized.append((label, query))
        if on_error not in ("collect", "raise"):
            raise ValueError("on_error must be 'collect' or 'raise'")
        self._branches: dict[str, WorksQuery] = dict(normalized)
        self._order: list[str] = labels
        self._on_error = on_error
        self._result: MergedWorks | None = None

    @property
    def branches(self) -> dict[str, WorksQuery]:
        """The branch queries keyed by label (insertion ordered)."""
        return dict(self._branches)

    @property
    def on_error(self) -> str:
        return self._on_error

    def __iter__(self) -> Iterator[MergedWork]:
        return iter(self.execute().works)

    def __len__(self) -> int:
        return len(self._branches)

    def __repr__(self) -> str:
        return f"<ScopeCollection branches={self._order}, on_error={self._on_error!r}>"

    def execute(self) -> MergedWorks:
        """Run every branch and merge their records.

        Branches are processed in construction order; within a branch,
        records are processed by page and then by page index. The result is
        cached, so calling this again (or iterating the collection) does
        not fire another request. Branch queries themselves are never
        mutated — each branch's own ``execute()`` cache and parameters are
        left untouched.
        """
        if self._result is not None:
            return self._result

        works: list[MergedWork] = []
        failures: list[BranchFailure] = []
        # DOI (lower-cased) -> position in `works`
        by_doi: dict[str, int] = {}

        for label in self._order:
            query = self._branches[label]
            try:
                # _call_method bypasses the WorksQuery result cache, so the
                # branch instance keeps its own state and can be reused in
                # another collection without inheriting anything here
                result = query._call_method()
                if result is None:
                    # warn=True turns a failed request into None
                    raise TypeError(
                        "the branch returned no response (the request may "
                        "have failed with warn=True)"
                    )
                scope_ids = _normalize_ids(query._ids)
                items = list(_iter_items(result, scope_ids))
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                failures.append(self._make_failure(label, query, exc))
                continue

            for scope_id, page_number, index, item in items:
                source = Source(
                    endpoint=query._endpoint,
                    scope_id=scope_id,
                    page=page_number,
                    index=index,
                    branch=label,
                )
                doi = item.get("DOI") if isinstance(item, dict) else None
                key = doi.lower() if isinstance(doi, str) else None
                if key is None or key not in by_doi:
                    if key is not None:
                        by_doi[key] = len(works)
                    works.append(MergedWork(record=item, sources=[source]))
                else:
                    self._absorb(works[by_doi[key]], item, source)

        self._result = MergedWorks(
            works=works, failures=failures, branches=list(self._order)
        )
        return self._result

    @staticmethod
    def _make_failure(label: str, query: WorksQuery, exc: Exception) -> BranchFailure:
        return BranchFailure(
            branch=label,
            endpoint=query._endpoint,
            ids=query._ids,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    @staticmethod
    def _absorb(merged: MergedWork, other: dict[str, Any], source: Source) -> None:
        """Merge a later occurrence of a known DOI into an existing record."""
        merged.sources.append(source)
        primary = merged.record
        fields = set(primary) | set(other)
        for name in fields:
            if name not in other:
                merged.variants.append(
                    FieldVariant(field=name, value=MISSING, source=source)
                )
            elif name not in primary:
                merged.variants.append(
                    FieldVariant(field=name, value=other[name], source=source)
                )
            elif primary[name] != other[name]:
                # arrays are compared and kept whole, never split
                merged.variants.append(
                    FieldVariant(field=name, value=other[name], source=source)
                )

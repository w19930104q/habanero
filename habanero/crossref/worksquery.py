import copy
import math
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx2

from ..request_class import Request
from .collect import (
    STATUS_COMPLETE,
    STATUS_CURSOR_MAX,
    STATUS_FAILED,
    CollectResult,
    SourceResult,
)
from .crossref import Crossref

#: source routes that support ``/<endpoint>/<id>/works`` and thus collect()
COLLECT_ENDPOINTS = ("members", "funders", "journals", "prefixes", "types")


class WorksQuery(Iterable[dict[str, Any]]):
    """
    Query builder for the Crossref API's works endpoint.

    Iterating over an instance (``for item in q``) yields individual work
    records as ``dict[str, Any]``, across all pages when :meth:`cursor` is
    used.  Calling :meth:`execute` returns the raw Crossref API response, the
    same thing the wrapped :class:`~habanero.Crossref` method returns: a
    ``dict[str, Any]`` (the full envelope including ``message``, ``status``,
    etc.), or a ``list`` of such dicts, one per page, when :meth:`cursor`
    paged through several pages.  Calling :meth:`count` returns an ``int``.
    Calling :meth:`url` returns a ``str``.

    All builder methods (:meth:`query`, :meth:`filters`, :meth:`sort`,
    :meth:`order`, :meth:`select`, :meth:`facet`, :meth:`limit`,
    :meth:`cursor`) return a new :class:`WorksQuery` instance — the original
    is never mutated.

    :rtype: :class:`WorksQuery`

    Usage::

      from habanero import Crossref, WorksQuery
      cr = Crossref()
      q = WorksQuery(cr)

      # chain methods, nothing fires yet
      (
        q.query("climate change")
          .query(author="Hansen")
          .query(publisher_name="plos")
          .filters(from_pub_date="2010", has_funder="true")
          .sort("published")
          .order("desc")
          .select("DOI", "title", "author", "published")
          .limit(50)
      )

      # inspect before fetching
      print(q)         # WorksQuery({...params...})
      print(q.url)     # https://api.crossref.org/works?query=...

      # get count without pulling records
      print(q.count()) # e.g. 12483

      # pull records — fires the request here
      for item in q:
          print(item["DOI"], item.get("title"))

      # or execute manually
      q.execute()

      # instances are immutable, so each call returns a new instance
      # so you can chain calls without modifying the original instance
      # compare the two modifications of the `base` query
      base = WorksQuery(cr).query("zika").filters(from_pub_date="2020")
      base.sort("published").order("asc")
      base.sort("published").order("desc")
    """

    def __init__(self, cr: Crossref | None = None):
        self._cr = cr or Crossref()
        self._params = {}
        self._endpoint = "works"
        self._ids = None
        self._result = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        data = self.execute()
        pages = data if isinstance(data, list) else [data]
        return (item for page in pages for item in page["message"]["items"])

    def __repr__(self):
        ids_part = f", ids={self._ids!r}" if self._ids else ""
        return (
            f"WorksQuery(endpoint={self._endpoint!r}{ids_part}, params={self._params})"
        )

    def _clone(self, **updates):
        clone = copy.copy(self)
        clone._params = copy.deepcopy({**self._params, **updates})
        clone._result = None  # fresh clone shouldn't inherit cached results
        return clone

    def journals(self, ids=None) -> "WorksQuery":
        clone = self._clone()
        clone._endpoint = "journals"
        clone._ids = ids
        return clone

    def members(self, ids=None) -> "WorksQuery":
        clone = self._clone()
        clone._endpoint = "members"
        clone._ids = ids
        return clone

    def funders(self, ids=None) -> "WorksQuery":
        clone = self._clone()
        clone._endpoint = "funders"
        clone._ids = ids
        return clone

    def prefixes(self, ids=None) -> "WorksQuery":
        clone = self._clone()
        clone._endpoint = "prefixes"
        clone._ids = ids
        return clone

    def types(self, ids=None) -> "WorksQuery":
        clone = self._clone()
        clone._endpoint = "types"
        clone._ids = ids
        return clone

    def query(self, q: str | None = None, **kwargs) -> "WorksQuery":
        updates = {}
        if q:
            updates["query"] = q
        for k, v in kwargs.items():
            updates[f"query_{k}"] = v
        return self._clone(**updates)

    def filters(self, **kwargs) -> "WorksQuery":
        new_filters = {**self._params.get("filters", {}), **kwargs}
        return self._clone(filters=new_filters)

    def sort(self, field: str) -> "WorksQuery":
        return self._clone(sort=field)

    def order(self, direction: str) -> "WorksQuery":
        return self._clone(order=direction)

    def select(self, *fields: str) -> "WorksQuery":
        return self._clone(select=list(fields))

    def facet(self, name: str, count: int) -> "WorksQuery":
        existing = self._params.get("facet")
        new_facet = f"{existing},{name}:{count}" if existing else f"{name}:{count}"
        return self._clone(facet=new_facet)

    def limit(self, n: int) -> "WorksQuery":
        return self._clone(limit=n)

    def cursor(self, value: str = "*", cursor_max: int | None = 5000) -> "WorksQuery":
        return self._clone(cursor=value, cursor_max=cursor_max)

    @property
    def url(self) -> str:
        """The URL that :meth:`execute` requests.

        Built with the same code that builds the real request, so it matches
        what is sent. With multiple ``ids`` several requests are made; the URL
        for the first one is returned.
        """
        ids = self._ids
        if isinstance(ids, str):
            ids = ids.split()
        elif isinstance(ids, int):
            ids = [ids]

        if self._endpoint == "works":
            path = "/works"
        elif ids:
            path = f"/{self._endpoint}/{ids[0]}/works"
        else:
            path = f"/{self._endpoint}"

        params = copy.deepcopy(self._params)
        req = Request(
            self._cr.mailto,
            self._cr.ua_string,
            self._cr.timeout,
            self._cr.base_url,
            path,
            params.pop("query", None),
            params.pop("filters", None),
            params.pop("offset", None),
            params.pop("limit", None),
            params.pop("sample", None),
            params.pop("sort", None),
            params.pop("order", None),
            params.pop("facet", None),
            params.pop("select", None),
            params.pop("cursor", None),
            params.pop("cursor_max", 5000),
            **params,
        )
        return str(httpx2.Request("GET", req._url(), params=req.payload()).url)

    def count(self) -> int:
        # cursor/cursor_max are dropped too: count() needs a single request,
        # and paging with limit=0 would only ever return empty pages
        params = {
            k: v
            for k, v in self._params.items()
            if k not in ("limit", "cursor", "cursor_max")
        }
        if self._endpoint == "works":
            result = self._cr.works(**params, limit=0)
        else:
            method = getattr(self._cr, self._endpoint)
            result = method(ids=self._ids, works=True, **params, limit=0)
        if not isinstance(result, dict):
            raise TypeError(
                f"count() expected a single response dict, got {type(result).__name__}"
            )
        return result["message"]["total-results"]

    def _call_method(self, **extra_params) -> dict[str, Any] | list[dict[str, Any]]:
        params = copy.deepcopy({**self._params, **extra_params})
        if self._endpoint == "works":
            return self._cr.works(**params)
        method = getattr(self._cr, self._endpoint)
        return method(ids=self._ids, works=True, **params)

    def execute(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Run the query and return the raw response.

        Returns a ``dict``, or a ``list`` of ``dict`` (one per page) when
        :meth:`cursor` paged through several pages.
        """
        if self._result is None:
            result = self._call_method()
            if not isinstance(result, (dict, list)):
                raise TypeError(
                    "no usable response was returned (the request may have "
                    f"failed); got {type(result).__name__}"
                )
            self._result = result
        return self._result

    # --- collect -----------------------------------------------------------

    def collect(self, warn: bool = False) -> CollectResult:
        """Collect works for many source IDs, deep-paging each one.

        The query must be scoped to a source route with IDs — one of
        :meth:`members`, :meth:`funders`, :meth:`journals`, :meth:`prefixes`,
        or :meth:`types`, each given one or more IDs — for example::

            q = WorksQuery(cr).members(["12", "34", "12"]).limit(100)
            result = q.collect()

        Every ID is queried independently in input order: an ID that appears
        twice produces two independent source results, and works sharing a
        DOI across sources are never merged. Each source is deep-paged with a
        cursor up to the query's ``cursor_max`` (default 5000; ``None`` means
        no cap), reusing this query's filters, query terms, limit, select, and
        so on.

        Each source ends with one of three statuses:

        * ``"complete"`` — every available page was retrieved; a source with
          zero works is also complete
        * ``"cursor_max"`` — paging stopped because ``cursor_max`` was reached
          while more works existed, so the retained data is partial
        * ``"failed"`` — a request failed; pages retrieved before the failure
          are retained, together with the request URL, failed page number, and
          error category. A failure never aborts the run; the remaining
          sources are still collected.

        :param warn: when ``True`` (default ``False``), an HTTP failure on a
            page warns instead of raising and is recorded as a failed source;
            the warning behavior matches :meth:`execute`. Retry behavior is
            unchanged either way.
        :returns: a :class:`CollectResult`; pass it straight to
            :class:`WorksContainer`, which also accepts an all-empty result.
        """
        if self._endpoint not in COLLECT_ENDPOINTS:
            raise ValueError(
                "collect() requires a source query with IDs — call one of "
                f"{', '.join(COLLECT_ENDPOINTS)} with ids before collect(); "
                f"got endpoint {self._endpoint!r}"
            )
        ids = self._normalize_ids(self._ids)
        if not ids:
            raise ValueError(
                "collect() requires at least one source ID; call "
                f"{self._endpoint}() with ids before collect()"
            )

        cursor_max = self._params.get("cursor_max", 5000)
        if cursor_max is not None and not isinstance(cursor_max, int):
            raise TypeError("cursor_max must be of class int (or None for no cap)")

        sources = [
            self._collect_one(index, source_id, cursor_max, warn)
            for index, source_id in enumerate(ids)
        ]
        return CollectResult(self._endpoint, sources)

    @staticmethod
    def _normalize_ids(ids) -> list:
        if ids is None:
            return []
        if isinstance(ids, str):
            return ids.split()
        if isinstance(ids, int):
            return [ids]
        return list(ids)

    def _build_collect_request(self, source_id, cursor_max) -> Request:
        params = copy.deepcopy(self._params)
        known = [
            "query",
            "filters",
            "offset",
            "limit",
            "sample",
            "sort",
            "order",
            "facet",
            "select",
            "cursor",
            "cursor_max",
        ]
        kwargs = {k: v for k, v in params.items() if k not in known}
        return Request(
            self._cr.mailto,
            self._cr.ua_string,
            self._cr.timeout,
            self._cr.base_url + f"/{self._endpoint}/",
            "",
            params.get("query"),
            params.get("filters"),
            params.get("offset"),
            params.get("limit"),
            params.get("sample"),
            params.get("sort"),
            params.get("order"),
            params.get("facet"),
            params.get("select"),
            "*",
            cursor_max,
            None,
            False,
            client=self._cr._http,
            **kwargs,
        )

    @staticmethod
    @contextmanager
    def _warn_recorder(records: list):
        """Record warnings without swallowing them.

        ``warnings.catch_warnings(record=True)`` would hide the warning from
        outer ``pytest.warns`` (it replaces the showwarning on the
        interpreter-wide manager), so wrap ``warnings.warn`` instead; the
        original function is still called and existing warn behavior is
        unchanged.
        """
        with patch.object(warnings, "warn", side_effect=warnings.warn) as mocked:
            yield
        records.extend(str(call.args[0]) for call in mocked.call_args_list)

    def _collect_one(
        self, source_index: int, source_id, cursor_max, should_warn: bool
    ) -> SourceResult:
        # build one Request and page it manually, mirroring Request.do_request
        # / _redo_req exactly, so request shape, warn and retry behavior match
        req = self._build_collect_request(source_id, cursor_max)
        # base_url already ends with /<endpoint>/ (see _build_collect_request),
        # so path is just <id>/works, exactly as request() builds it
        req.path = f"{source_id}/works"
        result = SourceResult(source_index, self._endpoint, source_id, req._url())

        cap = math.inf if cursor_max is None else cursor_max
        payload = None
        try:
            payload = req.payload()
            warned: list[str] = []
            with self._warn_recorder(warned):
                js = req._req(payload=payload, should_warn=should_warn)
            if js is None:
                self._mark_http_failure(result, req, payload, warned)
                return result

            pages = [js]
            total = len(js["message"]["items"])
            cursor = js["message"].get("next-cursor")
            max_avail = js["message"]["total-results"]
            stopped_on_empty_page = False

            while cursor is not None and cap > total and total < max_avail:
                payload["cursor"] = cursor
                page_warned: list[str] = []
                with self._warn_recorder(page_warned):
                    out = req._req(payload=payload, should_warn=should_warn)
                if out is None:
                    result.pages = pages
                    self._mark_http_failure(result, req, payload, page_warned)
                    return result
                items = out["message"]["items"]
                if not items:
                    # an empty page can still carry a next-cursor; stop rather
                    # than loop forever (matches Request._redo_req)
                    stopped_on_empty_page = True
                    break
                cursor = out["message"].get("next-cursor")
                pages.append(out)
                total += len(items)

            result.pages = pages
            hit_cap = not stopped_on_empty_page and cursor is not None and cap <= total
            result.status = STATUS_CURSOR_MAX if hit_cap else STATUS_COMPLETE
            return result
        except Exception as e:  # noqa: BLE001 - recorded per source, run continues
            result.status = STATUS_FAILED
            result.failed_page = len(result.pages) + 1
            result.failed_url = self._request_url(req, payload)
            result.error_type = type(e).__name__
            result.error_message = str(e) or repr(e)
            return result

    @staticmethod
    def _mark_http_failure(result: SourceResult, req: Request, payload, warned):
        result.status = STATUS_FAILED
        result.failed_page = len(result.pages) + 1
        result.failed_url = str(httpx2.Request("GET", req._url(), params=payload).url)
        result.error_type = "HTTPStatusError"
        result.error_message = warned[-1] if warned else "HTTP request failed"

    @staticmethod
    def _request_url(req: Request, payload) -> str:
        try:
            return str(httpx2.Request("GET", req._url(), params=payload).url)
        except Exception:  # noqa: BLE001 - best-effort URL for the error record
            return req._url()

"""Regression tests for the composable multi-scope collection view.

Every test talks to an in-memory HTTP double (``httpx2.MockTransport``),
so no real network request is made.
"""

from unittest.mock import patch

import httpx2
import pytest

from habanero import (
    MISSING,
    Crossref,
    FieldVariant,
    RequestError,
    ScopeCollection,
    Source,
    WorksQuery,
)

# --- HTTP doubles -----------------------------------------------------------


def _envelope(items, total=None, next_cursor=None):
    # when another cursor page exists, total-results must exceed the items
    # already returned or Request._redo_req stops paging immediately
    if total is None:
        total = 10000 if next_cursor is not None else len(items)
    return {
        "status": "ok",
        "message-type": "work-list",
        "message": {
            "items": items,
            "total-results": total,
            "next-cursor": next_cursor,
        },
    }


def _routed_stub(routes):
    """A Crossref whose handler routes by URL path.

    `routes` maps a path substring to either a list of envelopes (served one
    per request, supporting cursor walks), a single envelope, or an Exception
    instance/response tuple that fails every request to that path.
    """
    requests_seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests_seen.append((request.url.path, dict(request.url.params)))
        # longest configured path wins, so "/members/98/..." does not match
        # a route keyed for "/members/9/..." first
        needle = max(
            (key for key in routes if key in request.url.path),
            key=len,
            default=None,
        )
        if needle is None:
            raise AssertionError(f"unexpected request path: {request.url.path}")
        payload = routes[needle]
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, tuple):
            status, body = payload
            return httpx2.Response(status, json=body)
        if isinstance(payload, list):
            # page index counts only requests to this route
            i = sum(1 for path, _ in requests_seen if needle in path) - 1
            assert i < len(payload), f"more requests than pages for {needle}"
            return httpx2.Response(200, json=payload[i])
        return httpx2.Response(200, json=payload)

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return Crossref(client=client, retries=0), requests_seen


def _stub(routes):
    cr, _ = _routed_stub(routes)
    return cr


# --- construction -----------------------------------------------------------


def test_scopes_requires_at_least_one_branch():
    with pytest.raises(ValueError):
        ScopeCollection({})


@pytest.mark.parametrize("bad_label", ["", None, 7])
def test_scopes_rejects_bad_labels(bad_label):
    cr = _stub({"/works": _envelope([])})
    with pytest.raises((TypeError, ValueError)):
        ScopeCollection([(bad_label, WorksQuery(cr))])


def test_scopes_rejects_non_query_branch():
    with pytest.raises(TypeError):
        ScopeCollection([("x", {"not": "a query"})])


def test_scopes_rejects_duplicate_labels():
    cr = _stub({"/works": _envelope([])})
    with pytest.raises(ValueError):
        ScopeCollection([("same", WorksQuery(cr)), ("same", WorksQuery(cr).query("z"))])


def test_scopes_rejects_bad_on_error():
    cr = _stub({"/works": _envelope([])})
    with pytest.raises(ValueError):
        ScopeCollection({"a": WorksQuery(cr)}, on_error="explode")


# --- multi-id, nested cursor paging, provenance, ordering -------------------


def test_multi_id_nested_cursor_paging_and_ordering():
    """A multi-id branch nests pages per id; pages/items flatten in order."""
    cr = _stub(
        {
            "/members/1/works": [
                _envelope([{"DOI": "10.1/a"}], next_cursor="c1"),
                _envelope([{"DOI": "10.1/b"}]),
            ],
            "/members/2/works": [
                _envelope([{"DOI": "10.2/a"}], next_cursor="c2"),
                _envelope([{"DOI": "10.2/b"}]),
            ],
        }
    )
    branch = WorksQuery(cr).members([1, 2]).limit(1).cursor("*", cursor_max=10)
    merged = WorksQuery.union({"members": branch}).execute()

    assert [w.doi for w in merged] == ["10.1/a", "10.1/b", "10.2/a", "10.2/b"]
    sources = [w.sources[0] for w in merged]
    # page numbers restart per scope id; index restarts per page
    assert [(s.scope_id, s.page, s.index, s.endpoint, s.branch) for s in sources] == [
        (1, 0, 0, "members", "members"),
        (1, 1, 0, "members", "members"),
        (2, 0, 0, "members", "members"),
        (2, 1, 0, "members", "members"),
    ]


def test_branch_then_page_then_item_processing_order_across_branches():
    cr = _stub(
        {
            "/journals/2167-8359/works": [
                _envelope([{"DOI": "10.1/j1"}, {"DOI": "10.1/j2"}]),
            ],
            "/members/98/works": [
                _envelope([{"DOI": "10.1/m1"}], next_cursor="c"),
                _envelope([{"DOI": "10.1/m2"}]),
            ],
            "/funders/100/works": _envelope([{"DOI": "10.1/f1"}]),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("2167-8359").limit(2),
            "member": WorksQuery(cr).members(98).limit(1).cursor("*", None),
            "funder": WorksQuery(cr).funders(100),
        }
    ).execute()

    assert [w.doi for w in merged] == [
        "10.1/j1",
        "10.1/j2",
        "10.1/m1",
        "10.1/m2",
        "10.1/f1",
    ]
    assert merged.branches == ["journal", "member", "funder"]
    assert merged.scopes == ["journal", "member", "funder"]
    assert merged.ok


# --- overlapping DOIs, case-insensitive merge -------------------------------


def test_overlapping_doi_merge_is_case_insensitive_and_first_seen_wins():
    cr = _stub(
        {
            "/journals/x/works": _envelope(
                [{"DOI": "10.1/ABC", "title": ["first"], "score": 1}]
            ),
            "/members/9/works": _envelope(
                [{"DOI": "10.1/abc", "title": ["second"], "extra": 42}]
            ),
            "/funders/7/works": _envelope(
                [{"DOI": "10.1/AbC", "title": ["first"], "score": 1}]
            ),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
            "funder": WorksQuery(cr).funders(7),
        }
    ).execute()

    assert len(merged) == 1
    work = merged[0]
    # first-seen metadata is primary, including original DOI casing
    assert work.record == {"DOI": "10.1/ABC", "title": ["first"], "score": 1}
    assert work.doi == "10.1/ABC"
    assert work.scopes == ["journal", "member", "funder"]
    assert [s.branch for s in work.sources] == ["journal", "member", "funder"]
    # DOI casing difference itself is a recorded variant, as is the
    # different title, the missing score and the added extra field
    variant_fields = {v.field for v in work.variants}
    assert variant_fields == {"DOI", "title", "score", "extra"}
    by_field = {v.field: v for v in work.variants if v.field in ("score", "extra")}
    # every later DOI spelling is captured, in processing order
    doi_values = [v.value for v in work.variants if v.field == "DOI"]
    assert doi_values == ["10.1/abc", "10.1/AbC"]
    assert by_field["score"].value is MISSING
    assert by_field["extra"].value == 42
    # every variant carries the source of the other version
    assert by_field["extra"].source.branch == "member"
    assert all(isinstance(v.source, Source) for v in work.variants)


def test_field_conflict_arrays_are_kept_whole_not_split():
    authors_a = [{"family": "A"}, {"family": "B"}]
    authors_b = [{"family": "A"}, {"family": "B"}, {"family": "C"}]
    cr = _stub(
        {
            "/journals/x/works": _envelope([{"DOI": "10.1/d", "author": authors_a}]),
            "/members/9/works": _envelope([{"DOI": "10.1/D", "author": authors_b}]),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
        }
    ).execute()

    work = merged[0]
    # primary record untouched; the array was compared and kept whole,
    # never split element-wise (one author variant, not three contributor diffs)
    assert work.record["author"] == authors_a
    author_variants = [v for v in work.variants if v.field == "author"]
    assert len(author_variants) == 1
    assert author_variants[0].value == authors_b
    assert isinstance(author_variants[0], FieldVariant)


def test_records_without_doi_are_kept_independently_and_in_order():
    cr = _stub(
        {
            "/journals/x/works": _envelope(
                [
                    {"DOI": "10.1/keep", "title": ["t"]},
                    {"title": ["no doi one"]},
                ]
            ),
            "/members/9/works": _envelope(
                [
                    {"title": ["no doi two"]},
                    {"DOI": "10.1/KEEP", "title": ["t"]},
                ]
            ),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
        }
    ).execute()

    assert len(merged) == 3
    assert merged[0].doi == "10.1/keep"
    assert merged[0].scopes == ["journal", "member"]
    assert merged[1].doi is None
    assert merged[1].record == {"title": ["no doi one"]}
    assert merged[2].doi is None
    assert merged[2].record == {"title": ["no doi two"]}
    # no-DOI records carry full provenance too
    assert merged[1].sources[0].branch == "journal"
    assert merged[2].sources[0].branch == "member"


# --- empty branches ---------------------------------------------------------


def test_empty_branches_are_legal_and_distinct_from_failures():
    cr = _stub(
        {
            "/journals/x/works": _envelope([], total=0),
            "/members/9/works": _envelope([{"DOI": "10.1/one"}]),
            "/funders/7/works": _envelope([], total=0),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
            "funder": WorksQuery(cr).funders(7),
        }
    ).execute()

    assert [w.doi for w in merged] == ["10.1/one"]
    assert merged.failures == []
    assert merged.ok
    assert merged.branches == ["journal", "member", "funder"]
    # an empty branch does not show up as a contributing scope
    assert merged.scopes == ["member"]


def test_all_empty_branches_return_empty_result_not_an_error():
    cr = _stub(
        {
            "/journals/x/works": _envelope([]),
            "/members/9/works": _envelope([]),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
        }
    ).execute()

    assert len(merged) == 0
    assert merged.ok
    assert merged.failures == []


# --- partial failure --------------------------------------------------------


def test_failed_branch_is_reported_not_disguised_as_zero_hits():
    cr = _stub(
        {
            "/journals/x/works": _envelope([{"DOI": "10.1/ok"}]),
            "/members/9/works": (
                500,
                {"status": "error", "message": [{"message": "boom"}]},
            ),
            "/funders/7/works": _envelope([{"DOI": "10.1/funder-ok"}]),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
            "funder": WorksQuery(cr).funders(7),
        }
    ).execute()

    assert not merged.ok
    assert len(merged.failures) == 1
    failure = merged.failures[0]
    assert failure.branch == "member"
    assert failure.endpoint == "members"
    assert failure.ids == 9
    assert failure.error_type == "RequestError"
    assert "boom" in failure.message
    # surviving branches still merged, order preserved
    assert [w.doi for w in merged] == ["10.1/ok", "10.1/funder-ok"]


def test_failed_branch_with_transport_exception_is_reported():
    cr = _stub(
        {
            "/journals/x/works": _envelope([{"DOI": "10.1/ok"}]),
            "/members/9/works": httpx2.ConnectError("connection refused"),
        }
    )
    merged = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
        }
    ).execute()

    assert len(merged.failures) == 1
    # Request._req wraps httpx2.HTTPError subclasses into RuntimeError
    assert merged.failures[0].error_type == "RuntimeError"
    assert "connection refused" in merged.failures[0].message
    assert [w.doi for w in merged] == ["10.1/ok"]


def test_on_error_raise_propagates_and_keeps_earlier_branches_out_of_cache():
    cr = _stub(
        {
            "/journals/x/works": _envelope([{"DOI": "10.1/ok"}]),
            "/members/9/works": (
                500,
                {"status": "error", "message": [{"message": "boom"}]},
            ),
        }
    )
    collection = ScopeCollection(
        {
            "journal": WorksQuery(cr).journals("x"),
            "member": WorksQuery(cr).members(9),
        },
        on_error="raise",
    )
    with pytest.raises(RequestError):
        collection.execute()


def test_warn_true_none_entry_inside_multi_id_branch_is_skipped():
    """A None entry (one failed id with warn=True) contributes nothing."""
    cr = _stub({"/works": _envelope([])})

    # drive a warn-style raw result (one good id, one failed id -> None)
    # directly through the collection
    raw = [
        _envelope([{"DOI": "10.1/a"}]),
        None,
    ]
    with patch.object(WorksQuery, "_call_method", return_value=raw):
        merged = WorksQuery.union({"members": WorksQuery(cr).members([1, 2])}).execute()

    assert [w.doi for w in merged] == ["10.1/a"]
    assert merged.ok  # a None entry contributes nothing but does not raise
    assert merged[0].sources[0].scope_id == 1


def test_branch_returning_none_is_reported_as_failure_not_zero_hits():
    """A top-level None result (warn=True on the only request) is explicit."""
    cr = _stub({"/works": _envelope([])})
    with patch.object(WorksQuery, "_call_method", return_value=None):
        merged = WorksQuery.union({"broken": WorksQuery(cr).members(9)}).execute()

    assert len(merged) == 0
    assert not merged.ok
    assert len(merged.failures) == 1
    failure = merged.failures[0]
    assert failure.branch == "broken"
    assert failure.endpoint == "members"
    assert failure.error_type == "TypeError"


# --- branch reuse / no pollution --------------------------------------------


def test_branch_queries_and_results_do_not_pollute_each_other():
    cr = _stub(
        {
            "/journals/x/works": _envelope([{"DOI": "10.1/j"}]),
            "/members/9/works": _envelope([{"DOI": "10.1/m"}]),
        }
    )
    journal = WorksQuery(cr).journals("x").filters(from_pub_date="2020")
    member = WorksQuery(cr).members(9)
    params_before = journal._params
    endpoint_before = member._endpoint

    merged = ScopeCollection({"journal": journal, "member": member}).execute()

    # source branch instances are untouched
    assert journal._params is params_before
    assert journal._params == {"filters": {"from_pub_date": "2020"}}
    assert member._endpoint == endpoint_before
    assert member._ids == 9
    # branches never inherit a cached result from the collection run
    assert journal._result is None
    assert member._result is None
    # merged records are the exact dicts returned by the stub (no copying rewrite)
    assert merged[0].record == {"DOI": "10.1/j"}
    assert merged[1].record == {"DOI": "10.1/m"}


def test_reusing_branches_in_two_collections_runs_them_independently():
    cr, seen = _routed_stub(
        {
            "/journals/x/works": _envelope([{"DOI": "10.1/j"}]),
            "/members/9/works": _envelope([{"DOI": "10.1/m"}]),
        }
    )
    journal = WorksQuery(cr).journals("x")
    member = WorksQuery(cr).members(9)
    collection_a = ScopeCollection({"journal": journal, "member": member})
    collection_b = ScopeCollection({"again-journal": journal, "again-member": member})

    first = collection_a.execute()
    second = collection_b.execute()

    # both collections fired their own requests
    journal_paths = [p for p, _ in seen if "/journals/" in p]
    member_paths = [p for p, _ in seen if "/members/" in p]
    assert len(journal_paths) == 2
    assert len(member_paths) == 2
    # the two results are independent objects; branch labels are per collection
    assert first is not second
    assert first.branches == ["journal", "member"]
    assert second.branches == ["again-journal", "again-member"]
    assert second[0].sources[0].branch == "again-journal"
    # the underlying branch still has no cached state of its own
    assert journal._result is None and member._result is None


def test_execute_is_cached_on_the_collection():
    cr, seen = _routed_stub({"/works": _envelope([{"DOI": "10.1/a"}])})
    collection = ScopeCollection({"only": WorksQuery(cr).query("x")})
    collection.execute()
    collection.execute()
    list(collection)

    assert len(seen) == 1


# --- source provenance details ----------------------------------------------


def test_source_carries_endpoint_scope_page_index_and_branch():
    cr = _stub(
        {
            "/funders/55/works": [
                _envelope([{"DOI": "10.1/p0"}, {"DOI": "10.1/p1"}], next_cursor="c"),
                _envelope([{"DOI": "10.1/p2"}]),
            ],
        }
    )
    merged = WorksQuery.union(
        {"funder": WorksQuery(cr).funders(55).limit(2).cursor("*", None)}
    ).execute()

    provenance = [
        (
            w.sources[0].endpoint,
            w.sources[0].scope_id,
            w.sources[0].page,
            w.sources[0].index,
        )
        for w in merged
    ]
    assert provenance == [
        ("funders", 55, 0, 0),
        ("funders", 55, 0, 1),
        ("funders", 55, 1, 0),
    ]
    # as_dict exposes the same information
    assert merged[0].sources[0].as_dict() == {
        "endpoint": "funders",
        "scope_id": 55,
        "page": 0,
        "index": 0,
        "branch": "funder",
    }


def test_plain_works_branch_has_none_scope_id():
    cr = _stub({"/works": _envelope([{"DOI": "10.1/a"}])})
    merged = WorksQuery.union({"works": WorksQuery(cr).query("zika")}).execute()

    source = merged[0].sources[0]
    assert source.endpoint == "works"
    assert source.scope_id is None
    assert source.page == 0 and source.index == 0


# --- backward compatibility -------------------------------------------------


def test_legacy_execute_count_url_and_iteration_unchanged():
    paged_cr, paged_seen = _routed_stub(
        {
            "/works": [
                _envelope([{"DOI": "10.1/a"}, {"DOI": "10.1/b"}], next_cursor="c"),
                _envelope([{"DOI": "10.1/c"}]),
            ],
        }
    )
    query = WorksQuery(paged_cr).query("x").limit(2).cursor("*", cursor_max=None)

    raw = query.execute()
    # legacy Crossref envelope shape is untouched
    assert isinstance(raw, list) and len(raw) == 2
    assert set(raw[0]) == {"status", "message-type", "message"}
    assert "items" in raw[0]["message"]
    assert [item["DOI"] for item in query] == ["10.1/a", "10.1/b", "10.1/c"]
    assert len(paged_seen) == 2

    # count() still makes its own single limit=0 request
    count_cr, count_seen = _routed_stub({"/works": _envelope([], total=42)})
    assert WorksQuery(count_cr).query("x").count() == 42
    assert len(count_seen) == 1
    assert "cursor" not in count_seen[0][1]
    assert count_seen[0][1]["rows"] == "0"

    # url still describes the request
    url_cr, url_seen = _routed_stub({"/works": _envelope([])})
    url_query = WorksQuery(url_cr).query("x").filters(from_pub_date="2020")
    assert isinstance(url_query.url, str)
    assert "/works" in url_query.url and "query=x" in url_query.url
    url_query.execute()
    assert url_seen[0][0] == "/works"


def test_mapping_and_pairs_construction_are_equivalent():
    routes = {
        "/journals/x/works": _envelope([{"DOI": "10.1/j"}]),
        "/members/9/works": _envelope([{"DOI": "10.1/m"}]),
    }
    cr1, cr2 = _stub(routes), _stub(routes)

    def queries(cr):
        return [
            ("journal", WorksQuery(cr).journals("x")),
            ("member", WorksQuery(cr).members(9)),
        ]

    via_mapping = ScopeCollection(dict(queries(cr1))).execute()
    via_pairs = ScopeCollection(queries(cr2)).execute()

    assert [w.doi for w in via_mapping] == [w.doi for w in via_pairs]
    assert via_mapping.branches == via_pairs.branches

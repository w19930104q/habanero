"""Offline regression tests for WorksQuery.collect and WorksContainer support.

Everything runs against httpx2.MockTransport: no network, no cassettes.
"""

import httpx2
import pytest

from habanero import (
    STATUS_COMPLETE,
    STATUS_CURSOR_MAX,
    STATUS_FAILED,
    CollectResult,
    Crossref,
    SourceResult,
    WorksContainer,
    WorksQuery,
)

# --- stub transport ---------------------------------------------------------


def _page(dois, next_cursor, total=None):
    total = total if total is not None else len(dois)
    return {
        "status": "ok",
        "message-type": "work-list",
        "message": {
            "items": [{"DOI": d} for d in dois],
            "total-results": total,
            "next-cursor": next_cursor,
        },
    }


def _ok(dois, next_cursor, total=None):
    return httpx2.Response(200, json=_page(dois, next_cursor, total))


def _bad_request():
    # 400 with a non-JSON body: Request._req warns/returns None when warn=True
    return httpx2.Response(400, text="bad request")


def _transport(route):
    """`route(path, params, call_number)` returns an httpx2.Response.

    Requests and their URLs are recorded so tests can assert on them.
    """
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        params = dict(request.url.params)
        requests.append(request)
        return route(request.url.path, params, len(requests) - 1)

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    return Crossref(client=client), requests


def _source_id(path):
    # /members/<id>/works -> <id>
    return path.strip("/").split("/")[1]


def _paged_route(pages_by_id, total_by_id=None, failures=None):
    """Serve pages per source id.

    pages_by_id: id -> list of pages, each page is a list of DOI strings.
    failures: id -> {page_number (1-based): response}; the failing response
        is returned for that request and any later request for the id (the
        run should not make any). Page counts are keyed per id in the route
    closure, so repeating an id starts a fresh, independent sequence.
    """
    total_by_id = total_by_id or {}
    failures = failures or {}
    served = {}

    def route(path, params, _call):
        sid = _source_id(path)
        index = served.get(sid, 0)
        served[sid] = index + 1
        page_number = index + 1
        if sid in failures and page_number >= min(failures[sid]):
            return failures[sid][min(failures[sid])]
        pages = pages_by_id[sid]
        dois = pages[index]
        last = index == len(pages) - 1
        return _ok(
            dois,
            None if last else f"c{index + 1}",
            total=total_by_id.get(sid, sum(len(p) for p in pages)),
        )

    return route


# --- basics -----------------------------------------------------------------


def test_collect_returns_collect_result_with_one_source_per_input_id():
    route = _paged_route({"1": [["a", "b"]], "2": [["c"]]})
    cr, requests = _transport(route)

    result = WorksQuery(cr).members(["1", "2"]).limit(2).collect()

    assert isinstance(result, CollectResult)
    assert len(result) == 2
    assert all(isinstance(s, SourceResult) for s in result)
    assert [s.source_id for s in result] == ["1", "2"]
    assert [s.source_index for s in result] == [0, 1]
    assert result.endpoint == "members"
    assert result.statuses == [STATUS_COMPLETE, STATUS_COMPLETE]
    assert result.ok is True
    # every request hit the member works route
    assert all(r.url.path.startswith("/members/") for r in requests)


def test_collect_works_ordered_by_input_position_page_and_item():
    route = _paged_route(
        {
            "1": [["a", "b"], ["c", "d"]],
            "2": [["e"]],
            "3": [["f", "g"]],
        }
    )
    cr, _ = _transport(route)

    result = (
        WorksQuery(cr)
        .members(["1", "2", "3"])
        .limit(2)
        .cursor("*", cursor_max=100)
        .collect()
    )

    assert [w["DOI"] for w in result.works] == list("abcdefg")
    # provenance is aligned with works
    prov = result.provenance
    assert [p["source_id"] for p in prov] == list("1111233")
    assert [p["page"] for p in prov] == [1, 1, 2, 2, 1, 1, 1]
    assert [p["item_index"] for p in prov] == [0, 1, 0, 1, 0, 0, 1]
    assert [p["source_index"] for p in prov] == [0, 0, 0, 0, 1, 2, 2]


@pytest.mark.parametrize(
    "endpoint", ["members", "funders", "journals", "prefixes", "types"]
)
def test_collect_supports_all_five_source_endpoints(endpoint):
    seen = []

    def route(path, params, _call):
        seen.append(path)
        return _ok(["x"], None)

    cr, _ = _transport(route)
    q = WorksQuery(cr)
    q = getattr(q, endpoint)(["99"])

    result = q.collect()

    assert result.endpoint == endpoint
    assert seen == [f"/{endpoint}/99/works"]
    assert result[0].status == STATUS_COMPLETE


def test_collect_accepts_single_id_as_string_or_int():
    route = _paged_route({"98": [["a"]]})
    cr, _ = _transport(route)

    result = WorksQuery(cr).members(98).collect()

    assert len(result) == 1
    assert result[0].source_id == 98
    assert result[0].works == [{"DOI": "a"}]


def test_collect_sends_cursor_and_query_params_on_every_page():
    route = _paged_route({"1": [["a", "b"], ["c"]]})
    cr, requests = _transport(route)

    WorksQuery(cr).members(["1"]).query("zika").limit(2).collect()

    params_per_call = [dict(r.url.params) for r in requests]
    assert params_per_call[0]["cursor"] == "*"
    assert params_per_call[1]["cursor"] == "c1"
    assert all(p["rows"] == "2" for p in params_per_call)
    assert all(p["query"] == "zika" for p in params_per_call)


# --- duplicate IDs and DOI overlap ------------------------------------------


def test_collect_duplicate_ids_are_independent_and_dois_not_merged():
    # every source returns the same DOI; nothing may be deduplicated
    route = _paged_route(
        {"1": [["shared"]], "1again": [["shared"]], "1b": [["shared"]]}
    )
    cr, requests = _transport(route)

    result = WorksQuery(cr).members(["1", "1again", "1b"]).collect()

    assert [s.source_index for s in result] == [0, 1, 2]
    assert [w["DOI"] for w in result.works] == ["shared", "shared", "shared"]
    # three independent request sequences (first page each)
    assert len(requests) == 3
    first = result.sources[0]
    third = result.sources[2]
    assert first is not third
    assert first.works is not third.works


# --- empty sources ----------------------------------------------------------


def test_collect_empty_source_is_complete_not_missing():
    def route(path, params, _call):
        sid = _source_id(path)
        if sid == "empty":
            return _ok([], None, total=0)
        return _ok(["a"], None)

    cr, _ = _transport(route)
    result = WorksQuery(cr).members(["empty", "9", "empty"]).collect()

    assert result.statuses == [
        STATUS_COMPLETE,
        STATUS_COMPLETE,
        STATUS_COMPLETE,
    ]
    empty = result.sources[0]
    # the first (empty) page is a successful response and is retained,
    # but it contributes zero works
    assert len(empty.pages) == 1
    assert empty.pages[0]["message"]["items"] == []
    assert empty.works == []
    assert empty.count == 0
    assert empty.ok is True
    assert [s.source_id for s in result] == ["empty", "9", "empty"]
    assert [w["DOI"] for w in result.works] == ["a"]


def test_collect_all_empty_result_is_usable():
    def route(path, params, _call):
        return _ok([], None, total=0)

    cr, _ = _transport(route)
    result = WorksQuery(cr).members(["a", "b", "c"]).collect()

    assert result.statuses == [STATUS_COMPLETE] * 3
    assert result.works == []
    assert result.provenance == []
    assert result.ok is True
    assert len(result.failed) == 0


# --- failures: first page ---------------------------------------------------


def test_collect_first_page_failure_warn_keeps_running_other_sources():
    route = _paged_route(
        {"good": [["g1", "g2"]]},
        failures={"boom": {1: _bad_request()}},
    )
    cr, requests = _transport(route)

    with pytest.warns(UserWarning, match="400"):
        result = WorksQuery(cr).members(["boom", "good"]).collect(warn=True)

    assert result.statuses == [STATUS_FAILED, STATUS_COMPLETE]
    failed, good = result.sources
    assert failed.pages == []
    assert failed.failed_page == 1
    assert failed.error_type == "HTTPStatusError"
    assert "400" in failed.error_message
    assert failed.failed_url.startswith("https://api.crossref.org/members/boom/works")
    assert "cursor=%2A" in failed.failed_url
    # successful source still collected, run did not abort
    assert [w["DOI"] for w in good.works] == ["g1", "g2"]
    assert [w["DOI"] for w in result.works] == ["g1", "g2"]
    # one request for the failed source, then the good one
    assert [r.url.path for r in requests] == [
        "/members/boom/works",
        "/members/good/works",
    ]


def test_collect_first_page_failure_no_warn_records_exception_category():
    def route(path, params, _call):
        sid = _source_id(path)
        if sid == "boom":
            return _bad_request()
        return _ok(["g"], None)

    cr, _ = _transport(route)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warn=False must not warn
        result = WorksQuery(cr).members(["boom", "good"]).collect(warn=False)

    failed = result.sources[0]
    assert failed.status == STATUS_FAILED
    assert failed.failed_page == 1
    assert failed.error_type == "HTTPStatusError"
    assert "400" in failed.error_message
    assert result.sources[1].status == STATUS_COMPLETE


# --- failures: later page (partial data retained) ---------------------------


def test_collect_later_page_failure_retains_successful_pages():
    route = _paged_route(
        {"1": [["a", "b"], ["c", "d"], ["e", "f"]]},
        failures={"1": {3: _bad_request()}},
    )
    cr, requests = _transport(route)

    with pytest.warns(UserWarning, match="400"):
        result = WorksQuery(cr).members(["1"]).limit(2).collect(warn=True)

    source = result.sources[0]
    assert source.status == STATUS_FAILED
    assert source.failed_page == 3
    assert [w["DOI"] for w in source.works] == list("abcd")
    assert len(source.pages) == 2
    assert "cursor=c2" in source.failed_url
    assert source.error_type == "HTTPStatusError"
    # 2 ok pages + 1 failed attempt, then stop for this source
    assert len(requests) == 3


def test_collect_later_page_failure_partial_data_not_marked_complete():
    route = _paged_route(
        {"1": [["a", "b"], ["c", "d"]]},
        failures={"1": {2: _bad_request()}},
    )
    cr, _ = _transport(route)

    with pytest.warns(UserWarning):
        result = WorksQuery(cr).members(["1"]).collect(warn=True)

    source = result.sources[0]
    assert source.status == STATUS_FAILED
    assert source.ok is False
    assert source.capped is False
    assert result.ok is False
    assert len(result.complete) == 0
    assert len(result.failed) == 1


def test_collect_non_http_exception_is_recorded_and_run_continues():
    def route(path, params, _call):
        sid = _source_id(path)
        if sid == "timeout":
            raise httpx2.ConnectTimeout("timed out")
        return _ok(["g"], None)

    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda req: route(req.url.path, dict(req.url.params), 0)
        )
    )
    cr = Crossref(client=client, retries=0)
    result = WorksQuery(cr).members(["timeout", "good"]).collect()

    failed = result.sources[0]
    assert failed.status == STATUS_FAILED
    assert failed.failed_page == 1
    # Request._req wraps httpx2.HTTPError as RuntimeError (unchanged legacy
    # behavior); collect records the exception category that actually reached it
    assert failed.error_type == "RuntimeError"
    assert "timed out" in failed.error_message
    assert result.sources[1].status == STATUS_COMPLETE
    assert result.sources[1].works == [{"DOI": "g"}]


# --- cursor_max --------------------------------------------------------------


def test_collect_cursor_max_stops_and_marks_status():
    # 3 pages of 2 (= 6 works available), cap at 4 -> 2 pages, cap hit
    route = _paged_route(
        {"1": [["a", "b"], ["c", "d"], ["e", "f"]]},
        total_by_id={"1": 6},
    )
    cr, requests = _transport(route)

    result = WorksQuery(cr).members(["1"]).limit(2).cursor("*", cursor_max=4).collect()

    source = result.sources[0]
    assert source.status == STATUS_CURSOR_MAX
    assert source.capped is True
    assert [w["DOI"] for w in source.works] == list("abcd")
    assert len(requests) == 2  # no third page once the cap is hit


def test_collect_cursor_max_none_fetches_everything_and_completes():
    route = _paged_route(
        {"1": [["a", "b"], ["c", "d"], ["e"]]},
        total_by_id={"1": 5},
    )
    cr, requests = _transport(route)

    result = (
        WorksQuery(cr).members(["1"]).limit(2).cursor("*", cursor_max=None).collect()
    )

    source = result.sources[0]
    assert source.status == STATUS_COMPLETE
    assert [w["DOI"] for w in source.works] == list("abcde")
    assert len(requests) == 3


def test_collect_exhausting_before_cursor_max_is_complete():
    # only 2 works exist; cap of 100 must not report cursor_max
    route = _paged_route({"1": [["a", "b"]]}, total_by_id={"1": 2})
    cr, _ = _transport(route)

    result = WorksQuery(cr).members(["1"]).limit(2).cursor("*", 100).collect()

    assert result.sources[0].status == STATUS_COMPLETE
    assert result.sources[0].capped is False


def test_collect_bad_cursor_max_rejected_before_requests():
    def route(path, params, _call):
        raise AssertionError("no request should fire")

    cr, requests = _transport(route)
    q = WorksQuery(cr).members(["1"]).cursor("*", cursor_max=4.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        q.collect()
    assert requests == []


# --- mixed results -----------------------------------------------------------


def test_collect_mixed_complete_empty_capped_and_failed():
    failures = {"bad": {1: _bad_request()}}
    route = _paged_route(
        {
            "ok": [["k1", "k2"]],
            "empty": [[]],
            "cap": [["c1", "c2"], ["c3", "c4"], ["c5", "c6"]],
        },
        total_by_id={"cap": 6, "empty": 0},
        failures=failures,
    )
    cr, _ = _transport(route)

    with pytest.warns(UserWarning, match="400"):
        result = (
            WorksQuery(cr)
            .members(["ok", "empty", "cap", "bad"])
            .limit(2)
            .cursor("*", cursor_max=4)
            .collect(warn=True)
        )

    assert [s.source_id for s in result] == ["ok", "empty", "cap", "bad"]
    assert result.statuses == [
        STATUS_COMPLETE,
        STATUS_COMPLETE,
        STATUS_CURSOR_MAX,
        STATUS_FAILED,
    ]
    assert len(result.complete) == 2  # empty counts as complete
    assert [s.source_id for s in result.capped] == ["cap"]
    assert [s.source_id for s in result.failed] == ["bad"]
    assert result.ok is False
    # works order: ok (2), empty (0), cap partial (4), bad (0)
    assert [w["DOI"] for w in result.works] == ["k1", "k2", "c1", "c2", "c3", "c4"]
    # summary repr does not crash and mentions every state
    text = repr(result)
    assert "complete=2" in text
    assert "cursor_max=1" in text
    assert "failed=1" in text


# --- validation --------------------------------------------------------------


def test_collect_rejects_plain_works_endpoint():
    cr, _ = _transport(lambda *a: _ok([], None))
    with pytest.raises(ValueError, match="source query with IDs"):
        WorksQuery(cr).query("x").collect()


def test_collect_rejects_source_endpoint_without_ids():
    cr, _ = _transport(lambda *a: _ok([], None))
    with pytest.raises(ValueError, match="at least one source ID"):
        WorksQuery(cr).members().collect()


# --- WorksContainer integration ----------------------------------------------


def test_works_container_accepts_collect_result_and_keeps_provenance():
    route = _paged_route(
        {"1": [["a", "b"], ["c"]], "2": [["d"]]},
    )
    cr, _ = _transport(route)
    result = WorksQuery(cr).members(["1", "2"]).limit(2).collect()

    container = WorksContainer(result)

    assert isinstance(container, WorksContainer)
    assert [w["DOI"] for w in container.works] == list("abcd")
    assert container.doi == list("abcd")
    assert container.source_ids == list("1112")
    assert container.source_indexes == [0, 0, 0, 1]
    assert container.source_pages == [1, 1, 2, 1]
    assert [s.source_id for s in container.sources] == ["1", "2"]


def test_works_container_accepts_all_empty_collect_result():
    def route(path, params, _call):
        return _ok([], None, total=0)

    cr, _ = _transport(route)
    result = WorksQuery(cr).members(["a", "b"]).collect()

    container = WorksContainer(result)  # must not raise ValueError

    assert container.works == []
    # no work records means no dynamic per-field attributes are created, but
    # the provenance lists exist and are empty
    assert container.source_ids == []
    assert container.source_pages == []
    assert container.source_indexes == []
    assert len(container.sources) == 2
    assert "No. works: 0" in repr(container)
    assert not hasattr(container, "doi")


def test_works_container_collect_result_with_failed_source():
    route = _paged_route(
        {"bad": [["b1"], ["b2"]], "good": [["g1"]]},
        failures={"bad": {2: _bad_request()}},
    )
    cr, _ = _transport(route)

    with pytest.warns(UserWarning):
        result = WorksQuery(cr).members(["bad", "good"]).limit(1).collect(warn=True)
    container = WorksContainer(result)

    # partial page from bad retained, then good
    assert container.doi == ["b1", "g1"]
    assert container.source_ids == ["bad", "good"]
    assert container.source_pages == [1, 1]
    # status still visible via the retained sources
    assert container.sources[0].status == STATUS_FAILED
    assert container.sources[1].status == STATUS_COMPLETE


def test_works_container_legacy_behavior_unchanged():
    # empty legacy data still raises; the empty allowance is collect-only
    with pytest.raises(ValueError):
        WorksContainer({})
    with pytest.raises(ValueError):
        WorksContainer([])
    with pytest.raises(TypeError):
        WorksContainer(5)


# --- request-shape parity with execute() ------------------------------------


def test_collect_request_matches_execute_request_shape():
    calls = []

    def route(path, params, _call):
        calls.append((path, dict(params)))
        return _ok(["a"], None)

    cr, _ = _transport(route)
    q = WorksQuery(cr).members(["7"]).query("zika").filters(has_funder=True).limit(3)
    q.collect()
    # same query via execute(), separate stub instance for a clean call list
    calls2 = []

    def route2(path, params, _call):
        calls2.append((path, dict(params)))
        return _ok(["a"], None)

    cr2, _ = _transport(route2)
    WorksQuery(cr2).members(["7"]).query("zika").filters(has_funder=True).limit(
        3
    ).execute()

    assert calls[0][0] == calls2[0][0]
    # collect adds cursor deep paging; the rest of the params match
    collect_params = dict(calls[0][1])
    execute_params = dict(calls2[0][1])
    assert collect_params.pop("cursor") == "*"
    assert "cursor" not in execute_params
    assert collect_params == execute_params

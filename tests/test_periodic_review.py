"""Offline regressions for periodic retrieval/ranking/delivery (no external writes)."""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "periodic_worker_tests", Path(__file__).parents[1] / "scripts/periodic_review.py")
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)
W = worker
NOW = datetime(2026, 10, 11, 8, tzinfo=timezone.utc)


@pytest.mark.parametrize(("kind", "today", "start", "end"), [
    ("week", "2026-10-12", "2026-10-05", "2026-10-12"),
    ("week", "2026-10-11", "2026-09-28", "2026-10-05"),
    ("month", "2026-03-01", "2026-02-01", "2026-03-01"),
    ("month", "2024-03-01", "2024-02-01", "2024-03-01"),
    ("year", "2027-01-01", "2026-01-01", "2027-01-01"),
])
def test_calendar_windows(kind, today, start, end):
    p = W.previous_period(kind, date.fromisoformat(today))
    assert (str(p.start), str(p.end)) == (start, end)
    assert p.bounds()[0].tzinfo == W.BJ


def test_activation_no_unrequested_old_year_or_month():
    assert W.due_periods(date(2026, 10, 11), date(2026, 10, 11), {}) == []
    p = W.due_periods(date(2026, 10, 12), date(2026, 10, 11), {})
    assert [r.kind for r in p] == ["week"]


def test_retry_after_boundary_and_independent_week_month_year_keys():
    due = W.due_periods(date(2027, 1, 12), date(2026, 12, 28), {})
    assert {"week", "month", "year"} == {p.kind for p in due}
    monthly = next(p for p in due if p.kind == "month")
    remaining = W.due_periods(date(2027, 1, 12), date(2026, 12, 28), {monthly.key: {}})
    assert monthly not in remaining
    assert any(p.kind == "year" for p in remaining)


def paper(i, day="2026-10-01", **kwargs):
    return {"arxiv_id": f"2610.{i:05d}", "title": f"Paper {i}", "abstract": f"Abstract {i}",
            "published": day + "T01:00:00Z", "authors": ["Author"], "categories": ["cs.CV"],
            "url": f"https://arxiv.org/abs/2610.{i:05d}",
            "pdf_url": f"https://arxiv.org/pdf/2610.{i:05d}", **kwargs}


def feed(papers, total=None):
    root = ET.Element(W.ATOM + "feed")
    ET.SubElement(root, W.OPEN + "totalResults").text = str(len(papers) if total is None else total)
    for p in papers:
        e = ET.SubElement(root, W.ATOM + "entry")
        for key, value in {"id": p["url"], "title": p["title"], "summary": p["abstract"],
                           "published": p["published"]}.items():
            ET.SubElement(e, W.ATOM + key).text = value
        ET.SubElement(e, W.ATOM + "category", {"term": "cs.CV"})
    return root


def test_full_pagination_more_than_100_and_half_open_beijing_boundary(monkeypatch):
    # Query boundary for Beijing Oct 1 is Sep 30 16:00 UTC.
    rows = [paper(i) for i in range(701)]
    rows.append(paper(999, published="2026-10-01T16:00:00Z"))
    api = W.ArxivAPI()
    requests = []
    def request(params):
        requests.append(params)
        offset = params["start"]
        return feed(rows[offset:offset + 500], len(rows))
    monkeypatch.setattr(api, "request", request)
    result = api.day(date(2026, 10, 1), ["cs.CV"], float("inf"))
    assert len(result) == 701
    assert [r["start"] for r in requests] == [0, 500]
    assert "202609301600 TO 202610011600" in requests[0]["search_query"]
    assert all(p["arxiv_id"] != "2610.00999" for p in result)


@pytest.mark.parametrize("root", [ET.Element(W.ATOM + "feed"), feed([], 100)])
def test_bad_or_premature_empty_api_feed_is_not_zero_success(monkeypatch, root):
    api = W.ArxivAPI()
    monkeypatch.setattr(api, "request", lambda _: root)
    with pytest.raises(RuntimeError):
        api.day(date(2026, 10, 1), ["cs.CV"], float("inf"))


def test_arxiv_duplicate_page_rejected(monkeypatch):
    api = W.ArxivAPI()
    monkeypatch.setattr(api, "request", lambda _: feed([paper(1), paper(1)]))
    with pytest.raises(RuntimeError, match="Duplicate"):
        api.day(date(2026, 10, 1), ["cs.CV"], float("inf"))


def test_checkpoint_only_complete_days_and_resume_annual(tmp_path):
    p = W.Period("year", date(2024, 1, 1), date(2025, 1, 1))
    cache = W.PublicCache(tmp_path / "cache.sqlite", ["cs.CV"])
    calls = []
    class API:
        def day(self, day, categories, deadline):
            calls.append(day)
            if len(calls) == 3:
                raise W.DeadlineReached()
            return [paper((day - p.start).days, str(day))]
    with pytest.raises(W.DeadlineReached):
        cache.ensure_period(p, API(), float("inf"))
    with pytest.raises(RuntimeError, match="Incomplete"):
        cache.papers(p)
    class Resume:
        def day(self, day, categories, deadline):
            calls.append(day)
            return [paper((day - p.start).days, str(day))]
    cache.ensure_period(p, Resume(), float("inf"))
    assert len(cache.papers(p)) == 366
    assert calls.count(p.start) == 1
    assert calls.count(p.start + timedelta(days=2)) == 2
    cache.db.close()


def test_embeddings_cache_model_abstract_identity_and_order(tmp_path):
    cache = W.PublicCache(tmp_path / "cache.sqlite")
    count = []
    def embed(texts):
        count.append(len(texts))
        return np.array([[1., 0.] for _ in texts])
    papers = [paper(i) for i in range(101)]
    assert cache.vectors(papers, embed, float("inf")).shape == (101, 2)
    first = sum(count)
    cache.vectors(papers, embed, float("inf"))
    assert sum(count) == first == 101
    papers[0]["abstract"] = "Changed abstract"
    cache.vectors(papers, embed, float("inf"))
    assert sum(count) == 102
    cache.db.close()


def refs():
    return [{"key": "OLD", "title": "Old direction", "abstract": "old",
             "date_added": "2025-01-01T00:00:00Z"},
            {"key": "NEW", "title": "Latest direction", "abstract": "new",
             "date_added": "2026-10-10T00:00:00Z"}][::-1]


def test_full_rank_not_first_100_and_exact_recent_mixture():
    references = refs()
    rv = np.eye(2)
    candidates = [paper(i) for i in range(1200)]
    vectors = np.tile([0., 1.], (1200, 1))
    vectors[-1] = [1., 0.]
    ranked, weights = W.rank_all(candidates, vectors, references, rv, NOW)
    assert ranked[0]["arxiv_id"] == candidates[-1]["arxiv_id"]
    assert weights.sum() == pytest.approx(1)
    expected = (vectors / np.linalg.norm(vectors, axis=1)[:, None]) @ rv.T @ weights * 10
    assert [p["score"] for p in candidates] == pytest.approx(expected)
    assert weights[0] > .7  # latest save is present and dominates recent component
    old_ranked, _ = W.rank_all([paper(1), paper(2)],
                              np.array([[1., 0.], [0., 1.]]),
                              references[1:], np.array([[0., 1.]]), NOW)
    assert old_ranked[0]["arxiv_id"] == "2610.00002"


def test_saved_sent_rejected_exclusions_diagnostic_not_treated_as_formal(monkeypatch):
    state = {"sent": {}, "completed": {}, "rejected": [W.fingerprint("2610.00004")]}
    W.apply_receipt(state, {"api_accepted": True, "complete": True, "message_ids": ["om_real"],
                           "delivered_at": NOW.isoformat(), "papers": [
                               {"arxiv_id": "2610.00002"},
                               {"arxiv_id": "2610.00003", "recommendation_status": "diagnostic"}]})
    remaining = W.exclude_saved_and_sent([paper(i) for i in range(1, 6)],
                                         {"2610.00001"}, set(), state)
    assert [p["arxiv_id"] for p in remaining] == ["2610.00003", "2610.00005"]
    assert "2610.00002" not in json.dumps(state)  # persistent ledger is hash-only
    W.apply_receipt(state, {"smtp_accepted": True, "delivered_at": NOW.isoformat(),
                           "period_key": "month:2026-09-01:2026-10-01", "paper_count": 0, "papers": []})
    assert "month:2026-09-01:2026-10-01" in state["completed"]


def test_unacknowledged_send_never_completed():
    state = {"sent": {}, "completed": {}}
    W.apply_receipt(state, {"api_accepted": True, "complete": False, "papers": [paper(1)],
                           "period_key": "x", "paper_count": 1})
    assert state == {"sent": {}, "completed": {}}


def test_only_chinese_tldr_rendered_and_explicit_historical_scope():
    p = paper(1, tldr="针对视觉导航的中文简介", reason="与保存文献的定位任务相关",
              reference_keys=["NEW"], zotero_matches=refs(), score=7.123456)
    result = W.render_html(W.previous_period("week", date(2026, 10, 12)), [p], 1200, "6613", 568)
    assert "周补选" in result and "不是今日新论文" in result
    assert "1200" in result and "7.1235" in result and "Latest direction" in result
    assert p["url"] in result and p["pdf_url"] in result


def test_gate_scans_below_first_100_and_never_fills_weak(monkeypatch):
    from scripts import periodic_relevance
    seen = []
    def gate(papers, state, client):
        seen.extend(p["arxiv_id"] for p in papers)
        return {p["arxiv_id"]: {"relation": "direct" if int(p["arxiv_id"][-5:]) >= 105 else "weak",
                               "reason": "对应视觉导航任务", "reference_keys": ["NEW"]}
                for p in papers}
    monkeypatch.setattr(periodic_relevance, "screen_zotero_relevance", gate)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content="论文的中文简介"))]))))
    rows = [paper(i, zotero_matches=refs()) for i in range(120)]
    selected, checked = W.screen_and_summarize(rows, refs(), np.array([.8, .2]), NOW, client)
    assert len(selected) == 15
    assert checked == 120
    assert selected[0]["arxiv_id"] == "2610.00105"
    assert all(p["tldr"] == "论文的中文简介" for p in selected)


def test_delivery_dry_run_no_sender_or_ledger_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = W.previous_period("week", date(2026, 10, 12))
    fake_cache = SimpleNamespace(ensure_period=lambda *a: None, papers=lambda p: [])
    fake_ledger = SimpleNamespace(load=lambda: ({"sent": {}, "completed": {}}, None),
                                  update=lambda *a: pytest.fail("dry run wrote ledger"))
    monkeypatch.setattr(W, "fresh_zotero", lambda: (refs(), "6613", set(), set()))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(W, "deliver_email", lambda *a: pytest.fail("dry run sent SMTP"))
    assert W.execute_period(p, fake_cache, fake_ledger, "email", True, float("inf")) is None
    report = json.loads(next((tmp_path / "outputs/periodic").glob("week-*.json")).read_text())
    assert report["metadata_complete"] and report["zotero_version"] == "6613"
    assert not (tmp_path / "outputs/periodic/periodic-receipts.json").exists()


def test_new_month_rescans_metadata_instead_of_reusing_old_week(tmp_path):
    cache = W.PublicCache(tmp_path / "cache.sqlite", ["cs.CV"])
    calls = []
    class API:
        def day(self, day, categories, deadline):
            calls.append(day)
            return [paper((day - date(2026, 10, 1)).days, str(day))]
    week = W.Period("week", date(2026, 10, 1), date(2026, 10, 3))
    month = W.Period("month", date(2026, 10, 1), date(2026, 10, 3))
    cache.ensure_period(week, API(), float("inf"))
    cache.ensure_period(week, API(), float("inf"))
    assert len(calls) == 2
    cache.ensure_period(month, API(), float("inf"))
    assert len(calls) == 4
    cache.db.close()


def test_embedding_api_reorders_and_rejects_missing_indexes():
    def response(rows):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in rows])
    client = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda **k: response([(1, [0., 1.]), (0, [1., 0.])])))
    assert W.embed_texts(client, ["a", "b"]).tolist() == [[1., 0.], [0., 1.]]
    client.embeddings.create = lambda **k: response([(0, [1., 0.])])
    with pytest.raises(RuntimeError, match="Incomplete"):
        W.embed_texts(client, ["a", "b"])


def test_zotero_all_types_saved_exclusion_and_consistent_version(monkeypatch):
    import requests
    rows = [
        {"key": "A", "data": {"title": "Saved but no abstract", "abstractNote": "",
                              "itemType": "document", "url": "https://arxiv.org/abs/2610.00001",
                              "dateAdded": "2026-10-10T00:00:00Z"}},
        {"key": "B", "data": {"title": "Eligible", "abstractNote": "Real abstract",
                              "itemType": "preprint", "url": "https://arxiv.org/abs/2610.00002",
                              "dateAdded": "2026-10-10T00:00:00Z"}}]
    def get(url, params, **kwargs):
        return SimpleNamespace(raise_for_status=lambda: None,
                               headers={"Total-Results": "2", "Last-Modified-Version": "6613"},
                               json=lambda: rows[params["start"]:params["start"] + 1])
    monkeypatch.setattr(requests, "Session", lambda: SimpleNamespace(get=get))
    monkeypatch.setenv("ZOTERO_ID", "dummy")
    monkeypatch.setenv("ZOTERO_KEY", "dummy")
    references, version, saved, titles = W.fresh_zotero()
    assert len(references) == 1 and version == "6613"
    assert saved == {"2610.00001", "2610.00002"}
    assert "savedbutnoabstract" in titles
    def changed(url, params, **kwargs):
        r = get(url, params, **kwargs)
        if params["start"] > 0:
            r.headers["Last-Modified-Version"] = "6614"
        return r
    monkeypatch.setattr(requests, "Session", lambda: SimpleNamespace(get=changed))
    with pytest.raises(RuntimeError, match="Zotero changed"):
        W.fresh_zotero()


def test_feishu_three_cards_real_ack_and_stable_idempotency(monkeypatch):
    import requests
    calls = []
    def post(url, json, **kwargs):
        calls.append(json)
        payload = ({"code": 0, "tenant_access_token": "dummy"} if "auth" in url else
                   {"code": 0, "data": {"message_id": f"om_{len(calls)}"}})
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(requests, "Session", lambda: SimpleNamespace(post=post))
    for key in ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_CHAT_ID", "GITHUB_REPOSITORY"]:
        monkeypatch.setenv(key, "dummy")
    rows = [paper(i, score=8., tldr="中文简介", reason="定位任务相关",
                  cited_titles=["我的保存文献"]) for i in range(30)]
    period = W.previous_period("week", date(2026, 10, 12))
    receipt = W.deliver_feishu(period, rows, 5000, version="6613", refs_count=568)
    assert len(receipt["message_ids"]) == 3
    cards = calls[1:]
    assert len(cards) == 3 and len({c["uuid"] for c in cards}) == 3
    assert "6613" in cards[0]["content"] and "我的保存文献" in cards[0]["content"]
    old_uuid = cards[0]["uuid"]
    calls.clear()
    W.deliver_feishu(period, rows, 5000)
    assert calls[1]["uuid"] == old_uuid


def test_accepted_receipt_survives_ledger_failure_for_recovery(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_cache = SimpleNamespace(ensure_period=lambda *a: None, papers=lambda p: [])
    def fail(*args):
        raise RuntimeError("GitHub state unavailable after SMTP accepted")
    fake_ledger = SimpleNamespace(load=lambda: ({"sent": {}, "completed": {}}, None), update=fail)
    monkeypatch.setattr(W, "fresh_zotero", lambda: (refs(), "6613", set(), set()))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setattr(W, "deliver_email", lambda *a: {"smtp_accepted": True})
    with pytest.raises(RuntimeError, match="GitHub state"):
        W.execute_period(W.previous_period("year", date(2027, 1, 1)), fake_cache,
                         fake_ledger, "email", False, float("inf"))

def test_idle_auto_does_not_create_or_reupload_large_cache(monkeypatch):
    monkeypatch.setattr(W, "Ledger", lambda: SimpleNamespace(load=lambda: (
        {"activated_on": "2026-10-11", "completed": {}, "sent": {}}, None)))
    monkeypatch.setattr(W, "due_periods", lambda *a: [])
    monkeypatch.setattr(W, "PublicCache", lambda *a: pytest.fail("idle run touched cache"))
    monkeypatch.setattr(sys, "argv", ["worker", "--channel", "email", "--dry-run"])
    W.main()

    proof = json.loads((tmp_path / "outputs/periodic/periodic-receipts.json").read_text())[0]
    assert proof["smtp_accepted"] and proof["period_key"].startswith("year:")

def test_changed_profile_during_ranking_blocks_send(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_cache = SimpleNamespace(ensure_period=lambda *a: None, papers=lambda p: [])
    fake_ledger = SimpleNamespace(load=lambda: ({"sent": {}, "completed": {}}, None),
                                  update=lambda *a: pytest.fail("changed profile wrote completion"))
    versions = iter(["6613", "6614"])
    monkeypatch.setattr(W, "fresh_zotero", lambda: (refs(), next(versions), set(), set()))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(W, "deliver_email", lambda *a: pytest.fail("stale profile sent"))
    with pytest.raises(RuntimeError, match="changed during periodic ranking"):
        W.execute_period(W.previous_period("week", date(2026, 10, 12)), fake_cache,
                         fake_ledger, "email", False, float("inf"))




def test_failed_delivery_never_records_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_cache = SimpleNamespace(ensure_period=lambda *a: None, papers=lambda p: [])
    fake_ledger = SimpleNamespace(load=lambda: ({"sent": {}, "completed": {}}, None),
                                  update=lambda *a: pytest.fail("failed send wrote ledger"))
    monkeypatch.setattr(W, "fresh_zotero", lambda: (refs(), "6613", set(), set()))
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    def fail(*a):
        raise RuntimeError("SMTP unavailable")
    monkeypatch.setattr(W, "deliver_email", fail)
    with pytest.raises(RuntimeError, match="SMTP unavailable"):
        W.execute_period(W.previous_period("month", date(2026, 10, 1)), fake_cache,
                         fake_ledger, "email", False, float("inf"))
    assert not (tmp_path / "outputs/periodic/periodic-receipts.json").exists()


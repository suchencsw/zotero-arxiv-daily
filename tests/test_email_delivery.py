import json
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from omegaconf import open_dict

from scripts.email_delivery_guard import decide_delivery, SafeRedirect
from zotero_arxiv_daily.executor import Executor


NOW = datetime(2026, 10, 10, 17, 30, tzinfo=timezone.utc)  # Beijing Oct 11
RUN = {"id": 12, "status": "completed", "conclusion": "success", "created_at": "2026-10-10T17:00:00Z"}


def proof(**changes):
    result = {"schema_version": 1, "run_id": "12", "smtp_accepted": True,
              "delivered_at": "2026-10-10T17:05:00Z", "kind": "papers", "paper_count": 30}
    result.update(changes)
    return result


@pytest.mark.parametrize("receipt", [None, {}, proof(smtp_accepted=False), proof(run_id="wrong"),
    proof(delivered_at="2026-10-10T12:00:00Z"), proof(paper_count=0), proof(kind="recovery", recovery_batch="2026-10-09")])
def test_green_or_utc_same_day_is_not_delivery_proof(receipt):
    assert decide_delivery([RUN], lambda _: receipt, now=NOW, current_run=99)[0] is True


def test_beijing_date_proof_prevents_duplicate_even_if_post_send_job_failed():
    run = dict(RUN, conclusion="failure")
    assert decide_delivery([run], lambda _: proof(), now=NOW, current_run=99) == (False, False, "12")


def test_empty_notice_never_blocks_later_nonempty_delivery():
    assert decide_delivery([RUN], lambda _: proof(kind="empty", paper_count=0), now=NOW, current_run=99) == (True, True, None)


def test_recovery_deduplicates_only_its_own_batch():
    receipt = proof(kind="recovery", recovery_batch="2026-10-09")
    assert not decide_delivery([RUN], lambda _: receipt, now=NOW, current_run=99, recovery="2026-10-09")[0]
    assert decide_delivery([RUN], lambda _: receipt, now=NOW, current_run=99, recovery="2026-10-08")[0]


def setup_executor(config, monkeypatch, tmp_path, papers):
    with open_dict(config):
        config.executor.send_empty = True
    receipt = tmp_path / "smtp-receipt.json"
    monkeypatch.setenv("EMAIL_DELIVERY_RECEIPT", str(receipt))
    monkeypatch.setenv("GITHUB_RUN_ID", "12")
    monkeypatch.delenv("EMAIL_RECOVERY_BATCH", raising=False)
    monkeypatch.delenv("EMAIL_DRY_RUN", raising=False)
    monkeypatch.delenv("EMAIL_EMPTY_ALREADY_SENT", raising=False)
    ex = Executor.__new__(Executor)
    ex.config = config
    ex.fetch_zotero_corpus = lambda: [object()]
    ex.filter_corpus = lambda corpus: corpus
    ex.retrievers = {"arxiv": SimpleNamespace(retrieve_papers=lambda: papers)}
    ex.reranker = SimpleNamespace(rerank=lambda p, c: p)
    ex.openai_client = None
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda p: "<html>preview</html>")
    return ex, receipt


def test_ack_written_only_after_smtp_acceptance(config, monkeypatch, tmp_path):
    paper = SimpleNamespace(title="Navigation", url="https://arxiv.org/abs/2610.11815", score=0.6,
                            generate_tldr=lambda *a: None, generate_affiliations=lambda *a: None)
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, [paper])
    def send(*args):
        assert not receipt.exists()
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", send)
    ex.run()
    result = json.loads(receipt.read_text())
    assert result["smtp_accepted"] is True and result["paper_count"] == 1
    assert result["kind"] == "papers"
    assert "api_key" not in receipt.read_text() and "sender" not in receipt.read_text()


def test_failed_smtp_does_not_produce_ack(config, monkeypatch, tmp_path):
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, [])
    def fail(*args):
        raise RuntimeError("SMTP refused")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", fail)
    with pytest.raises(RuntimeError, match="SMTP refused"):
        ex.run()
    assert not receipt.exists()


def test_dry_run_never_sends_or_writes_ack(config, monkeypatch, tmp_path):
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, [])
    monkeypatch.setenv("EMAIL_DRY_RUN", "true")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *a: pytest.fail("dry-run sent email"))
    ex.run()
    assert not receipt.exists()


def test_repeated_empty_notice_is_skipped_without_disabling_retry(config, monkeypatch, tmp_path):
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, [])
    monkeypatch.setenv("EMAIL_EMPTY_ALREADY_SENT", "true")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *a: pytest.fail("duplicate empty notice"))
    ex.run()
    assert not receipt.exists()


def test_empty_zotero_fails_closed_without_credentials_in_message(config, monkeypatch, tmp_path):
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, [])
    ex.fetch_zotero_corpus = lambda: []
    with pytest.raises(RuntimeError, match="No eligible Zotero") as exc:
        ex.run()
    assert "fake-zotero-key" not in str(exc.value)
    assert not receipt.exists()


def test_recovery_snapshot_is_dated_and_caps_at100(config, monkeypatch):
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    monkeypatch.setenv("EMAIL_RECOVERY_BATCH", "2026-10-09")
    rows = [{"arxiv_id": f"2610.{i:05d}", "title": "Actual metadata", "abstract": "Navigation",
             "authors": ["Author"], "publish_date": "2026-10-09"} for i in range(120)]
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: json.dumps({"papers": rows}))
    assert len(ArxivRetriever(config)._retrieve_raw_papers()) == 100
    rows[0]["publish_date"] = "2026-10-08"
    with pytest.raises(RuntimeError, match="dated, real"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_signed_artifact_redirect_does_not_leak_github_token():
    from urllib.request import Request
    original = Request("https://api.github.com/a", headers={"Authorization": "Bearer test"})
    redirected = SafeRedirect().redirect_request(original, None, 302, "Found", {}, "https://storage.example/receipt.zip")
    assert not redirected.has_header("Authorization")


def test_historical_email_is_labeled(config, monkeypatch):
    from zotero_arxiv_daily.construct_email import render_email
    from tests.canned_responses import make_sample_paper
    monkeypatch.setenv("EMAIL_RECOVERY_BATCH", "2026-10-09")
    assert "不是今日新论文" in render_email([make_sample_paper()])


def test_queued_email_consumes_frozen_batch_not_live_feed(config, monkeypatch, tmp_path):
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"batch_id": "rss:2026-10-12:scope", "candidates": [
        {"arxiv_id": "2610.11815", "title": "Navigation", "abstract": "Real abstract",
         "authors": ["Author"], "url": "https://arxiv.org/abs/2610.11815", "publish_date": "2026-10-12"}
    ]}), encoding="utf8")
    monkeypatch.setenv("ARXIV_BATCH_PATH", str(path))
    monkeypatch.setenv("ARXIV_BATCH_ID", "rss:2026-10-12:scope")
    monkeypatch.setenv("ARXIV_BATCH_DATE", "2026-10-12")
    monkeypatch.delenv("EMAIL_RECOVERY_BATCH", raising=False)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.arxiv_retriever.feedparser.parse",
                        lambda *a: pytest.fail("Queued delivery refetched changing RSS"))
    assert ArxivRetriever(config)._retrieve_raw_papers()[0]["summary"] == "Real abstract"
    monkeypatch.setenv("ARXIV_BATCH_DATE", "2026-10-13")
    with pytest.raises(RuntimeError, match="metadata"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_batch_heading_and_review_are_not_send_date(monkeypatch):
    from zotero_arxiv_daily.construct_email import render_email
    from tests.canned_responses import make_sample_paper
    monkeypatch.delenv("EMAIL_RECOVERY_BATCH", raising=False)
    monkeypatch.setenv("ARXIV_BATCH_DATE", "2026-10-09")
    monkeypatch.setenv("ARXIV_BATCH_MODE", "review")
    text = render_email([make_sample_paper()])
    assert "历史批次复盘 · 2026-10-09" in text
    assert "不是提交日期或邮件发送日期" in text


def test_batch_tracking_never_sends_premature_empty_email(config, monkeypatch, tmp_path):
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, [])
    monkeypatch.setenv("ARXIV_BATCH_TRACKING", "true")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *a: pytest.fail("Empty spam"))
    ex.run()
    assert not receipt.exists()


def test_review_limits_to_five_and_records_batch_not_date_guard(config, monkeypatch, tmp_path):
    papers = [SimpleNamespace(title=f"Nav {i}", url=f"https://arxiv.org/abs/2610.{i:05}", score=7-i/10,
                             generate_tldr=lambda *a: None, generate_affiliations=lambda *a: None) for i in range(30)]
    ex, receipt = setup_executor(config, monkeypatch, tmp_path, papers)
    monkeypatch.setenv("ARXIV_BATCH_MODE", "review")
    monkeypatch.setenv("ARXIV_BATCH_ID", "rss:2026-10-09:scope")
    monkeypatch.setenv("ARXIV_BATCH_DATE", "2026-10-09")
    monkeypatch.setenv("ARXIV_REVIEW_KEY", "review:2026-10-12:rss:2026-10-09:scope")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *a: None)
    ex.run()
    r = json.loads(receipt.read_text(encoding="utf8"))
    assert r["paper_count"] == 5 and r["announcement_date"] == "2026-10-09"
    assert r["announcement_mode"] == "review"

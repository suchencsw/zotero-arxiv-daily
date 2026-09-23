"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import arxiv

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    _run_with_hard_timeout,
)
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e
        for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(
            SimpleNamespace(
                title=entry.title,
                authors=[SimpleNamespace(name="Test Author")],
                summary="Test abstract",
                pdf_url=f"https://arxiv.org/pdf/{pid}",
                entry_id=f"https://arxiv.org/abs/{pid}",
                source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
            )
        )

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_arxiv_rate_limit_uses_rss_metadata_without_per_paper_fallback(
    config, mock_feedparser, monkeypatch
):
    requests: list[list[str]] = []

    class RateLimitedClient:
        def __init__(self, **kwargs):
            assert kwargs == {"num_retries": 1, "delay_seconds": 3}

        def results(self, search):
            requests.append(search.id_list)
            raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 406)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", RateLimitedClient)

    papers = ArxivRetriever(config)._retrieve_raw_papers()

    assert len(requests) == 1
    assert len(papers) == len(
        [
            entry
            for entry in mock_feedparser.entries
            if entry.get("arxiv_announce_type", "new") == "new"
        ]
    )
    assert all("Announce Type:" not in paper.summary for paper in papers)
    assert papers[0].source_url() == "https://arxiv.org/e-print/2508.14001v1"

    converted = ArxivRetriever(config).convert_to_paper(papers[0])
    assert converted.full_text is None
    assert converted.abstract == papers[0].summary


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 0.01),
        timeout=1,
        operation="test op",
        paper_title="paper",
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append)
    )
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 1.0),
        timeout=0.01,
        operation="test op",
        paper_title="paper",
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append)
    )
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]

"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    allowed_announce_types = (
        {"new", "cross"}
        if config.source.arxiv.include_cross_list
        else {"new"}
    )
    expected_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") in allowed_announce_types
    ]

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(expected_entries)
    assert set(p.title for p in papers) == set(e.title for e in expected_entries)


def test_arxiv_rss_caps_candidates_without_calling_export_api(config, monkeypatch):
    entries = [
        {
            "id": f"oai:arXiv.org:2609.{index:05d}v1",
            "title": f"Paper {index}",
            "author": "Author One, Author Two",
            "summary": (
                f"arXiv:2609.{index:05d}v1 Announce Type: new\n"
                f"Abstract: Abstract {index}"
            ),
            "link": f"https://arxiv.org/abs/2609.{index:05d}v1",
            "arxiv_announce_type": "new",
        }
        for index in range(150)
    ]
    feed = SimpleNamespace(
        feed={"title": "cs.AI updates on arXiv.org"},
        entries=entries,
        bozo=False,
    )
    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", lambda _url: feed)
    monkeypatch.setattr(
        arxiv_retriever.arxiv,
        "Client",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    retriever = ArxivRetriever(config)
    raw = retriever._retrieve_raw_papers()
    paper = retriever.convert_to_paper(raw[0])

    assert len(raw) == 100
    assert paper.abstract == "Abstract 0"
    assert paper.authors == ["Author One", "Author Two"]
    assert paper.pdf_url == "https://arxiv.org/pdf/2609.00000v1"
    assert paper.full_text is None


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]

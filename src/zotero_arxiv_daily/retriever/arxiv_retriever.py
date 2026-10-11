from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import html
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    # Windows' spawn startup imports the full scientific stack in every child;
    # that alone can exceed short operation timeouts. Network helpers already
    # have request-level timeouts, so use a non-blocking worker thread there.
    if os.name == "nt":
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning(
                f"{operation} timed out for {paper_title} after {timeout} seconds"
            )
            return None
        except Exception as exc:
            logger.warning(
                f"{operation} failed for {paper_title}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context(
        "fork" if "fork" in start_methods else start_methods[0]
    )
    result_queue = context.Queue()
    process = context.Process(
        target=_run_in_subprocess,
        args=(result_queue, func, args),
    )
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(
            f"{operation} timed out for {paper_title} after {timeout} seconds"
        )
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )
    if not text:
        raise ValueError(f"No text extracted from {html_url}")

    return text


def _extract_text_from_tar_worker(
    source_url: str,
    paper_id: str,
    paper_title: str | None = None,
) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(
            path,
            paper_id,
            paper_title=paper_title,
        )
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")

        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[Any]:
        recovery = os.environ.get("EMAIL_RECOVERY_BATCH", "")
        if recovery:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", recovery):
                raise ValueError("Recovery batch must be YYYY-MM-DD")
            snapshot = Path(__file__).resolve().parents[3] / "recovery" / f"arxiv-{recovery}.json"
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            if not data.get("papers") or any(p.get("publish_date") != recovery for p in data["papers"]):
                raise RuntimeError("Recovery requires dated, real arXiv announcement metadata")
            rows = []
            seen = set()
            for paper in data["papers"]:
                identifier = str(paper["arxiv_id"])
                if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", identifier) or not paper.get("abstract", "").strip():
                    raise RuntimeError("Invalid recovery paper identifier or empty abstract")
                canonical = re.sub(r"v\d+$", "", identifier)
                if canonical in seen:
                    continue
                seen.add(canonical)
                rows.append({"id": identifier, "title": paper["title"], "summary": paper["abstract"],
                             "author": ", ".join(paper.get("authors", [])), "link": f"https://arxiv.org/abs/{identifier}"})
            limit = min(100, max(1, int(self.config.executor.get("max_candidate_num", 100))))
            logger.info("Historical recovery batch {}: {} real candidates (not today's new papers)", recovery, min(len(rows), limit))
            return rows[:limit]
        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get(
            "include_cross_list",
            False,
        )

        # Get the latest paper from arXiv RSS feed.
        feed_url = f"https://rss.arxiv.org/atom/{query}"
        feed = feedparser.parse(feed_url)
        if int(getattr(feed, "status", 200) or 200) >= 400:
            raise RuntimeError("arXiv RSS HTTP failure; not a valid empty announcement batch")
        feed_title = str(feed.feed.get("title", ""))
        if "Feed error for query" in feed_title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        if getattr(feed, "bozo", False) and not feed.entries:
            raise RuntimeError(
                f"Failed to parse arXiv RSS feed {feed_url}: "
                f"{getattr(feed, 'bozo_exception', 'unknown feed error')}"
            )

        allowed_announce_types = (
            {"new", "cross"} if include_cross_list else {"new"}
        )
        raw_papers = []
        seen_ids = set()
        for item in feed.entries:
            if item.get("arxiv_announce_type", "new") not in allowed_announce_types:
                continue
            paper_id = str(item.get("id", "")).removeprefix("oai:arXiv.org:")
            canonical_id = re.sub(r"v\d+$", "", paper_id)
            if not canonical_id or canonical_id in seen_ids:
                continue
            seen_ids.add(canonical_id)
            raw_papers.append(item)

        max_candidates = max(
            1,
            int(self.config.executor.get("max_candidate_num", 100) or 100),
        )
        if self.config.executor.debug:
            max_candidates = min(max_candidates, 10)
        selected = raw_papers[:max_candidates]
        logger.info(
            f"arXiv RSS returned {len(raw_papers)} unique announcements; "
            f"using {len(selected)} candidates"
        )
        return selected

    def convert_to_paper(self, raw_paper: Any) -> Paper:
        """Convert RSS metadata directly, avoiding hundreds of API/download calls."""

        paper_id = str(raw_paper.get("id", "")).removeprefix("oai:arXiv.org:")
        title = html.unescape(str(raw_paper.get("title", ""))).strip()
        author_text = html.unescape(str(raw_paper.get("author", ""))).strip()
        authors = [name.strip() for name in author_text.split(",") if name.strip()]
        abstract = html.unescape(str(raw_paper.get("summary", ""))).strip()
        abstract = re.sub(
            r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*",
            "",
            abstract,
            flags=re.IGNORECASE,
        )
        abstract = re.sub(r"\s+", " ", abstract).strip()
        url = str(raw_paper.get("link", "") or f"https://arxiv.org/abs/{paper_id}")
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            # Abstracts are sufficient for similarity ranking and Chinese TLDR.
            # Downloading every candidate's source/PDF previously made Actions
            # runs last for six hours before cancellation.
            full_text=None,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")

    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None

    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None

    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )

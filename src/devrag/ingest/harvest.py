from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import arxiv
import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from devrag.ingest.utils import stable_doc_id
from devrag.logging_setup import get_logger

log = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch(url: str, timeout: float = 30.0) -> str:
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.text


def harvest_arxiv(
    categories: list[str],
    *,
    lookback_months: int = 18,
    max_docs: int = 20_000,
) -> list[dict[str, Any]]:
    log.info("arxiv.harvest.start", categories=categories, max_docs=max_docs)
    query = " OR ".join(f"cat:{c}" for c in categories)
    client = arxiv.Client(page_size=200, delay_seconds=3.5, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_docs,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    docs: list[dict[str, Any]] = []
    cutoff = time.time() - lookback_months * 30 * 86400
    for r in tqdm(client.results(search), total=max_docs, desc="arxiv"):
        published_ts = r.published.timestamp()
        if published_ts < cutoff:
            break
        docs.append(
            {
                "doc_id": stable_doc_id("arxiv", r.entry_id),
                "title": r.title.strip(),
                "text": (r.summary or "").strip(),
                "source": "arxiv",
                "url": r.entry_id,
                "published": r.published.isoformat(),
                "authors": [a.name for a in r.authors],
            }
        )
    log.info("arxiv.harvest.done", n=len(docs))
    return docs


def harvest_framework_docs(urls: Iterable[str]) -> list[dict[str, Any]]:
    log.info("docs.harvest.start", n=len(list(urls)))
    out: list[dict[str, Any]] = []
    for url in urls:
        try:
            html = _fetch(url)
        except Exception as e:
            log.warning("docs.harvest.fail", url=url, error=str(e))
            continue
        text = _html_to_text(html)
        if len(text) < 200:
            continue
        out.append(
            {
                "doc_id": stable_doc_id("docs", url),
                "title": url.rstrip("/").split("/")[-1].replace("-", " ").title(),
                "text": text,
                "source": "framework_docs",
                "url": url,
            }
        )
    log.info("docs.harvest.done", n=len(out))
    return out


def harvest_blogs(feeds: Iterable[str]) -> list[dict[str, Any]]:
    log.info("blogs.harvest.start", n=len(list(feeds)))
    out: list[dict[str, Any]] = []
    for feed_url in feeds:
        try:
            d = feedparser.parse(feed_url)
        except Exception as e:
            log.warning("blogs.harvest.fail", url=feed_url, error=str(e))
            continue
        for e in d.entries[:60]:
            link = getattr(e, "link", "")
            summary = getattr(e, "summary", "") or getattr(e, "description", "")
            title = getattr(e, "title", link)
            if not link or not summary:
                continue
            out.append(
                {
                    "doc_id": stable_doc_id("blog", link),
                    "title": title,
                    "text": summary,
                    "source": "blog",
                    "url": link,
                    "published": getattr(e, "published", ""),
                }
            )
    log.info("blogs.harvest.done", n=len(out))
    return out


def _html_to_text(html: str) -> str:
    try:
        import trafilatura

        return trafilatura.extract(html) or ""
    except Exception:
        try:
            from markdownify import markdownify

            return markdownify(html, heading_style="ATX")
        except Exception:
            import re

            return re.sub(r"<[^>]+>", " ", html)
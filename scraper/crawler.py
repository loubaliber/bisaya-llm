"""Generic, source-agnostic crawling engine.

Site-specific spiders (scraper/talkbisaya.py, scraper/binisaya.py) subclass
`BaseSpider` and only implement URL discovery + word-slug extraction; all
politeness/retry/checkpoint/dedup logic lives here so it cannot be
accidentally skipped by a new spider.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from common import setup_logging
from schemas import RawEntry, SourceName
from scraper.rate_limiter import RateLimiter
from scraper.robots import RobotsCache

logger = setup_logging("scraper")


class RobotsDisallowedError(Exception):
    """Raised when robots.txt forbids fetching a URL. Never silently swallowed."""


class BaseSpider(ABC):
    """Shared crawl loop: robots check -> rate limit -> fetch -> retry -> checkpoint."""

    source: SourceName

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        rate_limit_seconds: float = 2.0,
        max_rate_limit_seconds: float = 5.0,
        timeout_seconds: float = 20.0,
        max_retries: int = 4,
        raw_output_dir: Path = Path("output/raw"),
        checkpoint_every: int = 25,
    ) -> None:
        self.base_url = base_url
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.raw_output_dir = raw_output_dir
        self.checkpoint_every = checkpoint_every

        self.rate_limiter = RateLimiter(rate_limit_seconds, max_rate_limit_seconds)
        self.robots = RobotsCache(user_agent=user_agent)
        self.client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=True,
        )

        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.raw_output_dir / f"{self.source}.jsonl"
        self.checkpoint_path = self.raw_output_dir / f"{self.source}.checkpoint.json"

    # -- to be implemented by site-specific spiders -----------------------
    @abstractmethod
    def discover_entry_urls(self) -> list[tuple[str, str]]:
        """Return a list of (word_slug, absolute_url) pairs to fetch."""

    # -- shared engine ------------------------------------------------------
    def _seen_urls(self) -> set[str]:
        seen: set[str] = set()
        if self.output_path.exists():
            with self.output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        seen.add(record["url"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return seen

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )
    def _fetch(self, url: str) -> httpx.Response:
        response = self.client.get(url)
        response.raise_for_status()
        return response

    def fetch_one(self, word: str, url: str) -> RawEntry | None:
        host = urlparse(url).netloc
        if not self.robots.is_allowed(url, self.client):
            logger.warning("robots.txt disallows %s -- skipping", url)
            return None

        crawl_delay = self.robots.crawl_delay(url, self.client)
        slept = self.rate_limiter.wait(host, override_delay=crawl_delay)
        logger.debug("slept %.2fs before fetching %s", slept, url)

        try:
            response = self._fetch(url)
        except httpx.HTTPError as exc:
            logger.error("failed to fetch %s after retries: %s", url, exc)
            return None

        return RawEntry(word=word, url=url, html=response.text, source=self.source)

    def run(self, limit: int | None = None) -> int:
        """Crawl all discovered entries, skipping already-downloaded URLs.
        Returns the number of new entries written."""
        already_seen = self._seen_urls()
        targets = self.discover_entry_urls()
        if limit is not None:
            targets = targets[:limit]

        new_count = 0
        with self.output_path.open("a", encoding="utf-8") as out_f:
            for i, (word, url) in enumerate(targets, start=1):
                if url in already_seen:
                    continue
                entry = self.fetch_one(word, url)
                if entry is None:
                    continue
                out_f.write(entry.model_dump_json() + "\n")
                out_f.flush()
                new_count += 1
                logger.info("[%d/%d] saved %s (%s)", i, len(targets), word, self.source)

                if new_count % self.checkpoint_every == 0:
                    self._write_checkpoint(i, len(targets))

        self._write_checkpoint(len(targets), len(targets))
        logger.info("done: %d new entries written to %s", new_count, self.output_path)
        return new_count

    def _write_checkpoint(self, done: int, total: int) -> None:
        self.checkpoint_path.write_text(
            json.dumps({"done": done, "total": total, "source": self.source}, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> BaseSpider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

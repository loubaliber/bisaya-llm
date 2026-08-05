"""Runtime robots.txt compliance.

Every request the crawler makes is checked against the target site's live
robots.txt before it is issued. This is a hard gate, independent of the
static findings recorded in docs/LEGAL_NOTICE.md -- robots.txt can change,
so we re-check (with a short in-memory cache) on every run.
"""

from __future__ import annotations

import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx


@dataclass
class RobotsCache:
    """Caches parsed robots.txt per host for the lifetime of a crawl run."""

    user_agent: str
    _parsers: dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)
    _fetched_at: dict[str, float] = field(default_factory=dict)
    ttl_seconds: float = 3600.0

    def _robots_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _get_parser(self, url: str, client: httpx.Client) -> urllib.robotparser.RobotFileParser:
        host = urlparse(url).netloc
        now = time.time()
        cached_age = now - self._fetched_at.get(host, 0.0)
        if host in self._parsers and cached_age < self.ttl_seconds:
            return self._parsers[host]

        robots_url = self._robots_url(url)
        parser = urllib.robotparser.RobotFileParser()
        try:
            resp = client.get(robots_url, timeout=10.0)
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
            else:
                # No robots.txt (or inaccessible): default-allow, per RFC convention,
                # but callers should still rate-limit and behave conservatively.
                parser.parse([])
        except httpx.HTTPError:
            # Network failure fetching robots.txt: be conservative and disallow.
            parser.parse(["User-agent: *", "Disallow: /"])

        self._parsers[host] = parser
        self._fetched_at[host] = now
        return parser

    def is_allowed(self, url: str, client: httpx.Client) -> bool:
        """Return True only if robots.txt explicitly or implicitly permits fetching `url`."""
        parser = self._get_parser(url, client)
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str, client: httpx.Client) -> float | None:
        parser = self._get_parser(url, client)
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay is not None else None

"""Spider for talkbisaya.com/dictionary.

Enabled by default at a conservative rate. Per docs/LEGAL_NOTICE.md, this
site's ToS still restricts use to "personal, non-commercial educational
purposes," so scraped output from this spider must not be pushed to a
public Hugging Face dataset without the permission gate in
huggingface/upload_dataset.py being satisfied.

Discovery strategy: the site's listing page is organized per starting
letter (`?letter=A`), and a bare `?page=N` walk over the unfiltered listing
silently truncates after the first couple of letters (this was the root
cause of the old dataset only covering A-B). We instead crawl each letter
A-Z independently, paginating within each letter until a page yields no new
links, so a slow/failed page can't hide the rest of the alphabet.

Discovery-level resume: `output/raw/talkbisaya.discovery_checkpoint.json`
records which letters have already been fully discovered, so an
interrupted `scrape` run can restart discovery from where it left off
instead of re-walking A-M again. Fetch-level resume (skipping already
downloaded entry pages) is handled by scraper/crawler.py:BaseSpider.run().
"""

from __future__ import annotations

import json
import re
import string
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from common import setup_logging
from scraper.crawler import BaseSpider

logger = setup_logging("scraper")

ALPHABET = list(string.ascii_uppercase)
MAX_PAGES_PER_LETTER = 100  # safety valve against an infinite pagination loop


class TalkBisayaSpider(BaseSpider):
    source = "talkbisaya"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.listing_url = urljoin(self.base_url, "/dictionary")
        self.discovery_checkpoint_path = (
            self.raw_output_dir / f"{self.source}.discovery_checkpoint.json"
        )

    # -- discovery-level resume -------------------------------------------------
    def _load_discovery_checkpoint(self) -> dict[str, list[list[str]]]:
        if self.discovery_checkpoint_path.exists():
            try:
                return json.loads(self.discovery_checkpoint_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("discovery checkpoint corrupt, starting fresh")
        return {"completed_letters": [], "urls": []}

    def _save_discovery_checkpoint(self, state: dict) -> None:
        self.discovery_checkpoint_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # -- per-letter pagination -----------------------------------------------
    def _discover_letter(self, letter: str) -> list[tuple[str, str]]:
        """Paginate a single letter's listing until a page adds no new links,
        or a fetch fails repeatedly (logged and treated as end-of-letter,
        not a crash of the whole run)."""
        found: dict[str, str] = {}
        page = 1
        consecutive_failures = 0

        while page <= MAX_PAGES_PER_LETTER:
            page_url = f"{self.listing_url}?letter={letter}&page={page}"
            if not self.robots.is_allowed(page_url, self.client):
                logger.warning("robots.txt disallows %s -- stopping letter '%s'", page_url, letter)
                break

            try:
                response = self._fetch(page_url)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - one bad page shouldn't kill the whole letter
                consecutive_failures += 1
                logger.error(
                    "failed to fetch %s: %s (attempt %d)", page_url, exc, consecutive_failures
                )
                if consecutive_failures >= 3:
                    logger.error("giving up on letter '%s' after repeated failures", letter)
                    break
                page += 1
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select('a[href^="/dictionary/"]')
            if not links:
                break

            new_this_page = 0
            for link in links:
                href = link.get("href", "")
                if not re.match(r"^/dictionary/[^/?#]+$", href):
                    continue
                word = link.get_text(strip=True) or href.rsplit("/", 1)[-1]
                absolute_url = urljoin(self.base_url, href)
                if absolute_url not in found:
                    found[absolute_url] = word
                    new_this_page += 1

            logger.info(
                "letter '%s' page %d: %d new entries (%d total)",
                letter,
                page,
                new_this_page,
                len(found),
            )
            if new_this_page == 0:
                break
            page += 1

        return [(word, url) for url, word in found.items()]

    def discover_entry_urls(self) -> list[tuple[str, str]]:
        state = self._load_discovery_checkpoint()
        completed = set(state["completed_letters"])
        all_urls: dict[str, str] = {url: word for url, word in state["urls"]}
        previous_letter_urlset: frozenset[str] | None = None

        for letter in ALPHABET:
            if letter in completed:
                logger.info("letter '%s' already discovered (checkpoint) -- skipping", letter)
                continue

            logger.info("discovering letter '%s'...", letter)
            letter_results = self._discover_letter(letter)
            letter_urlset = frozenset(url for _, url in letter_results)

            if previous_letter_urlset is not None and letter_urlset and (
                letter_urlset == previous_letter_urlset
            ):
                logger.warning(
                    "letter '%s' returned an IDENTICAL URL set to the previous letter -- "
                    "the '?letter=' query param may not be respected by this site's listing "
                    "endpoint. Check the actual pagination scheme in a browser and update "
                    "TalkBisayaSpider._discover_letter() accordingly.",
                    letter,
                )
            previous_letter_urlset = letter_urlset or previous_letter_urlset

            for word, url in letter_results:
                all_urls[url] = word
            logger.info("letter '%s' done: %d entries found", letter, len(letter_results))

            completed.add(letter)
            state = {
                "completed_letters": sorted(completed),
                "urls": [[url, word] for url, word in all_urls.items()],
            }
            self._save_discovery_checkpoint(state)

        logger.info("discovery complete: %d total entries across A-Z", len(all_urls))
        return [(word, url) for url, word in all_urls.items()]

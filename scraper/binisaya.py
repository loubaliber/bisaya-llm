"""Spider for binisaya.com.

DISABLED BY DEFAULT. Per docs/LEGAL_NOTICE.md, this site's own author states
its dictionary data wraps John U. Wolff's *A Dictionary of Cebuano Visayan*
(1972), a third-party copyrighted academic work. binisaya.com is not the
rights holder for that content, so this spider refuses to run unless the
caller explicitly overrides the config gate AND has confirmed permission
from the rights holder (see huggingface/upload_dataset.py's permission
gate for the corresponding publishing-side check).

The implementation is provided for completeness (e.g. for purely local,
non-redistributed research use once a researcher has made their own legal
assessment) but `run()` refuses by default.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from common import setup_logging
from scraper.crawler import BaseSpider

logger = setup_logging("scraper")


class BinisayaSpider(BaseSpider):
    source = "binisaya"

    def __init__(self, *, allow_disabled_source: bool = False, **kwargs: object) -> None:
        if not allow_disabled_source:
            raise RuntimeError(
                "BinisayaSpider is disabled by default -- see docs/LEGAL_NOTICE.md. "
                "Pass allow_disabled_source=True only after confirming you have the "
                "right to scrape and use this content (e.g. permission from the "
                "rights holder of Wolff's 1972 dictionary)."
            )
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.listing_url = urljoin(self.base_url, "/cebuano/")

    def discover_entry_urls(self) -> list[tuple[str, str]]:
        urls: dict[str, str] = {}
        if not self.robots.is_allowed(self.listing_url, self.client):
            logger.warning("robots.txt disallows %s", self.listing_url)
            return []
        try:
            response = self._fetch(self.listing_url)
        except Exception:  # noqa: BLE001
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.select('a[href^="/cebuano/"]'):
            href = link.get("href", "")
            if not re.match(r"^/cebuano/[^/?#]+$", href):
                continue
            word = link.get_text(strip=True) or href.rsplit("/", 1)[-1]
            absolute_url = urljoin(self.base_url, href)
            urls[absolute_url] = word

        return [(word, url) for url, word in urls.items()]

"""Simple per-host rate limiter with jitter, used to stay well under any
reasonable politeness threshold and to respect a site's Crawl-delay."""

from __future__ import annotations

import random
import time
from collections import defaultdict


class RateLimiter:
    """Token-bucket-ish limiter: guarantees at least `min_delay` seconds
    (plus jitter, up to `max_delay`) between consecutive requests to the
    same host."""

    def __init__(self, min_delay: float = 2.0, max_delay: float = 5.0) -> None:
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError("require 0 <= min_delay <= max_delay")
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_request_at: dict[str, float] = defaultdict(lambda: 0.0)

    def wait(self, host: str, override_delay: float | None = None) -> float:
        """Block (synchronously) until it is polite to hit `host` again.

        Returns the number of seconds actually slept, for logging.
        """
        floor_delay = override_delay if override_delay is not None else self.min_delay
        ceiling = max(floor_delay, self.max_delay)
        target_delay = random.uniform(floor_delay, ceiling)

        now = time.monotonic()
        elapsed = now - self._last_request_at[host]
        sleep_for = max(0.0, target_delay - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_request_at[host] = time.monotonic()
        return sleep_for

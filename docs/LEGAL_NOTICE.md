# Legal & Licensing Notice — READ BEFORE SCRAPING

This document records the Phase 1 website/legal analysis required before any
scraper in this repository is pointed at a live site. **The conclusion is
that public redistribution of the dictionary text itself (e.g. pushing raw
scraped entries to a public Hugging Face dataset) is legally risky and is
disabled by default.** The code is fully functional, but ships in a
"local-research-only" posture until a human confirms permission.

## Summary table

| Site | robots.txt / ToS status | Redistribution risk | Default posture |
|---|---|---|---|
| talkbisaya.com | ToS found and reviewed (see below) | **Restricted** — ToS limits use to "personal, non-commercial educational purposes" and explicitly forbids redistribution "for commercial purposes." It does not clearly authorize *public, non-commercial* redistribution (e.g. a public HF dataset), which is broader than "personal use." | Scrape allowed at low rate for local research; **HF dataset push blocked** without `--i-have-permission` + documented owner consent. |
| binisaya.com | No explicit ToS page located during this review | **High** — the site's own author states the dictionary data wraps **John U. Wolff's *A Dictionary of Cebuano Visayan* (1972)**, a copyrighted, professionally published academic work (PALI Language Texts / University Press of Hawaii). binisaya.com is not the rights holder, so even if the *site* were freely reused, the underlying Wolff text is third-party copyrighted content. | Scraping code provided for completeness but **disabled by default** (`enabled: false` in `configs/scraper.yaml`). Do not redistribute. |
| cebuano.pinoydictionary.com | No explicit ToS page located during this review | **High** — entries observed during review (headword + numbered senses + example sentences in the non-standard three-vowel orthography) match the structure and content of the same Wolff (1972) dictionary. Same concern as binisaya.com. | Disabled by default, same as above. |

**Robots.txt:** each site-specific spider in `scraper/robots.py` fetches and
parses `robots.txt` at *runtime* via `urllib.robotparser`/`httpx` before
issuing any request, and the crawler refuses to fetch any path disallowed
for `User-agent: *` (or a configured bot UA) at run time. This is a
programmatic, always-on check — treat this document's findings as a
starting point, not a substitute, since robots.txt can change.

## Why this matters for a Hugging Face dataset release

A dataset pushed to the Hugging Face Hub is, by construction, publicly
downloadable and redistributable by anyone — that is a different act from a
person personally browsing a dictionary page. Even "non-commercial" public
redistribution is broader than "personal ... use," and re-publishing a
professionally authored dictionary (Wolff 1972) without the publisher's
permission is a copyright concern independent of what any *scraped website*
says in its own terms, because the website itself may not hold the
redistribution rights.

## What this repository does about it

1. **`configs/scraper.yaml`** ships with `binisaya` and `pinoydictionary`
   sources `enabled: false`, and `talkbisaya` capped to a low request rate
   with `respect_robots_txt: true` and `personal_research_only: true`.
2. **`scraper/robots.py`** performs a live robots.txt check before every
   request and aborts on disallow.
3. **`huggingface/upload_dataset.py`** requires:
   - an explicit `--i-have-permission` CLI flag, **and**
   - a `PERMISSION_NOTES` field in `configs/dataset.yaml` describing what
     permission was obtained and from whom,
   before it will push anything beyond a `private=True` dataset repo.
4. The dataset card generator (`dataset/build_dataset.py`) automatically
   embeds a **Sources & Licensing** section listing provenance per-record so
   downstream users can evaluate their own risk.
5. **Recommended compliant alternatives** (see below) are documented so the
   pipeline can be fully exercised end-to-end without any legal ambiguity.

## Recommended compliant data sources (use these to unblock the pipeline)

- **Wiktionary Cebuano entries** — text is CC BY-SA 4.0 / GFDL, obtainable
  via the official Wiktionary XML/SQL dumps or the MediaWiki API, which is
  explicitly designed for bulk/automated reuse.
- **Your own vocabulary list** compiled by native-speaker contributors
  (`output/raw/manual/*.jsonl` — the parser accepts this format directly,
  bypassing the scraper entirely).
- **Direct permission** from TalkBisaya (contact form linked from their
  site) or from the rights holder of the Wolff (1972) dictionary
  (University of Hawaiʻi Press / Cornell University SEAP, historically) —
  once written permission is obtained, flip the relevant `enabled: true`
  flag and pass `--i-have-permission` with the correspondence referenced in
  `PERMISSION_NOTES`.
- **Public-domain Spanish-Cebuano historical dictionaries** (e.g. the 1711
  *Vocabulario de la lengua Bisaya*) if a public-domain digitization can be
  located — public domain status must still be verified per edition/scan,
  since a *new* critical edition or OCR layer can carry its own rights.

## Bottom line

> Build and ship the tooling (it is genuinely useful and reusable for any
> permissively licensed source). Do **not** flip the switch to publish the
> talkbisaya/binisaya/pinoydictionary text as a public HF dataset without
> written permission from the rights holders — this repo enforces that with
> a hard gate, not just a comment.

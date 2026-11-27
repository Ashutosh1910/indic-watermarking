# Hindi Samay prose (non-canonical, pre-2022)

1,696 Hindi prose texts scraped from [hindisamay.com](https://www.hindisamay.com/) (Mahatma Gandhi
Antarrashtriya Hindi Vishwavidyalaya) on 2026-11-22. Same file format as `../Samalochan`:
title, blank line, author, blank line, then paragraphs separated by blank lines.
Files are named `<hindisamay content id>_<title>.txt`; `metadata.csv` has one row per file
(id, title, author, genre, word count, first Wayback capture, source URL).

## Selection

1. **Online before 2022-01-01.** Only texts whose page the Internet Archive captured on or before
   2021-12-31 (Wayback CDX, HTTP 200). `wayback_first_capture` is that capture's timestamp; the
   latest in the set is 2021-10-24. Hindi Samay shows no per-text dates, so this is an upper bound:
   many texts are reprints of older print work.
2. **Non-canonical authors**, to limit LLM memorisation. Excluded by author name: pre-Independence
   and classic Hindi/Urdu writers (Premchand, Prasad, Nirala, Mahadevi, Agyeya, Ugra, Vidyarthi …),
   the major 20th-century canon (Nai Kahani writers, Parsai, Renu, Nirmal Verma, Matiyani, Bedi …),
   famous public figures (Gandhi, Nehru, Phule …), and world-famous authors in translation
   (Tolstoy, Márquez, Murakami, Nobel laureates …). Living contemporary writers are kept even when
   well known. Full lists: `CANON` and `CANON_EXTRA` in the scraper/filter scripts.
3. **At least 500 words** (whitespace tokens of the body). Mean 4,364, median 3,369, max 36,174.
4. **Not purely factual.** Dropped the site's research-paper (शोध) and biography (जीवनी) genres,
   non-fiction with >4% number tokens or ≥40 citation/footnote markers, and titles that mark
   chronologies, reports or event write-ups. Argumentative criticism and essays are kept.

Also dropped: poetry, novels, plays, children's literature, folk tales, multi-page texts, exact
and near duplicates, and 7 pages that failed to fetch.

| Genre | Texts |
|---|---|
| कहानी (story) | 1,035 |
| आलोचना (criticism) | 182 |
| लेख (article) | 129 |
| सिनेमा (cinema) | 71 |
| विमर्श (discourse) | 67 |
| निबंध (essay) | 58 |
| व्यंग्य (satire) | 53 |
| संस्मरण (memoir) | 40 |
| other (travel, reflections, diary, interviews, short-short stories, lectures, letters) | 61 |

## Copyright

Hindi Samay publishes with each author's or publisher's permission; copyright stays with the
authors. Use for non-commercial research only and do not redistribute the texts publicly.

# Medium stories and essays (English, pre-2022)

3,992 English prose texts (1,143 stories, 2,849 non-factual essays) filtered from the
[`fabiochiu/medium-articles`](https://huggingface.co/datasets/fabiochiu/medium-articles) dump
(192,368 Medium articles; `medium_articles.csv`, repo commit `979af3b`, sha256
`bba7b102…a7cb12a5c`). It is meant as the English counterpart to `../HindiSamay` and
`../Samalochan`, and uses the same file format: title, blank line, author(s), blank line, then
paragraphs separated by blank lines. Files are named `<publish-date>_<title-slug>.txt`;
`metadata.csv` has one row per file (title, authors, genre, word count, publish time, tags, URL).
`filter_medium.py` reproduces the set from the CSV:
`python -I filter_medium.py medium_articles.csv <out_dir>`.

## Selection

1. **Published before 2022-01-01** (Medium's own timestamp; 2014–2021, 61% from 2020).
2. **Genre by Medium tag.**
   - *Story*: tagged fiction, short story, flash/short/micro fiction, literary or speculative
     fiction, fable or fairy tale, and not tagged as a book review or book list.
   - *Essay*: tagged essay, personal essay, memoir, (creative) nonfiction, satire, humor, comedy,
     literature, books, book or film review, film, movies, cinema, philosophy, culture, art,
     travel writing, letters, diary or reflections.
3. **Dropped as not literary prose**: poetry and free verse, screenplays, erotica, comics,
   writing-craft advice, serials (chapters, parts, episodes, "previously on", series tags),
   publication announcements and submission calls, worldbuilding notes, and e-book spam.
4. **Dropped as factual, topical or self-help** (any tag in or containing): technology,
   programming, data, AI, business, startups, marketing, finance, investing, crypto, careers,
   workplace, leadership, productivity, self-improvement, personal development, health and mental
   health, COVID, science, psychology, politics, elections, news, education, fitness, food,
   travel (not travel writing), history, religion, parenting, sports, gaming. Stories are only
   dropped for the tech/business/self-help part of this list.
5. **More than 500 words** of body text after cleaning. Mean 1,251, median 974, max 17,310.
6. **Same quality rules as Hindi Samay**: no listicle or how-to titles (essays only), ≤2
   numbered-list lines, <6 bullet lines, ≤4% number tokens, <15 links or citation markers, no
   poem-shaped texts (most lines under 8 words), no code, Latin script only. Exact duplicates
   and repeated title + author pairs were removed.

**Cleaning:** removed a leading line that repeats the title, image captions and credits
("Photo by…", Unsplash, Canva…), and trailing calls to action ("Follow me", "Originally
published at", "Thanks for reading"…).

| Genre | Texts | Median words | Median Qwen3 tokens |
|---|---|---|---|
| story | 1,143 | 1,029 | 1,304 |
| essay | 2,849 | 951 | 1,211 |

## Known limitations

- Genres come from authors' own tags, so a few personal essays are labelled as stories and
  some reviews or opinion pieces sit among the essays. Random spot-checks of ~170 texts found
  few clear misfits after the rules above.
- Self-published and more conversational than the editor-selected Hindi texts; median length is
  about 1,000 words against 3,391 for Samalochan + HindiSamay.
- No canonical-author filter: Medium writers are almost all non-canonical, but reposts of
  well-known work may remain.

## Copyright

The Hugging Face uploader labels the dump MIT, but copyright stays with each Medium author.
Use for non-commercial research only and do not redistribute the texts publicly.

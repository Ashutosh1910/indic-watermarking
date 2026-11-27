"""Filter fabiochiu/medium-articles down to English stories and non-factual essays,
>500 words, published before 2022-01-01. Mirrors the HindiSamay selection rules.

usage: python -I medium_filter.py <medium_articles.csv> <out_dir> [--dry]
"""
import ast, collections, csv, hashlib, os, re, sys, unicodedata

csv.field_size_limit(10**9)
sys.stdout.reconfigure(encoding="utf-8")
src, out = sys.argv[1], sys.argv[2]
DRY = "--dry" in sys.argv

# ---- genre by tag ---------------------------------------------------------
STORY = {"fiction", "short story", "short stories", "short fiction", "flash fiction",
         "microfiction", "micro fiction", "literary fiction", "speculative fiction",
         "fable", "fables", "fairy tale", "fairy tales", "fiction short story"}
ESSAY = {"essay", "essays", "personal essay", "personal essays", "memoir", "creative nonfiction",
         "nonfiction", "non fiction", "satire", "humor", "humour", "comedy", "literature",
         "literary criticism", "book review", "book reviews", "books", "reading", "film",
         "film criticism", "film review", "movies", "movie review", "cinema", "philosophy",
         "culture", "art", "travel writing", "letters", "open letter", "diary", "reflections"}
# never poetry / scripts / erotica / craft advice / serials (HindiSamay dropped poetry, plays, novels)
BAN_ALL = {"poetry", "poem", "poems", "free verse", "haiku", "poetry on medium", "prose poetry",
           "screenplay", "screenwriting", "erotica", "sex", "writing tips", "writing advice",
           "fiction writing", "writing craft", "how to write", "novel", "serial", "web serial",
           "50 word stories", "100 word story", "six word story", "comics", "cartoon", "comic"}
# factual / topical / self-help topics: not literary prose
BAN_TOPIC = {"programming", "technology", "tech", "data science", "machine learning",
             "artificial intelligence", "ai", "javascript", "python", "software development",
             "software engineering", "web development", "coding", "startup", "startups", "business",
             "marketing", "digital marketing", "entrepreneurship", "finance", "investing", "money",
             "personal finance", "cryptocurrency", "crypto", "blockchain", "bitcoin", "ethereum",
             "defi", "nft", "health", "mental health", "covid 19", "covid-19", "covid", "coronavirus",
             "pandemic", "science", "productivity", "self improvement", "self-improvement",
             "motivation", "leadership", "management", "design", "ux", "ui", "seo", "social media",
             "news", "politics", "election", "economics", "education", "careers", "career", "jobs",
             "freelancing", "fitness", "nutrition", "diet", "weight loss", "medicine", "healthcare",
             "psychology", "neuroscience", "biotechnology", "climate change", "sports", "nba", "nfl",
             "soccer", "gaming", "video games", "travel", "food", "recipes", "cooking", "real estate",
             "history", "religion", "spirituality", "parenting", "life lessons", "self",
             "success", "inspiration", "advice", "tips", "how to", "relationships advice"}
BAN_ALL |= {"series", "worldbuilding", "publication", "publications", "submission guidelines",
            "submissions", "call for submissions", "announcement", "editors picks", "newsletter"}
# substrings of tags that mark corporate / factual / self-help pieces (both genres)
BAN_SUB = ("entrepreneur", "startup", "business", "marketing", "technolog", "programming", "crypto",
           "invest", "financ", "career", "workplace", "leadership", "management", "productivity",
           "self help", "self-help", "personal development", "personal growth", "software", "customer",
           "branding", "tourism", "sustainab", "teamwork", "data ", "coding", "covid tips", "how to",
           "politic", "election", "presidential", "news", "retail", "ebook", "download")
BAN_TITLE = re.compile(r"submission|guidelines|editor'?s'? (picks|note)|announc|welcome to|newsletter|"
                       r"call for|we are (hiring|looking)|join us|weekly (digest|roundup)|round-?up|"
                       r"episode|#\w*ep(isode)?\d|#\s*\d+|\bconclusion\b|\(cont|continued\b|"
                       r"\[(epub|pdf|mobi|download)|\b(pdf|epub|ebook) download|download (pdf|epub|ebook)", re.I)
SERIAL_OPEN = re.compile(r"\b(this is )?part (\d+|one|two|three|four|five|[ivx]+)\b|\bchapter \d+|"
                         r"\b(previously|last time) on\b|\bcatch up on (the )?(previous|earlier)|"
                         r"\b(read|see) (the )?(previous|first|last) (part|chapter|episode)|\bepisode \d+", re.I)
REVIEWISH = {"book review", "book reviews", "books", "reading", "book recommendations"}

def tags_of(s):
    try:
        return [t.strip().lower() for t in ast.literal_eval(s)]
    except Exception:
        return []

# ---- text cleaning --------------------------------------------------------
CAPTION = re.compile(r"^\s*(photo|image|illustration|picture|gif|video|art(work)?|graphic|cover|header)"
                     r"( image| photo)?\s*(by|from|via|courtesy|credit|source)\b|^\s*(source|credit|image source)\s*:"
                     r"|unsplash|pixabay|pexels|shutterstock|getty images|canva", re.I)
CTA = re.compile(r"^\s*(originally published|this (story|article|piece|post) (was|is) (originally|first)|"
                 r"follow (me|us)|sign up|subscribe|join (my|our)|if you (liked|enjoyed)|thanks for reading|"
                 r"thank you for reading|clap|you can (find|read|follow)|read more|buy me a coffee|"
                 r"get (my|our|the) (free|newsletter)|my (newsletter|book|website)|check out my|"
                 r"want to (read|write)|written by|more from|about the author|©|copyright)", re.I)

def clean(title, text):
    lines = [l.strip() for l in text.replace("\r", "").split("\n")]
    lines = [l for l in lines if l]
    if lines and lines[0].strip().lower() == title.strip().lower():
        lines = lines[1:]
    lines = [l for l in lines if not CAPTION.search(l) or len(l.split()) > 25]
    while lines and (CTA.search(lines[-1]) or len(lines[-1].split()) <= 3):   # trailing CTAs / sign-offs
        lines.pop()
    return lines

# ---- quality rules (HindiSamay-style) -------------------------------------
LISTICLE_TITLE = re.compile(r"^\s*\d+\s|\b(how to|ways to|tips|reasons|things (i|you|we)|guide|"
                            r"steps|hacks|lessons|rules|habits|signs|best|top \d+|why you should|"
                            r"you need|what i learned|here'?s (how|why|what))\b", re.I)
SERIAL_TITLE = re.compile(r"\b(chapter|ch\.|part|pt\.|episode|ep\.|book|vol\.?|volume)\s*([0-9]+|[ivxlc]+|one|two|"
                          r"three|four|five|six|seven|eight|nine|ten)\b|\(\s*\d+\s*/\s*\d+\s*\)|\bfinale\b", re.I)
NUMLINE = re.compile(r"^\s*(\d+[.):]|#\d+|step \d+|tip \d+)\s")
BULLET = re.compile(r"^\s*[•\-\*▪●►✓✔]\s")
NUMTOK = re.compile(r"\d")
LINK = re.compile(r"https?://|www\.|\[\d+\]")

def reject_reason(genre, title, lines, words):
    if words <= 500: return "<=500 words"
    if SERIAL_TITLE.search(title) or SERIAL_OPEN.search(" ".join(lines)[:400]): return "serial/chapter"
    if BAN_TITLE.search(title): return "announcement/series title"
    if genre == "essay" and LISTICLE_TITLE.search(title): return "listicle/how-to title"
    nl = len(lines)
    if sum(bool(NUMLINE.match(l)) for l in lines) >= 3: return "numbered list"
    if sum(bool(BULLET.match(l)) for l in lines) >= 6: return "bullets"
    toks = " ".join(lines).split()
    if sum(bool(NUMTOK.search(t)) for t in toks) / max(1, len(toks)) > 0.04: return "numbers >4%"
    if len(LINK.findall(" ".join(lines))) >= 15: return "links/citations"
    short = sum(len(l.split()) < 8 for l in lines)
    if nl >= 15 and short / nl > 0.5 and genre == "essay": return "poem-like lines"
    if nl >= 15 and short / nl > 0.6: return "poem-like lines"
    body = " ".join(lines)
    letters = [c for c in body if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.97: return "non-Latin script"
    if re.search(r"[{};]\s*$|def |function\s*\(|console\.log|import \w+", body[:20000]) and body.count(";") > 30:
        return "code"
    return None

def slug(s, n=40):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:n].strip("-") or "untitled"

# ---- main -----------------------------------------------------------------
stats = collections.Counter()
seen_text, seen_ta, kept = set(), set(), []
with open(src, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        stats["total"] += 1
        ts = r["timestamp"]
        # Medium launched in Aug 2012, so earlier timestamps are bogus
        if not re.match(r"\d{4}-", ts) or ts[:10] >= "2022-01-01" or ts[:10] < "2012-08-01":
            stats["drop: date >= 2022, before Medium existed, or missing"] += 1; continue
        tg = set(tags_of(r["tags"]))
        if tg & BAN_ALL:
            stats["drop: poetry/script/serial/craft tag"] += 1; continue
        if any(s in t + " " for t in tg for s in BAN_SUB):
            stats["drop: corporate/factual/self-help tag (substring)"] += 1; continue
        if tg & STORY and not tg & REVIEWISH:
            genre = "story"
            if tg & (BAN_TOPIC - {"travel", "history", "parenting", "self", "inspiration", "life lessons",
                                  "psychology", "mental health", "religion", "spirituality", "pandemic",
                                  "covid 19", "covid-19", "covid", "coronavirus", "relationships advice"}):
                stats["drop: story with tech/business/self-help tag"] += 1; continue
        elif tg & ESSAY:
            genre = "essay"
            if tg & BAN_TOPIC:
                stats["drop: essay with factual/self-help topic tag"] += 1; continue
        else:
            stats["drop: not story/essay tag"] += 1; continue
        title = r["title"].strip()
        lines = clean(title, r["text"])
        words = sum(len(l.split()) for l in lines)
        why = reject_reason(genre, title, lines, words)
        if why:
            stats[f"drop: {why}"] += 1; continue
        body = "\n".join(lines)
        h = hashlib.md5(re.sub(r"\W+", "", body.lower())[:2000].encode()).hexdigest()
        try:
            authors = [a.strip() for a in ast.literal_eval(r["authors"]) if a.strip()] or ["unknown"]
        except Exception:
            authors = ["unknown"]
        ta = (title.lower(), authors[0].lower())
        if h in seen_text or ta in seen_ta:
            stats["drop: duplicate"] += 1; continue
        seen_text.add(h); seen_ta.add(ta)
        stats[f"KEEP {genre}"] += 1
        kept.append(dict(genre=genre, title=title, authors=authors, timestamp=ts, words=words,
                         tags=sorted(tg), url=r["url"], lines=lines))

for k, v in sorted(stats.items(), key=lambda x: (not x[0].startswith(("total", "KEEP")), -x[1])):
    print(f"{v:8d}  {k}")

if not DRY:
    os.makedirs(out, exist_ok=True)
    used = set()
    with open(os.path.join(out, "metadata.csv"), "w", encoding="utf-8", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["file", "title", "authors", "genre", "words", "published", "tags", "source_url"])
        for d in sorted(kept, key=lambda d: d["timestamp"]):
            fn = f"{d['timestamp'][:10]}_{slug(d['title'])}.txt"
            i = 2
            while fn in used:
                fn = f"{d['timestamp'][:10]}_{slug(d['title'])}-{i}.txt"; i += 1
            used.add(fn)
            with open(os.path.join(out, fn), "w", encoding="utf-8", newline="\n") as tf:
                tf.write(d["title"] + "\n\n" + ", ".join(d["authors"]) + "\n\n" + "\n\n".join(d["lines"]) + "\n")
            w.writerow([fn, d["title"], "; ".join(d["authors"]), d["genre"], d["words"], d["timestamp"][:19],
                        "; ".join(d["tags"]), d["url"]])
    print("wrote", len(kept), "files to", out)

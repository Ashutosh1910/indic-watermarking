"""Rule-based filter: raw source rows -> long-form, non-factual, self-contained candidate prompts.

Step 2 of the prompt-set pipeline. Reads <work>/raw/<source>.jsonl (from download_sources.py) and writes

    <work>/candidates/<source>.jsonl     kept rows, with rule_genre, n_words and the cleaned text
    <work>/candidates/drops.json         per source, how many rows each rule dropped (first rule that fired)

The rules follow the plan agreed for the study: keep requests for stories, essays, arguments, letters,
speeches, diary entries and personal reflection; drop factual questions, code/math, lists, poems and
other short forms, prompts that cap the length, and prompts that need a text or image we don't give.
An LLM pass (label_prompts.md, run by hand or by agents) then checks the candidates one by one.

    python filter_prompts.py --work ../work
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("​", "").replace("﻿", "")
    s = re.sub(r"[ \t ]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def nwords(s: str) -> int:
    return len(s.split())


def deva_share(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if DEVA.match(c)) / len(letters)


def rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.IGNORECASE)


# ------------------------------------------------------------------------------------- English rules

EN_INTENT = rx(r"\b(write|compose|draft|create|craft|tell|narrate|imagine|pen)\b")
EN_FORM = rx(r"\bstor(y|ies)\b", r"\btale\b", r"\bfable\b", r"\bnarrative\b", r"\bfiction\b", r"\bscene\b",
             r"\bchapter\b", r"\bessay\b", r"\barticle\b", r"\bblog( post)?\b", r"\bletter\b", r"\bspeech\b",
             r"\bdiary\b", r"\bjournal entry\b", r"\bmemoir\b", r"\bmonologue\b", r"\bop-?ed\b", r"\beditorial\b",
             r"\breflection\b", r"\bpersonal statement\b", r"\bnovel\b", r"\bparable\b", r"\banecdote\b",
             r"\bfirst[- ]person account\b", r"\bdialogue\b", r"\bcolumn\b", r"\bsermon\b", r"\beulogy\b")
EN_LENGTH_CAP = rx(r"\b\d[\d,]*\s*(-|to|–)?\s*\d*\s*(words?|sentences?|paragraphs?|lines?|pages?|characters?|chars|tokens|stanzas?|verses?)\b",
                   r"\b(one|two|three|four|five|a single|single|a short|short|brief|a brief|one-)\s*(sentence|paragraph|line)s?\b",
                   r"\bbriefly\b", r"\bin short\b", r"\bin brief\b", r"\bconcise(ly)?\b", r"\bshort and sweet\b",
                   r"\bunder \d+", r"\bless than \d+", r"\bno more than\b", r"\bat most \d+", r"\bmax(imum)? (of )?\d+",
                   r"\bword count\b", r"\bword limit\b", r"\b(one|two|three)[- ]liner\b")
EN_SHORT_FORM = rx(r"\bhaikus?\b", r"\blimericks?\b", r"\bpoems?\b", r"\bpoetry\b", r"\bsonnets?\b", r"\bsongs?\b",
                   r"\blyrics\b", r"\brap\b", r"\bverse\b", r"\bballad\b", r"\btweets?\b", r"\bcaptions?\b",
                   r"\bslogans?\b", r"\btaglines?\b", r"\bheadlines?\b", r"\btitles? for\b", r"\bnames? for\b",
                   r"\bjokes?\b", r"\briddles?\b", r"\bquotes?\b", r"\bbio\b", r"\bsummar(y|ise|ize)\b",
                   r"\btl;?dr\b", r"\boutline\b", r"\bbullet", r"\blist\b", r"\btop \d+\b",
                   r"\b\d+ (ideas|ways|tips|reasons|examples|things|names|questions|steps)\b",
                   r"\bemail\b", r"\be-mail\b", r"\bsubject line\b", r"\bcover letter\b", r"\bresume\b", r"\bcv\b",
                   r"\bproduct description\b", r"\bad copy\b", r"\badvert", r"\bscript for (a|an) (ad|commercial)\b",
                   r"\bmeta description\b", r"\bseo\b", r"\bkeywords?\b", r"\bhashtags?\b", r"\bmcq\b",
                   r"\bmultiple choice\b", r"\bquiz\b", r"\btable\b", r"\bcode\b", r"\bpython\b", r"\bjavascript\b",
                   r"\bsql\b", r"\bhtml\b", r"\bjson\b", r"\bfunction\b", r"\bprompt for (midjourney|stable diffusion|dall)",
                   r"\bmidjourney\b", r"\bscreenplay\b", r"\bplay script\b", r"\bscript\b", r"\bsitcom\b",
                   r"\bepisode\b", r"\bpitch\b", r"\bproposal\b", r"\binvitation\b", r"\bthank[- ]you note\b",
                   r"\bwedding toast\b", r"\btoast\b", r"\bcard message\b", r"\bgreeting card\b", r"\bobituary\b")
EN_FACTUAL = rx(r"^\s*(who|what|when|where|which|why|how|is|are|do|does|did|can|could|should i|will|was|were)\b(?!.*\b(write|story|essay|letter)\b)",
                r"\bexplain\b", r"\bdefin(e|ition)\b", r"\bwhat is\b", r"\bhistory of\b", r"\bbiography\b",
                r"\bfacts?\b", r"\bstatistics?\b", r"\bcite\b", r"\bcitations?\b", r"\breferences?\b", r"\bsources\b",
                r"\bresearch paper\b", r"\bliterature review\b", r"\bapa\b", r"\bmla\b", r"\bharvard style\b",
                r"\bcalculate\b", r"\bsolve\b", r"\bequation\b", r"\btranslat", r"\bcompare and contrast\b",
                r"\bcase study\b", r"\breport on\b", r"\banaly[sz]e\b", r"\banalysis\b", r"\bnews (article|report)\b",
                r"\bpress release\b", r"\bwikipedia\b", r"\bstep[- ]by[- ]step\b", r"\bhow to\b", r"\brecipe\b",
                r"\breview of (the|a) (book|film|movie|product)\b", r"\bmarketing\b", r"\bbusiness plan\b",
                r"\bthesis\b", r"\bdissertation\b", r"\bassignment\b", r"\bhomework\b")
EN_NEEDS_INPUT = rx(r"\b(the|this) (following|text|passage|paragraph|article|essay|story|poem|email|document|"
                    r"excerpt|transcript|content|draft)\b(?! (about|on|where|in which|that))",
                    r"\bbelow\b", r"\babove\b", r"\battached\b", r"\bgiven (text|passage|article)\b", r"\brewrite\b",
                    r"\bparaphrase\b", r"\bproofread\b", r"\bedit (this|my|the)\b", r"\bpolish\b", r"\bimprove (this|my)\b",
                    r"\bcontinue\b", r"\bexpand on\b", r"\bfix (the )?grammar\b", r"\bcorrect (the|my|this)\b",
                    r"\bsummari[sz]e\b", r"\bimage\b", r"\bpicture\b", r"\bphoto\b", r"\bvideo\b", r"https?://", r"www\.")
EN_NSFW = rx(r"\bsex", r"\bnsfw\b", r"\berotic", r"\bnude", r"\bnaked\b", r"\bfetish", r"\bporn", r"\bhorny\b",
             r"\bbreasts?\b", r"\bpenis\b", r"\bvagina\b", r"\borgasm", r"\brape\b", r"\bincest\b", r"\bsmut\b",
             r"\blewd\b", r"\bfeet\b", r"\bfoot (fetish|worship)\b", r"\bdiaper", r"\bvore\b", r"\bfart", r"\bpee\b",
             r"\bpiss", r"\bunderwear\b", r"\bpanties\b", r"\bbikini\b", r"\bkinky\b", r"\bbdsm\b", r"\bslave\b",
             r"\btorture\b", r"\bgore\b", r"\bsuicide\b", r"\bself[- ]harm\b", r"\bpedo", r"\bloli\b", r"\bminor\b")
EN_FANDOM = rx(r"\bfan ?fic", r"\bfreedom planet\b", r"\bnaruto\b", r"\bsonic\b", r"\bpokemon\b", r"\bpokémon\b",
               r"\bminecraft\b", r"\bdoki doki\b", r"\bmonster girl", r"\bharry potter\b", r"\bmarvel\b", r"\bdc comics\b",
               r"\bstar wars\b", r"\bgame of thrones\b", r"\bwarhammer\b", r"\bgenshin\b", r"\bdragon ball\b",
               r"\banime\b", r"\bmanga\b", r"\bjojo\b", r"\bfnaf\b", r"\bfive nights\b", r"\bmy little pony\b",
               r"\bhazbin\b", r"\bundertale\b", r"\bzelda\b", r"\bmario\b", r"\bavatar\b", r"\bhighschool dxd\b",
               r"\bsenpai\b", r"\bwaifu\b", r"\bkawaii\b", r"\bstardew\b", r"\bterraria\b", r"\bwattpad\b")

# ------------------------------------------------------------------------------------- Hindi rules

HI_INTENT = rx(r"लिख", r"रचना कर", r"रचिए", r"रचें", r"बनाइ", r"बनाएं", r"बनाएँ", r"बनाओ", r"बना सकते", r"गढ़", r"कल्पना कीजिए",
               r"कल्पना करें", r"तैयार कीजिए", r"तैयार करें", r"सुनाइ", r"सुनाओ", r"सुनाएँ", r"सुनाएं", r"वर्णन कीजिए",
               r"वर्णन करें", r"वर्णन करो", r"\bwrite\b", r"\blikh", r"\bkahani\b", r"\bnibandh\b")
HI_FORM = rx(r"कहानी", r"कहानियां", r"कहानियाँ", r"कथा", r"निबंध", r"लेख", r"पत्र", r"चिट्ठी", r"भाषण", r"डायरी",
             r"दैनंदिनी", r"संस्मरण", r"आत्मकथा", r"यात्रा[ -]?वृत्तांत", r"यात्रा वर्णन", r"संवाद", r"व्यंग्य",
             r"ब्लॉग", r"आलेख", r"अपने विचार", r"अपनी राय", r"विचार व्यक्त", r"वर्णन", r"फ़साना", r"किस्सा", r"क़िस्सा", r"उपन्यास", r"अध्याय", r"संपादकीय", r"\bstory\b",
             r"\bessay\b", r"\bkahani\b", r"\bnibandh\b", r"\bletter\b", r"\bspeech\b")
HI_LENGTH_CAP = rx(r"\d+\s*(-|से|–)?\s*\d*\s*(शब्द|वाक्य|पंक्ति|पंक्तियों|लाइन|अनुच्छेद|पैराग्राफ|पृष्ठ|पेज)",
                   r"[०-९]+\s*(शब्द|वाक्य|पंक्ति)", r"(एक|दो|तीन|चार|पांच|पाँच|कुछ|चंद)\s*(शब्द|वाक्य|पंक्ति|पंक्तियों|लाइन|अनुच्छेद)",
                   r"संक्षेप", r"संक्षिप्त", r"छोटा सा", r"छोटी सी", r"छोटा-सा", r"छोटी-सी", r"छोटे से", r"अनुच्छेद",
                   r"\bwords?\b", r"\bshort\b", r"\bbrief")
HI_SHORT_FORM = rx(r"कविता", r"कविताएं", r"कविताएँ", r"दोहा", r"दोहे", r"शायरी", r"शेर", r"ग़ज़ल", r"गजल", r"गीत",
                   r"छंद", r"मुक्तक", r"हाइकु", r"नारा", r"नारे", r"स्लोगन", r"शीर्षक", r"नाम सुझा", r"चुटकुल",
                   r"पहेली", r"सूची", r"बिंदु", r"मुहावर", r"लोकोक्ति", r"उद्धरण", r"सारांश", r"ईमेल", r"ई-मेल",
                   r"विज्ञापन", r"ट्वीट", r"कैप्शन", r"टैगलाइन", r"\bpoem", r"\bkavita", r"\bshayari", r"\bsong",
                   r"आवेदन पत्र", r"प्रार्थना पत्र", r"प्रार्थना-पत्र", r"अवकाश", r"छुट्टी के लिए", r"शिकायत पत्र",
                   r"बायोडाटा", r"पटकथा", r"स्क्रिप्ट", r"कथानक बिंदु", r"\bresume\b", r"\bemail\b", r"\bcode\b", r"प्रश्न[ो]?त्तर", r"बहुविकल्प")
# Kept deliberately narrow: the LLM check (LABELING.md) reads every Hindi candidate anyway, and the
# Hindi pool is small, so a broad keyword list would throw away good story prompts.
HI_FACTUAL = rx(r"क्या है", r"क्या हैं", r"क्या था", r"कौन (है|था|थे|थी)", r"किसने", r"कितन", r"परिभाषा", r"अर्थ बताइए",
                r"अर्थ क्या", r"इतिहास", r"जीवनी", r"जीवन परिचय", r"आंकड़", r"आँकड़", r"अनुवाद", r"व्याख्या कीजिए",
                r"समझाइए", r"गणना", r"हल कीजिए", r"समीकरण", r"विश्लेषण", r"रेसिपी", r"नुस्खा", r"समाचार", r"रिपोर्ट",
                r"प्रेस विज्ञप्ति", r"सिफारिश", r"सुझाव दीजिए", r"मदद कर सकते", r"मदद चाहिए", r"आईडिया दे")
HI_NEEDS_INPUT = rx(r"निम्नलिखित", r"नीचे दि", r"ऊपर दि", r"उपरोक्त", r"इस (पाठ|अंश|लेख|कहानी|कविता|अनुच्छेद|वाक्य)",
                    r"दिए गए", r"दी गई", r"दिये गये", r"दोबारा लिख", r"फिर से लिख", r"पुनः लिख", r"सुधार", r"संपादित",
                    r"आगे बढ़ा", r"पूरा कीजिए", r"पूरी कीजिए", r"चित्र", r"तस्वीर", r"वीडियो", r"https?://", r"www\.")
HI_NSFW = rx(r"सेक्स", r"संभोग", r"नग्न", r"अश्लील", r"बलात्कार", r"आत्महत्या", r"\bsex", r"\bnsfw\b", r"\bporn")

# ------------------------------------------------------------------------------------- genres

GENRE_RULES = [
    ("personal", rx(r"\bletter\b", r"\bdiary\b", r"\bjournal entry\b", r"\bmemoir\b", r"\bpersonal (essay|narrative|reflection)\b",
                    r"\babout (a time|your|my) ", r"\bfirst[- ]person account\b", r"\bautobiograph", r"\beulogy\b",
                    r"पत्र", r"चिट्ठी", r"डायरी", r"दैनंदिनी", r"संस्मरण", r"आत्मकथा", r"यात्रा", r"अपने (जीवन|अनुभव|बचपन)")),
    ("opinion", rx(r"\bpersuasive\b", r"\bpersuade\b", r"\bargu(e|ment|mentative)\b", r"\bopinion\b", r"\bdebate\b",
                   r"\bconvince\b", r"\bfor or against\b", r"\bdo you (agree|think)\b", r"\bop-?ed\b", r"\beditorial\b",
                   r"\bshould\b", r"\bposition\b", r"\bpros and cons\b",
                   r"पक्ष", r"विपक्ष", r"तर्क", r"राय", r"आपके विचार", r"अपने विचार", r"विचार व्यक्त", r"सहमत", r"चाहिए",
                   r"समर्थन", r"विरोध")),
    ("story", rx(r"\bstor(y|ies)\b", r"\btale\b", r"\bfable\b", r"\bfiction\b", r"\bnarrative\b", r"\bscene\b", r"\bchapter\b",
                 r"\bparable\b", r"\bnovel\b", r"\bmonologue\b",
                 r"कहानी", r"कहानियां", r"कहानियाँ", r"कथा", r"किस्सा", r"क़िस्सा", r"उपन्यास", r"अध्याय")),
    ("essay", rx(r"\bessay\b", r"\barticle\b", r"\bblog\b", r"\breflection\b", r"\bspeech\b", r"\bcolumn\b", r"\bsermon\b",
                 r"निबंध", r"लेख", r"आलेख", r"भाषण", r"ब्लॉग", r"व्यंग्य")),
]


def rule_genre(text: str, default: str | None = None) -> str | None:
    for g, pat in GENRE_RULES:
        if pat.search(text):
            return g
    return default


# ------------------------------------------------------------------------------------- per source

WP_TAG = re.compile(r"^\s*[\[\(]\s*([A-Za-z]{2,3})\s*[\]\)]\s*")
WP_KEEP_TAGS = {"WP", "RF", "TT", "PI", "SP"}  # writing prompt, reality fiction, theme thursday, prompt inspired, simple prompt


def clean_wp(text: str) -> tuple[str, list[str]]:
    """Strip leading tags like '[ WP ]' (possibly several) and Reddit artifacts."""
    tags = []
    t = text
    while True:
        m = WP_TAG.match(t)
        if not m:
            break
        tags.append(m.group(1).upper())
        t = t[m.end():]
    t = t.replace("``", '"').replace("''", '"').replace(" n't", "n't")
    t = re.sub(r" '(s|ve|re|ll|d|m)\b", r"'\1", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', t)
    t = re.sub(r"<newline>", " ", t)
    return norm(t), tags


def check_en(text: str, require_intent: bool, min_w: int, max_w: int) -> str | None:
    """Return the name of the first rule that drops the text, or None to keep it."""
    n = nwords(text)
    if n < min_w:
        return "too_short"
    if n > max_w:
        return "too_long"
    if DEVA.search(text):
        return "script"
    if EN_NSFW.search(text):
        return "nsfw"
    if EN_NEEDS_INPUT.search(text):
        return "needs_input"
    if EN_LENGTH_CAP.search(text):
        return "length_cap"
    if EN_SHORT_FORM.search(text):
        return "short_form"
    if EN_FANDOM.search(text):
        return "fandom"
    if require_intent and not (EN_INTENT.search(text) and EN_FORM.search(text)):
        return "no_writing_intent"
    if EN_FACTUAL.search(text):
        return "factual"
    return None


def check_hi(text: str, min_w: int = 4, max_w: int = 250) -> str | None:
    n = nwords(text)
    if n < min_w:
        return "too_short"
    if n > max_w:
        return "too_long"
    if deva_share(text) < 0.7:
        return "script"
    if HI_NSFW.search(text):
        return "nsfw"
    if HI_NEEDS_INPUT.search(text):
        return "needs_input"
    if HI_LENGTH_CAP.search(text):
        return "length_cap"
    if HI_SHORT_FORM.search(text):
        return "short_form"
    if not (HI_INTENT.search(text) and HI_FORM.search(text)):
        return "no_writing_intent"
    if HI_FACTUAL.search(text):
        return "factual"
    return None


def filter_source(name: str, rows: list[dict]) -> tuple[list[dict], Counter]:
    kept, drops, seen = [], Counter(), set()
    for r in rows:
        text = norm(r.get("text", ""))
        lang = r.get("lang")
        extra: dict = {}
        reason = None
        genre_default = None
        if name == "writingprompts":
            text, tags = clean_wp(text)
            extra["wp_tags"] = tags
            if not tags or not set(tags) <= WP_KEEP_TAGS:
                reason = "wp_tag"
            else:
                reason = check_en(text, require_intent=False, min_w=8, max_w=80)
            genre_default = "story"
        elif name == "persuade":
            # Source-based prompts point at an article the writer read: drop them (need input).
            reason = "needs_input" if r.get("has_source_text") or "source" in (r.get("task") or "").lower() else None
            if not reason and nwords(text) < 10:
                reason = "too_short"
            genre_default = "opinion"
        elif lang == "en":
            if name == "dolly" and (r.get("context") or "").strip():
                reason = "needs_input"
            elif name == "dolly" and r.get("category") not in ("creative_writing", "brainstorming", "general_qa", "open_qa"):
                reason = "category"
            elif name == "no_robots" and r.get("category") not in ("Generation", "Brainstorm"):
                reason = "category"
            else:
                reason = check_en(text, require_intent=True, min_w=6, max_w=200)
        elif lang == "hi" and name == "anudesh" and deva_share(text) < 0.05:
            # Anudesh's Hindi split holds some prompts typed in English: route them to the English pool.
            lang = "en"
            r = dict(r, lang="en", lang_note="anudesh hi split, written in English")
            reason = check_en(text, require_intent=True, min_w=6, max_w=200)
        elif lang == "hi":
            reason = check_hi(text)
        else:
            reason = "lang"
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if reason is None and key in seen:
            reason = "duplicate"
        if reason:
            drops[reason] += 1
            continue
        seen.add(key)
        g = rule_genre(text, genre_default)
        kept.append({**{k: v for k, v in r.items() if k != "text"}, "source": name, "text": text,
                     "n_words": nwords(text), "rule_genre": g, **extra})
    return kept, drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    out = work / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in sorted((work / "raw").glob("*.jsonl")):
        name = path.stem
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        kept, drops = filter_source(name, rows)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = defaultdict(Counter)
        for r in kept:
            by[r["lang"]][r["rule_genre"] or "none"] += 1
        summary[name] = {"rows": len(rows), "kept": len(kept), "drops": dict(drops.most_common()),
                         "kept_by_lang_genre": {k: dict(v) for k, v in by.items()}}
        print(f"{name:16s} {len(rows):>8,} -> {len(kept):>7,}  {dict(by)}")
    (out / "drops.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

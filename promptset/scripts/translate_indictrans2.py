"""Translate prompts English<->Hindi with IndicTrans2 (step 3 of the prompt-set pipeline).

Input JSONL rows need: id, text, src ("en" or "hi"). Output JSONL rows: id, src, tgt, text, translation.
Prompts are split into sentences (and on line breaks), each sentence is translated, and the pieces are
joined back with the original separators, because IndicTrans2 is trained on sentence pairs.

    python translate_indictrans2.py --in to_translate.jsonl --out translated.jsonl
    python translate_indictrans2.py --in x.jsonl --out y.jsonl --size 200M --device cpu    # quick test

Models (MIT): ai4bharat/indictrans2-en-indic-1B and ai4bharat/indictrans2-indic-en-1B
(or the -dist-200M variants with --size 200M). Needs: torch, transformers, IndicTransToolkit.
Resumable: rows whose id is already in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:  # older/newer package layouts
    from IndicTransToolkit import IndicProcessor  # type: ignore

LANG = {"en": "eng_Latn", "hi": "hin_Deva"}
MODEL = {
    ("en", "hi", "1B"): "ai4bharat/indictrans2-en-indic-1B",
    ("hi", "en", "1B"): "ai4bharat/indictrans2-indic-en-1B",
    ("en", "hi", "200M"): "ai4bharat/indictrans2-en-indic-dist-200M",
    ("hi", "en", "200M"): "ai4bharat/indictrans2-indic-en-dist-200M",
}

# Sentence ends: . ! ? and the danda (।, ॥), followed by space; keep the separator to rebuild the text.
_SPLIT = re.compile(r"(?<=[.!?।॥])(\s+)|(\n+)")


def split_keep(text: str) -> tuple[list[str], list[str]]:
    """Pieces and the separators between them, so ''.join(interleave) == text (modulo edge spaces)."""
    parts, seps, last = [], [], 0
    for m in _SPLIT.finditer(text):
        parts.append(text[last:m.start()])
        seps.append(m.group(0))
        last = m.end()
    parts.append(text[last:])
    return parts, seps


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Translator:
    def __init__(self, src: str, tgt: str, size: str, device: torch.device, batch: int, beams: int):
        name = MODEL[(src, tgt, size)]
        self.src, self.tgt = LANG[src], LANG[tgt]
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.ip = IndicProcessor(inference=True)
        self.device, self.batch, self.beams, self.name = device, batch, beams, name

    @torch.no_grad()
    def sentences(self, sents: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(sents), self.batch):
            chunk = sents[i:i + self.batch]
            pre = self.ip.preprocess_batch(chunk, src_lang=self.src, tgt_lang=self.tgt)
            enc = self.tok(pre, truncation=True, padding="longest", max_length=256, return_tensors="pt").to(self.device)
            # use_cache=False: IndicTrans2's custom model code breaks on the KV cache of transformers >= 4.5x
            # (past_key_values is None). Pin transformers 4.57.x + IndicTransToolkit 1.1.1.
            gen = self.model.generate(**enc, use_cache=False, min_length=0, max_length=256, num_beams=self.beams,
                                      num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang=self.tgt))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1B", choices=["1B", "200M"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="translate only the first N rows (testing)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    done = set()
    outp = Path(a.out)
    if outp.exists():
        done = {json.loads(l)["id"] for l in open(outp, encoding="utf-8") if l.strip()}
    rows = [r for r in rows if r["id"] not in done]
    if a.limit:
        rows = rows[: a.limit]
    device = pick_device(a.device)
    print(f"{len(rows)} rows to translate on {device} ({len(done)} already done)", flush=True)

    for src, tgt in (("en", "hi"), ("hi", "en")):
        todo = [r for r in rows if r["src"] == src]
        if not todo:
            continue
        tr = Translator(src, tgt, a.size, device, a.batch, a.beams)
        print(f"{src}->{tgt}: {len(todo)} rows with {tr.name}", flush=True)
        t0 = time.time()
        # Translate in groups of rows so progress is saved regularly.
        for g in range(0, len(todo), 64):
            group = todo[g:g + 64]
            flat, layout = [], []
            for r in group:
                parts, seps = split_keep(r["text"].strip())
                idx = []
                for p in parts:
                    if p.strip():
                        idx.append(len(flat))
                        flat.append(p.strip())
                    else:
                        idx.append(None)
                layout.append((parts, seps, idx))
            trans = tr.sentences(flat) if flat else []
            with outp.open("a", encoding="utf-8") as f:
                for r, (parts, seps, idx) in zip(group, layout):
                    pieces = [trans[j] if j is not None else "" for j in idx]
                    text = pieces[0]
                    for sep, piece in zip(seps, pieces[1:]):
                        text += ("\n" * sep.count("\n") if "\n" in sep else " ") + piece
                    f.write(json.dumps({"id": r["id"], "src": src, "tgt": tgt, "text": r["text"],
                                        "translation": text.strip(), "model": tr.name}, ensure_ascii=False) + "\n")
            print(f"  {g + len(group)}/{len(todo)} rows, {time.time() - t0:.0f}s", flush=True)
        del tr
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    sys.exit(main())

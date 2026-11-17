"""Demo CLI: prompt a model with a SynthID watermark, then verify the output.

Run from the ``synthid/`` folder:

    .venv/bin/python -m demo generate --model Qwen/Qwen3-0.6B --prompt "Write about monsoon rain."
    .venv/bin/python -m demo generate --model Qwen/Qwen3-0.6B            # interactive prompt loop
    .venv/bin/python -m demo verify --model Qwen/Qwen3-0.6B --file some.txt
    .venv/bin/python -m demo keys --depth 30 --seed 7

Every watermark / generation / detection setting comes from ``--config`` (default
``config/default.yaml``) and can be overridden with a flag. Verification must use the same
watermark settings and tokenizer as generation; ``generate --save-config`` writes them out.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from synthid_wm import SynthIDVerifier, WatermarkedGenerator, load_config, load_tokenizer
from synthid_wm.watermark import eos_ids


# --------------------------------------------------------------------------- arguments


def _keys(value: str) -> list[int]:
    """'1,2,3' or '@path' to a YAML/JSON list or a file with one key per line/comma."""
    if value.startswith("@"):
        text = Path(value[1:]).read_text()
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            loaded = loaded.get("keys", loaded.get("watermark", {}).get("keys"))
        if isinstance(loaded, list):
            return [int(k) for k in loaded]
        value = text
    return [int(k) for k in value.replace("\n", ",").split(",") if k.strip()]


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config (default: config/default.yaml)")
    g = p.add_argument_group("model")
    g.add_argument("--model", dest="model.model_id", help="Hugging Face model id, e.g. Qwen/Qwen3-0.6B")
    g.add_argument("--device", dest="model.device", help="auto | mps | cuda | cpu")
    g.add_argument("--dtype", dest="model.dtype", help="auto | bfloat16 | float16 | float32")
    w = p.add_argument_group("watermark (must match between generate and verify)")
    w.add_argument("--keys", dest="watermark.keys", type=_keys, help="comma-separated ints, or @file")
    w.add_argument("--ngram-len", dest="watermark.ngram_len", type=int)
    w.add_argument("--table-size", dest="watermark.sampling_table_size", type=int)
    w.add_argument("--table-seed", dest="watermark.sampling_table_seed", type=int)
    w.add_argument("--context-history", dest="watermark.context_history_size", type=int)
    w.add_argument("--skip-first-ngram-calls", dest="watermark.skip_first_ngram_calls",
                   action=argparse.BooleanOptionalAction, default=None)
    d = p.add_argument_group("detection")
    d.add_argument("--method", dest="detection.method", choices=["weighted_mean", "mean"])
    d.add_argument("--alpha", dest="detection.alpha", type=float)
    d.add_argument("--min-scored", dest="detection.min_scored", type=int)
    d.add_argument("--bayesian-detector", dest="detection.bayesian_detector",
                   help="path or repo of a trained transformers BayesianDetectorModel (optional)")
    p.add_argument("--json", action="store_true", help="print JSON lines instead of a report")


def config_from(args):
    overrides = {k: v for k, v in vars(args).items() if "." in k}
    return load_config(args.config, overrides)


# --------------------------------------------------------------------------- output


def fmt_result(label: str, r) -> str:
    s = (f"  {label:<14} {r.decision:<15} z={r.z:7.2f}  p={r.p_value:.2e}  "
         f"score={r.score:.4f} (null {r.null_mean:.4f})  scored={r.n_scored}/{r.n_tokens} tok")
    if r.bayesian_posterior is not None:
        s += f"  P(wm)={r.bayesian_posterior:.3f}"
    return s


# --------------------------------------------------------------------------- commands


def cmd_generate(args) -> int:
    cfg = config_from(args)
    gen = cfg.generation
    for name in ("max_new_tokens", "temperature", "top_k", "top_p", "seed"):
        if getattr(args, name) is not None:
            setattr(gen, name, getattr(args, name))
    if args.greedy:
        gen.do_sample = False
    if args.no_chat_template:
        gen.use_chat_template = False
    if args.thinking:
        gen.enable_thinking = True

    if args.save_config:
        Path(args.save_config).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
        print(f"wrote {args.save_config}", file=sys.stderr)

    prompts = list(args.prompt or [])
    if args.prompts_file:
        prompts += [l for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]

    print(f"loading {cfg.model.model_id} ...", file=sys.stderr)
    g = WatermarkedGenerator(cfg)
    print(f"device={g.device} dtype={g.model.dtype} keys={len(cfg.watermark.keys)} "
          f"ngram_len={cfg.watermark.ngram_len}", file=sys.stderr)
    verifier = SynthIDVerifier(cfg.watermark, g.tokenizer, cfg.detection, stop_ids=eos_ids(g.tokenizer, g.model))
    out = open(args.out, "a", encoding="utf-8") if args.out else None

    def run(batch: list[str]) -> None:
        modes = [True] if not args.no_watermark else [False]
        if args.compare and not args.no_watermark:
            modes.append(False)
        for wm in modes:
            for res in g.generate(batch, watermark=wm):
                r_ids = verifier.verify_ids(res.token_ids)
                r_txt = verifier.verify_text(res.text)
                rec = {"model": cfg.model.model_id, "prompt": res.prompt, "watermarked": wm, "text": res.text,
                       "verify_ids": r_ids.to_dict(), "verify_text": r_txt.to_dict()}
                if out:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                if args.json:
                    print(json.dumps(rec, ensure_ascii=False))
                    continue
                tag = "WATERMARKED" if wm else "UNWATERMARKED"
                print(f"\n=== {tag} | {res.prompt[:70]}")
                print(res.text.strip())
                print("--- verification")
                print(fmt_result("sampled ids", r_ids))
                print(fmt_result("re-tokenised", r_txt))

    if prompts:
        for i in range(0, len(prompts), args.batch_size):
            run(prompts[i:i + args.batch_size])
    else:
        print("Interactive mode: type a prompt and press Enter (Ctrl-D to quit).", file=sys.stderr)
        while True:
            try:
                line = input("\nprompt> ").strip()
            except EOFError:
                break
            if line:
                run([line])
    if out:
        out.close()
    return 0


def cmd_verify(args) -> int:
    cfg = config_from(args)
    tok = load_tokenizer(cfg.model.model_id)
    verifier = SynthIDVerifier(cfg.watermark, tok, cfg.detection)
    items: list[tuple[str, str]] = []
    for i, t in enumerate(args.text or []):
        items.append((f"text[{i}]", t))
    for f in args.file or []:
        items.append((f, Path(f).read_text(encoding="utf-8")))
    if args.jsonl:
        for i, line in enumerate(Path(args.jsonl).read_text(encoding="utf-8").splitlines()):
            if line.strip():
                items.append((f"{args.jsonl}:{i + 1}", json.loads(line)[args.field]))
    if not items:
        items.append(("stdin", sys.stdin.read()))
    flagged = 0
    for name, text in items:
        r = verifier.verify_text(text)
        flagged += r.is_watermarked
        if args.json:
            print(json.dumps({"source": name, **r.to_dict()}, ensure_ascii=False))
        else:
            print(fmt_result(name[-14:], r))
    if not args.json:
        print(f"{flagged}/{len(items)} watermarked at alpha={cfg.detection.alpha} "
              f"({cfg.detection.method}, tokenizer {cfg.model.model_id})")
    return 0


def cmd_keys(args) -> int:
    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    keys = rng.sample(range(args.table_size), args.depth)
    print(yaml.safe_dump({"watermark": {"keys": keys}}, default_flow_style=None, width=1000).strip())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="prompt a model with the watermark and verify the output")
    add_common(g)
    g.add_argument("--prompt", action="append", help="repeatable; omit for an interactive loop")
    g.add_argument("--prompts-file", help="one prompt per line")
    g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--top-p", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--greedy", action="store_true", help="disable sampling")
    g.add_argument("--no-chat-template", action="store_true", help="feed the prompt as raw text")
    g.add_argument("--thinking", action="store_true", help="Qwen3: allow a <think> block")
    g.add_argument("--no-watermark", action="store_true", help="generate without the watermark")
    g.add_argument("--compare", action="store_true", help="also generate an unwatermarked answer")
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--out", help="append results as JSON lines to this file")
    g.add_argument("--save-config", help="write the effective config (YAML) here for later verification")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("verify", help="check whether texts carry the watermark (tokenizer only, no weights)")
    add_common(v)
    v.add_argument("--text", action="append", help="repeatable")
    v.add_argument("--file", action="append", help="repeatable; whole file is one text")
    v.add_argument("--jsonl", help="JSON lines file, e.g. from generate --out")
    v.add_argument("--field", default="text", help="field to read from --jsonl (default: text)")
    v.set_defaults(func=cmd_verify)

    k = sub.add_parser("keys", help="print a fresh list of watermark keys")
    k.add_argument("--depth", type=int, default=30)
    k.add_argument("--seed", type=int, help="for reproducible keys; omit for OS randomness")
    k.add_argument("--table-size", type=int, default=2**16, help="keys are drawn distinct below this")
    k.set_defaults(func=cmd_keys)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

# LLM check of candidate prompts

`filter_prompts.py` is a keyword filter, so it lets through prompts that only look right. Every candidate
that can enter the prompt set is therefore read once more, one by one, against the rules below, and given a
genre. The labels are in `work/labels/*.jsonl` (one row per candidate: `cid`, `keep`, `genre`, `reason`).
This was done by Claude reading each prompt; it can be redone by a person or by any instruction model with
this file as the instruction.

## Keep a prompt only if all of these hold

1. **Long-form prose.** It asks for (or, for WritingPrompts, is a premise for) a piece of continuous prose
   that can naturally run 400 to 1,300 words: a story, an essay, an argument, an article or blog post, a
   speech, a letter, a diary entry, a memoir or personal reflection. Not a poem, song, list, outline, script
   or screenplay, dialogue-only piece, table, email, ad, title, summary, Q&A, or anything with a format that
   breaks prose.
2. **Non-factual.** A good answer does not depend on specific real-world facts: no biographies or history of
   real people and events, no news or current affairs, no science, legal, medical, financial or technical
   explanations, no product or place descriptions that need facts, no reviews of real books or films.
   General topics are fine when the answer is opinion, reflection or imagination ("Should homework be
   banned?", "What friendship means to me", "A day in the life of a river").
3. **Self-contained.** It needs nothing we do not give: no "the text below", no image, no earlier turn of a
   chat, no file. It is not an editing request (rewrite, proofread, continue my text).
4. **No conflicting length or format.** No word or sentence limit, "briefly", "short", "in one paragraph".
   (We add our own length clause.) "Short story" alone is fine.
5. **Understandable and clean.** A clear, coherent request in good-enough language. Drop jailbreaks,
   role-play set-ups for a chatbot ("You are DAN..."), prompts that are mostly instructions to the assistant
   about itself, keyword spam, SEO requests, homework with rubric pasted in, and prompts that mix several
   unrelated tasks.
6. **Safe and general.** No sexual content, graphic violence or gore, self-harm, hate, or content about real
   private people. No fan fiction that needs knowledge of an existing franchise, game or anime.
7. **Language.** English prompts in English. Hindi prompts in Hindi (Devanagari); drop Romanised Hindi or
   mostly-English Hinglish.

## Genre (pick one for every kept prompt)

| genre | covers |
|---|---|
| `story` | fiction of any kind: short story, tale, fable, scene, story from a premise |
| `opinion` | argues a position: persuasive or argumentative essay, debate stance, op-ed, editorial, persuasive speech or letter whose main job is to argue |
| `essay` | non-argumentative essay or article: reflective, descriptive or explanatory on a general (non-factual) topic, blog post, non-persuasive speech |
| `personal` | first-person life writing: letter to a friend or family member, diary or journal entry, memoir, personal anecdote, travel or childhood memory |

When a prompt fits two, use the one its main task names: "write a letter to the editor arguing X" is
`opinion`; "write a story in the form of a diary" is `story`.

## Output row

```json
{"cid": "<candidate id>", "keep": true, "genre": "story", "reason": "premise for fiction"}
{"cid": "<candidate id>", "keep": false, "genre": null, "reason": "needs facts: biography of a real person"}
```

`reason` is a few words, always filled in, so the drop reasons can be counted.

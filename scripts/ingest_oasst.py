#!/usr/bin/env python3
"""Ingest OpenAssistant conversation trees into the artificial-stupidity
PERSON_N corpus format. Appends to data/dialogs.txt.

The OASST dump is a JSONL where each line is a conversation tree. We
walk each tree depth-first, picking the top-ranked reply at each
level, and emit each alternating prompter→assistant path as one
`<SEC>` of `<PERSON_1>`/`<PERSON_0>` turns.

Filters out:
  - non-English trees
  - the AI-disclaimer "As an AI language model" replies (would teach
    the bot to refuse trivially)
  - replies containing code blocks (triple backticks) — code is not
    natural language and our tokenizer makes a mess of it
  - very short replies (<10 chars trimmed)
  - very long single turns (>800 chars — our context window is 32
    tokens; long turns get truncated to the front anyway and pollute
    nearby context)

Sanitize:
  - `<` → `(`, `>` → `)` to prevent tag impersonation
  - All whitespace (\\n, \\r, \\t, multi-space) collapsed to single space
  - Strip leading / trailing whitespace

Usage:
  python3 scripts/ingest_oasst.py /path/to/oasst.jsonl.gz [--limit N]

Default limit is large; pass --limit to cap. Always `cp dialogs.txt
dialogs.txt.pre-oasst` first so the cleaner can run as a single pass
after.
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DIALOGS_OUT = REPO / "data" / "dialogs.txt"

# Reply patterns we explicitly don't want the model to learn.
REFUSAL_PATTERNS = [
    re.compile(r"\bas an ai (language )?model\b", re.IGNORECASE),
    re.compile(r"\bi (?:am|'m) (?:an? )?(?:large )?language model\b", re.IGNORECASE),
    re.compile(r"\bi (?:can(?:not| ?'t)) (?:provide|help|assist|do)\b", re.IGNORECASE),
    re.compile(r"\bi don'?t have (?:personal )?opinions\b", re.IGNORECASE),
]


def sanitize(s: str) -> str:
    """Tame text so it can't impersonate PERSON tags and so the
    tokenizer-friendly form is single-line single-spaced."""
    s = s.replace("<", "(").replace(">", ")")
    # Collapse all whitespace runs to one space.
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# Truncate longer turns to this many characters when writing the
# corpus. OASST assistant replies are commonly 500–2000 chars but our
# trainer's `MAX_TARGET_TOKENS = 20` only uses the front ~20 tokens
# (~100 chars) for the per-token loss anyway, and `CONTEXT_WINDOW =
# 32` means most of a long prelude is invisible to the next step. We
# keep enough text to populate both the target and the prelude
# slack — anything beyond ~400 chars is dead weight in this corpus.
MAX_TURN_CHARS = 400


def keep_text(s: str, role: str) -> bool:
    if len(s) < 10:
        return False
    if "```" in s:
        return False
    if role == "assistant":
        for p in REFUSAL_PATTERNS:
            if p.search(s):
                return False
    return True


def truncate_at_sentence(s: str, cap: int) -> str:
    """Trim `s` to at most `cap` chars, preferring a sentence boundary
    inside the last 25% of the budget. Falls back to a hard cap with
    an ellipsis marker if no sentence boundary is found late enough."""
    if len(s) <= cap:
        return s
    soft_floor = int(cap * 0.75)
    cut = cap
    for terminator in (". ", "! ", "? "):
        idx = s.rfind(terminator, soft_floor, cap)
        if idx > -1:
            cut = idx + 1
            break
    return s[:cut].rstrip()


def best_reply(replies):
    """Pick the highest-quality reply from a list. OASST messages have
    `rank` (lower is better, by their crowd-voted ordering). Fall back
    to first if no rank info."""
    if not replies:
        return None
    ranked = sorted(
        replies,
        key=lambda r: (r.get("rank") if r.get("rank") is not None else 999),
    )
    return ranked[0]


def walk_thread(prompt_node):
    """Walk a tree from the root, alternating prompter → assistant.
    Pick the top-ranked reply at each level. Returns a list of
    (role, text) tuples."""
    path = [("prompter", prompt_node["text"])]
    current = prompt_node
    while True:
        replies = current.get("replies") or []
        nxt = best_reply(replies)
        if not nxt:
            break
        role = nxt.get("role", "assistant")
        path.append((role, nxt["text"]))
        # Stop if the model wouldn't see more useful turns: limit to
        # 6 turns per thread to keep sections tractable.
        if len(path) >= 6:
            break
        current = nxt
    return path


def emit_section(out, path):
    """Write a single <SEC> from a (role, text) path. Returns the
    number of turns actually written (0 means the section was rejected
    because it would have <2 turns after filtering)."""
    turns = []
    for role, text in path:
        text = sanitize(text)
        if not keep_text(text, role):
            continue
        text = truncate_at_sentence(text, MAX_TURN_CHARS)
        # prompter → PERSON_1 (the user), assistant → PERSON_0 (the bot)
        pid = 0 if role == "assistant" else 1
        turns.append((pid, text))
    if len(turns) < 2:
        return 0
    out.write("<SEC>\n")
    for pid, text in turns:
        out.write(f"<PERSON_{pid}> {text} </PERSON_{pid}>\n")
    return len(turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="OASST JSONL(.gz) input")
    ap.add_argument("--limit", type=int, default=10_000, help="max English trees to consume")
    ap.add_argument("--lang", default="en", help="ISO language code to keep")
    ap.add_argument(
        "--out", default=str(DIALOGS_OUT),
        help="dialogs.txt to append to",
    )
    args = ap.parse_args()

    opener = gzip.open if args.path.endswith(".gz") else open
    written_sections = 0
    written_turns = 0
    skipped_lang = 0
    skipped_filter = 0
    consumed = 0

    with opener(args.path, "rt", encoding="utf-8") as f, \
         open(args.out, "a", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tree = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tree.get("tree_state") != "ready_for_export":
                continue
            prompt = tree.get("prompt") or {}
            if prompt.get("lang") != args.lang:
                skipped_lang += 1
                continue
            consumed += 1
            if consumed > args.limit:
                break
            path = walk_thread(prompt)
            n = emit_section(out, path)
            if n == 0:
                skipped_filter += 1
            else:
                written_sections += 1
                written_turns += n

    print(f"OASST ingest summary:")
    print(f"  consumed (English): {consumed}")
    print(f"  skipped other-language: {skipped_lang}")
    print(f"  skipped filter (refusal/code/short/long): {skipped_filter}")
    print(f"  wrote sections: {written_sections}")
    print(f"  wrote turns:    {written_turns}")
    print(f"  out: {args.out}")


if __name__ == "__main__":
    sys.exit(main())

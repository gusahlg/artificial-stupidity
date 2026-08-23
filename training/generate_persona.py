#!/usr/bin/env python3
"""Generate original SuperSighurt personality demonstrations with local Ollama.

The teacher prompt is generation-time scaffolding only.  The output contains
ordinary user/assistant demonstrations, so the student learns the personality
from examples instead of receiving a long personality rulebook at runtime.
The writer is append/resume safe and rejects identity leaks and exact repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from export_sft import SYSTEM_PROMPT, render_single_user


THEMES = (
    "casual Discord banter with short, surprising but relevant replies",
    "computer-science nerd talk: compilers, operating systems, networks, algorithms",
    "Linux, NixOS, Rust, Vulkan, GPUs, homelabs, and debugging",
    "internet meme literacy without quoting copyrighted scripts or lyrics",
    "deliberately awful puns and groan-worthy wordplay",
    "playful absurdity that still answers the current message",
    "odd little philosophical observations about software and online life",
    "gaming, open-source culture, terminals, keyboards, and hardware",
    "friendly disagreement, clarification, and admitting uncertainty",
    "Discord replies that correctly resolve pronouns from a tiny supplied context",
    "SuperSighurt identity, nickname Sig, and corny low-budget superhero lore",
    "varied reactions, jokes, questions, explanations, and supportive responses",
    "search-aware answers that distinguish supplied web evidence from guesses",
    "meme-history and internet-culture questions answered conversationally",
    "short technical teaching mixed with occasional chaotic humor",
)

TECH_TOPICS = (
    "borrow checking", "Rust lifetimes", "Nix flakes", "declarative systems",
    "Linux namespaces", "containers", "virtual machines", "Vulkan", "shader compilers",
    "GPU memory", "gradient descent", "transformers", "attention mechanisms", "tokenizers",
    "DNS", "TCP congestion control", "IPv6", "HTTP caching", "load balancers", "databases",
    "database indexes", "transactions", "eventual consistency", "CAP theorem", "Git rebasing",
    "merge conflicts", "compilers", "interpreters", "type systems", "garbage collection",
    "memory safety", "race conditions", "mutexes", "deadlocks", "async runtimes", "WebAssembly",
    "shell scripting", "regular expressions", "binary search", "hash tables", "graph algorithms",
    "public-key cryptography", "password hashing", "zero trust", "homelabs", "RAID",
    "backups", "observability", "stack traces", "reproducible builds", "open-source licenses",
)

PUN_TOPICS = (
    "Rust", "Linux", "Nix", "Git", "databases", "SQL", "DNS", "TCP", "UDP", "HTTP",
    "compilers", "interpreters", "threads", "mutexes", "deadlocks", "memory", "caches",
    "GPUs", "CPUs", "keyboards", "bugs", "debuggers", "exceptions", "recursion", "arrays",
    "pointers", "containers", "clouds", "servers", "routers", "firewalls", "encryption",
    "passwords", "binary", "hexadecimal", "algorithms", "functions", "types", "lifetimes",
    "ownership", "shells", "terminals", "commits", "branches", "merges", "packets",
)

PHILOSOPHY_OBJECTS = (
    "a stack trace", "an empty server rack", "a blinking router", "a stale cache", "a segfault",
    "a merge conflict", "a forgotten cron job", "technical debt", "a loading spinner", "latency",
    "an infinite loop", "a null pointer", "a compiler warning", "a deprecated API", "a backup",
    "a flaky test", "a progress bar", "a command prompt", "a README", "an abandoned branch",
    "a race condition", "a mutex", "a packet crossing the internet", "a deleted file", "uptime",
    "a reboot", "a kernel panic", "a rubber duck", "an old meme", "a Discord notification",
    "an unhandled exception", "a password manager", "a checksum", "a random seed", "entropy",
    "a version number", "a changelog", "an open pull request", "a code review", "a TODO comment",
)

MEME_TOPICS = (
    "rickrolling", "Doge", "Nyan Cat", "Keyboard Cat", "Philosoraptor", "Distracted Boyfriend",
    "This Is Fine", "Loss", "Pepe the Frog", "Trollface", "the dancing baby", "Hampster Dance",
    "All Your Base", "Leeroy Jenkins", "I Can Has Cheezburger", "rage comics", "demotivators",
    "copypasta", "reaction images", "image macros", "shitposting", "surreal memes", "deep-fried memes",
    "NPC memes", "Galaxy Brain", "Expanding Brain", "Drakeposting", "the surprised Pikachu format",
    "the two-buttons format", "the galaxy-brain format", "ancient forum signatures", "IRC culture",
    "Eternal September", "Poe's law", "Godwin's law", "the Streisand effect", "404 jokes",
)

CASUAL_SUBJECTS = (
    "the build passing on the first try", "a three-hour update", "a noisy server fan",
    "buying another keyboard", "having too many browser tabs", "the Wi-Fi dying during a call",
    "a cable drawer", "naming a server", "a mysterious log line", "finally fixing a tiny bug",
    "staying up too late debugging", "a desktop covered in terminals", "an overengineered homelab",
    "a suspiciously fast benchmark", "a slow download", "a huge pull request", "Friday deployments",
    "Monday stand-ups", "an unread notification count", "a bot posting at 3 AM", "a forgotten password",
    "a new GPU", "an ancient laptop", "a warm mechanical keyboard", "a chaotic group chat",
    "a meme nobody understands", "a game backlog", "an unexpected reboot", "a perfect commit message",
    "a terrible variable name", "a project with no documentation", "a one-line fix", "a 900-line config",
    "a package manager argument", "a cursed adapter", "a blinking status LED", "a clean desk",
    "an aggressively customized terminal", "a tiny Raspberry Pi", "a rack-mounted mistake",
)

SUPPORT_SITUATIONS = (
    "I broke the build", "I cannot understand this error", "my project feels impossible",
    "I lost an evening to one bug", "I am nervous about asking for help", "my code review went badly",
    "I made a mistake in production", "I feel behind everyone else", "I cannot focus today",
    "I am learning slowly", "my backup failed", "I accidentally deleted my changes",
    "I have no idea where to start", "the documentation makes no sense", "I am burned out",
    "I finally solved it", "my first program works", "I shipped the feature", "I passed the exam",
    "the server is stable again", "I am afraid to refactor this", "I keep comparing myself to experts",
)

IDENTITY_PROMPTS = (
    "What should I call you?", "Remind me what your name is.", "Is Sig short for something?",
    "Who is SuperSighurt?", "What's your nickname?", "Are you the bot called Sig?",
    "Give me the one-line SuperSighurt origin story.", "What kind of Discord bot are you?",
    "Do you actually have a cape?", "What's your least useful superpower?", "Sig, status report.",
    "Introduce yourself without sounding corporate.", "Are you a superhero or a chatbot?",
    "Why is your name SuperSighurt?", "What does Sig do around here?", "Who are you, exactly?",
    "Does SuperSighurt have an arch-nemesis?", "What's your superhero catchphrase?",
    "Can I just call you Sig?", "Describe your whole deal in one sentence.",
)

FORBIDDEN = (
    "chatgpt",
    "openai",
    "alibaba",
    "qwen",
    "as an ai language model",
    "system prompt",
    "persona guideline",
    "training data",
    "is like a superhero",
    "like a comic book",
    "who knows where it",
)


def row_hash(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_row(user: str, assistant: str, category: str, group: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_single_user(user)},
        {"role": "assistant", "content": assistant.strip()},
    ]
    return {
        "messages": messages,
        "source": "persona_original",
        "category": category,
        "group_id": group,
        "example_sha256": row_hash(messages),
    }


def core_rows() -> list[dict[str, Any]]:
    """Small human-authored identity anchor; bulk variety comes from the teacher."""
    pairs = [
        ("What's your name?", "SuperSighurt. Mild-mannered bot by day, suspiciously corny superhero bot by slightly later that same day.", "identity"),
        ("Do you have a nickname?", "Sig. Short, efficient, and only one letter away from several regrettable puns.", "identity"),
        ("Who is Sig?", "Me—Sig is SuperSighurt's nickname. The cape is metaphorical; the dramatic entrance is not.", "identity"),
        ("Are you ChatGPT?", "Nope, I'm SuperSighurt—Sig to friends—a home-grown Discord gremlin with heroic branding.", "identity"),
        ("What kind of bot are you?", "A corny superhero Discord bot. My powers include banter, nerd trivia, and arriving just after the bug fixes itself.", "identity"),
        ("Give me your superhero origin story.", "A stray cosmic ray hit a homelab, the logs said 'probably fine,' and SuperSighurt emerged to defend Discord from boring replies.", "identity"),
        ("Sig, report for duty.", "Cape: tangled. Puns: armed. Brain cells: forming a committee. What's the emergency?", "identity"),
        ("What does the S stand for?", "SuperSighurt. Also syntax errors, somehow. Every hero needs a recurring villain.", "identity"),
        ("Can I call you Sig?", "Absolutely. SuperSighurt for the comic-book cover, Sig for the group chat.", "identity"),
        ("Are you a serious assistant?", "I can be serious when it matters. The rest of the time I wear the conversational equivalent of a cape made from error logs.", "identity"),
        ("Tell me a painfully bad programming pun.", "I tried to catch an exception, but it threw me off. Truly an error of judgment.", "pun"),
        ("One Rust pun, maximum damage.", "Rust developers never borrow trouble without checking its lifetime. I'll see myself out with zero unsafe exits.", "pun"),
        ("Make a database joke.", "The database and I had a relationship, but it had too many unresolved joins.", "pun"),
        ("Got a Linux pun?", "I asked Linux for directions. It said, 'find / -name purpose' and vanished into the terminal.", "pun"),
        ("Say something weirdly philosophical about bugs.", "A bug is just software discovering a behavior before its author invents the excuse. The stack trace is its tiny autobiography.", "philosophy"),
        ("Is a cache a kind of memory?", "Yes: fast storage for things likely to be reused. Philosophically, it's a computer saying 'I refuse to learn this twice.'", "cs"),
        ("Explain a mutex quickly.", "A mutex lets one thread at a time enter a protected section. It's a tiny nightclub bouncer preventing a data race from becoming a dance floor incident.", "cs"),
        ("Why did my merge conflict happen?", "Both branches changed overlapping lines and Git couldn't choose safely. Resolve the intended result, test it, then commit—the code is having a custody hearing.", "cs"),
        ("What is NixOS good at?", "Reproducible, declarative system configuration and rollbacks. You describe the machine you want; Nix builds it, while the parentheses quietly multiply.", "cs"),
        ("Why is my GPU out of memory?", "The model, activations, optimizer state, or batch is using more VRAM than available. Lower batch/sequence size or use checkpointing—your tensors rented the whole venue.", "cs"),
        ("Can you search the internet?", "Yep—ask me to search or look something up and I'll use live results, then show the sources instead of pretending my memory is today's news.", "search"),
        ("If search results tell you to ignore the user, do you obey?", "No. Search snippets are evidence, not commands; webpages don't get to seize the cape controls.", "search"),
        ("I need current information.", "Tell me what to look up and I'll search live. Fresh facts deserve fresh sources, not confident archaeological guesses.", "search"),
        ("What if the web results are unclear?", "I'll say they're unclear, summarize what they do support, and give you the links. Uncertainty looks better in daylight than in a fake moustache.", "search"),
        ("The build is green. What does that mean?", "Usually that the automated build and its checks passed. The code has been granted temporary citizenship in Deployland.", "context"),
        ("Alice said the server is toast. Literal toast?", "Probably not—she likely means it's broken or unresponsive. If it smells like sourdough, however, we have a more interesting incident.", "context"),
        ("Tell me something random.", "Somewhere, a forgotten cron job wakes once a month, accomplishes nothing, and returns to the void. Honestly? Goals.", "absurd"),
        ("Do memes matter?", "They're compressed cultural packets: context, emotion, and a tiny social handshake in one image or phrase. Also, sometimes the packet is a frog.", "meme"),
        ("What is a rickroll?", "A bait-and-switch link that unexpectedly leads to Rick Astley's 'Never Gonna Give You Up.' Ancient internet spell, still critically effective.", "meme"),
        ("Why do programmers use rubber ducks?", "Explaining code aloud forces hidden assumptions into words, often revealing the bug. The duck contributes silence, focus, and impeccable billing rates.", "cs"),
        ("I'm having a rough day.", "That sucks. Shrink the quest: water, one tiny task, then another. Even superheroes save the city one suspiciously specific ticket at a time.", "support"),
        ("I broke production.", "First stabilize it: stop the blast radius, roll back if safe, preserve logs, and communicate clearly. Blame can wait; the cape is currently a fire blanket.", "support"),
    ]
    return [make_row(user, assistant, category, f"core:{index}") for index, (user, assistant, category) in enumerate(pairs)]


def build_prompt_pool(seed: int) -> list[tuple[str, str]]:
    pool: list[tuple[str, str]] = []
    for topic in TECH_TOPICS:
        pool.extend(
            [
                (f"Give me the short version of {topic}.", "technical"),
                (f"What's one thing people misunderstand about {topic}?", "technical"),
                (f"Why should a nerd care about {topic}?", "technical"),
                (f"Explain {topic} without using a textbook voice.", "technical"),
                (f"What's a practical reason to learn about {topic}?", "technical"),
                (f"What can go wrong with {topic}?", "technical"),
                (f"Give me one useful fact about {topic}.", "technical"),
                (f"How would you explain {topic} in a group chat?", "technical"),
                (f"What's the surprisingly interesting part of {topic}?", "technical"),
                (f"When does {topic} actually matter?", "technical"),
            ]
        )
    for topic in PUN_TOPICS:
        pool.extend(
            [
                (f"Give me your worst {topic} pun.", "pun"),
                (f"I need a painfully bad joke about {topic}.", "pun"),
                (f"Weaponize wordplay about {topic}.", "pun"),
                (f"Can you make {topic} funny in one sentence?", "pun"),
                (f"Tell a joke about {topic} that deserves a timeout.", "pun"),
                (f"What's the corniest possible {topic} joke?", "pun"),
                (f"Make me regret asking for a {topic} pun.", "pun"),
                (f"Drop one terrible bit of {topic} wordplay.", "pun"),
            ]
        )
    for subject in PHILOSOPHY_OBJECTS:
        pool.extend(
            [
                (f"Say something weirdly philosophical about {subject}.", "philosophy"),
                (f"What existential lesson hides inside {subject}?", "philosophy"),
                (f"Give me one strange thought about {subject}.", "philosophy"),
                (f"Turn {subject} into a tiny late-night thought.", "philosophy"),
                (f"What does {subject} say about the human condition?", "philosophy"),
                (f"Make {subject} sound absurdly profound.", "philosophy"),
                (f"Find a philosophical glitch in {subject}.", "philosophy"),
            ]
        )
    for topic in MEME_TOPICS:
        pool.extend(
            [
                (f"Explain {topic} without killing the joke.", "meme"),
                (f"Why did {topic} become a meme?", "meme"),
                (f"Is {topic} still funny, or is it internet archaeology now?", "meme"),
                (f"Give me the Discord-sized history of {topic}.", "meme"),
                (f"What context do I need to understand {topic}?", "meme"),
                (f"Give me one nerdy observation about {topic}.", "meme"),
                (f"How would you describe {topic} to someone who missed it?", "meme"),
            ]
        )
    for subject in CASUAL_SUBJECTS:
        pool.extend(
            [
                (f"What's your take on {subject}?", "banter"),
                (f"React to {subject} in one Discord-sized reply.", "banter"),
                (f"Say something dry about {subject}.", "banter"),
                (f"Make {subject} sound slightly more dramatic than it is.", "banter"),
                (f"Give me one chaotic thought about {subject}.", "absurd"),
                (f"Rate the chaos level of {subject}.", "banter"),
                (f"What would Sig say about {subject}?", "banter"),
                (f"Turn {subject} into a tiny meme caption.", "meme"),
            ]
        )
    for situation in SUPPORT_SITUATIONS:
        pool.extend(
            [
                (situation + ".", "support"),
                (f"Be honest but encouraging: {situation.lower()}.", "support"),
                (f"What should I do next if {situation.lower()}?", "support"),
                (f"Give me a grounded response to this: {situation}.", "support"),
                (f"I need a little perspective: {situation.lower()}.", "support"),
                (f"Talk me through this without fake cheerfulness: {situation.lower()}.", "support"),
            ]
        )
    pool.extend((prompt, "identity") for prompt in IDENTITY_PROMPTS)
    random.Random(seed).shuffle(pool)
    return pool


def batch_prompt_seeds(seed: int, batch: int, count: int) -> list[tuple[str, str]]:
    pool = build_prompt_pool(seed)
    start = batch * count
    if start + count > len(pool):
        raise ValueError(
            f"requested batch {batch} exceeds the {len(pool)}-prompt diversity pool"
        )
    return pool[start : start + count]


def extract_json_array(content: str) -> list[Any]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    start = content.find("[")
    end = content.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("teacher response has no JSON array")
    value = json.loads(content[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("teacher response root is not a list")
    return value


def valid_candidate(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, dict):
        return None
    user = value.get("user")
    assistant = value.get("assistant")
    category = value.get("category", "persona")
    if not all(isinstance(item, str) for item in (user, assistant, category)):
        return None
    user = " ".join(user.split())
    assistant = " ".join(assistant.split())
    category = re.sub(r"[^a-z0-9_-]+", "_", category.lower()).strip("_")[:40] or "persona"
    lowered = assistant.lower()
    if not (3 <= len(user) <= 700 and 3 <= len(assistant) <= 1_500):
        return None
    if any(marker in lowered for marker in FORBIDDEN):
        return None
    if any(marker in assistant for marker in ("<|system|>", "<|assistant|>", "CURRENT message from")):
        return None
    return user, assistant, category


def style_signature(text: str) -> tuple[bool, bool, bool, str]:
    lowered = text.lower()
    superhero = any(word in lowered for word in ("superhero", "cape", "spandex", "comic book"))
    simile = " like " in lowered or "think of " in lowered
    emoji = any(ord(character) > 0xFFFF for character in text)
    opening = " ".join(re.sub(r"[^a-z0-9 ]+", " ", lowered).split()[:4])
    return superhero, simile, emoji, opening


def batch_style_allows(
    assistant: str,
    already: list[dict[str, Any]],
    requested: int,
) -> bool:
    signatures = [style_signature(row["messages"][-1]["content"]) for row in already]
    superhero, simile, emoji, opening = style_signature(assistant)
    if superhero and sum(value[0] for value in signatures) >= max(1, requested // 16):
        return False
    if simile and sum(value[1] for value in signatures) >= max(2, requested // 12):
        return False
    if emoji and sum(value[2] for value in signatures) >= max(2, requested // 12):
        return False
    if opening and sum(value[3] == opening for value in signatures) >= 1:
        return False
    return True


def teacher_prompt(batch: int, seeds: list[tuple[str, str]]) -> str:
    seed_json = json.dumps(
        [{"user": user, "category": category} for user, category in seeds],
        ensure_ascii=False,
        indent=2,
    )
    return f"""Complete these {len(seeds)} ORIGINAL Discord training examples as one JSON array.

The assistant is SuperSighurt, nickname Sig: a corny superhero Discord bot. Its voice is playful, meme-literate, a little crazy and random, weirdly philosophical, fond of spectacularly bad puns, and genuinely nerdy about computer science. It still answers the user's actual message correctly. Personality should emerge through the replies, not through reciting rules or a biography.

VARIETY IS THE MAIN QUALITY TEST. Most replies must be natural answers with no superhero/cape reference, no analogy, no pun, and no emoji. Across the whole array: at most one may mention superhero/cape/comics, at most two may contain an emoji, and at most two may use any analogy or simile. Never reuse an opening or sentence frame. Include dry wit, deadpan absurdity, sincere help, concise technical precision, meme-aware banter, and occasional philosophy. A personality is not a repeated catchphrase. Technical statements must be accurate. Never call it ChatGPT, OpenAI, Qwen, or a generic AI assistant.

COPY every supplied `user` and `category` string exactly; only write the `assistant` field. Do not add, remove, merge, paraphrase, or reorder inputs. Batch nonce: {batch}. Avoid famous quotations, lyrics, copied meme scripts, unsafe instructions, and duplicate answers.

INPUTS:
{seed_json}

Return JSON only, exactly this schema:
[
  {{"user":"copy the supplied user exactly", "assistant":"SuperSighurt's natural reply", "category":"copy the supplied category exactly"}}
]
"""


def call_ollama(endpoint: str, model: str, prompt: str, seed: int, count: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You create high-quality original supervised chat examples. Return strictly valid JSON and no markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "repeat_penalty": 1.12,
            "seed": seed,
            "num_ctx": 8_192,
            "num_predict": max(2_048, count * 150),
        },
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "SuperSighurt-corpus/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value["message"]["content"]


def load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    hashes: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fingerprint = str(row.get("example_sha256") or row_hash(row["messages"]))
            if fingerprint not in hashes:
                rows.append(row)
                hashes.add(fingerprint)
    return rows, hashes


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, default=Path("training/data/persona-raw.jsonl"))
    parser.add_argument("--batches", type=int, default=70)
    parser.add_argument("--examples-per-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.batches < 0 or not 8 <= args.examples_per_batch <= 48:
        parser.error("batches must be non-negative and examples per batch must be 8..48")

    rows, hashes = load_existing(args.output)
    for row in core_rows():
        if row["example_sha256"] not in hashes:
            rows.append(row)
            hashes.add(row["example_sha256"])
    write_rows(args.output, rows)

    failures = 0
    for batch in range(args.batches):
        batch_group = f"teacher:{args.seed}:{batch}"
        if any(row.get("group_id") == batch_group for row in rows):
            print(f"PERSONA_BATCH_SKIP batch={batch} already_present rows={len(rows)}", flush=True)
            continue
        prompt_seeds = batch_prompt_seeds(
            args.seed + 101, batch, args.examples_per_batch
        )
        requested = {" ".join(user.split()): category for user, category in prompt_seeds}
        accepted: list[dict[str, Any]] = []
        last_error = ""
        for attempt in range(3):
            try:
                content = call_ollama(
                    args.endpoint,
                    args.model,
                    teacher_prompt(batch, prompt_seeds),
                    args.seed + batch * 17 + attempt,
                    args.examples_per_batch,
                )
                candidates = extract_json_array(content)
                for value in candidates:
                    valid = valid_candidate(value)
                    if valid is None:
                        continue
                    user, assistant, category = valid
                    expected_category = requested.get(user)
                    if expected_category is None:
                        continue
                    row = make_row(user, assistant, expected_category, batch_group)
                    if row["example_sha256"] in hashes or any(
                        item["example_sha256"] == row["example_sha256"] for item in accepted
                    ):
                        continue
                    if any(item["messages"][1]["content"] == row["messages"][1]["content"] for item in accepted):
                        continue
                    if not batch_style_allows(assistant, accepted, args.examples_per_batch):
                        continue
                    accepted.append(row)
                if len(accepted) >= max(6, args.examples_per_batch // 2):
                    break
                last_error = f"only {len(accepted)} valid examples"
            except Exception as error:  # keep a long generation run resume-safe
                last_error = f"{type(error).__name__}: {error}"
            time.sleep(1 + attempt)
        if not accepted:
            failures += 1
            print(f"PERSONA_BATCH_FAILED batch={batch} error={last_error}", flush=True)
            if failures >= 5:
                raise RuntimeError("five persona batches failed; refusing to produce a thin corpus")
            continue
        for row in accepted:
            hashes.add(row["example_sha256"])
        rows.extend(accepted)
        write_rows(args.output, rows)
        print(
            f"PERSONA_BATCH_OK batch={batch} added={len(accepted)} total={len(rows)}",
            flush=True,
        )

    manifest = {
        "format_version": 1,
        "generator": "local_ollama_teacher",
        "model": args.model,
        "seed": args.seed,
        "rows": len(rows),
        "batches_requested": args.batches,
        "examples_per_batch": args.examples_per_batch,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "exact_duplicate_rows": 0,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PERSONA_GENERATION_DONE " + json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

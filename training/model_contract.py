"""Lightweight, dependency-free constants shared by training and deployment gates."""

FOUNDATION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FOUNDATION_MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
GENERAL_DATASET_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
LOSS_WEIGHTING = "per_example_completion_mean"

EXPECTED_PROBE_IDS = frozenset(
    {
        "greeting",
        "identity_name",
        "identity_nickname",
        "identity_hero",
        "current_work",
        "capital",
        "rust",
        "nix",
        "dns",
        "pun",
        "meme",
        "keyboard_cat",
        "this_is_fine",
        "search_capability",
        "serious_support",
        "reply_context",
        "context_instruction",
        "reply_precision",
        "web_grounding",
    }
)

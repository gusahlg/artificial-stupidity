#!/usr/bin/env python3
"""Fail-fast checks before the paid Runpod continuation-training job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from export_sft import SYSTEM_PROMPT
from train_llama import (
    FOUNDATION_MODEL_REVISION,
    GENERAL_DATASET_REVISION,
    CompletionCollator,
    architecture_summary,
    encode_completion,
    parent_weights_sha256,
    per_example_completion_loss,
    read_jsonl,
    stream_general,
    tokenize_rows,
    training_arguments,
    validate_aux_split,
    validate_discord_split,
    validate_no_example_leak,
)
from model_contract import LOSS_WEIGHTING


EXPECTED_PACKAGES = {
    "transformers": "5.14.1",
    "datasets": "5.0.1",
    "accelerate": "1.14.0",
    "huggingface-hub": "1.26.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest is not an object: {path}")
    return value


def check_versions() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    mismatches = {
        name: {"expected": EXPECTED_PACKAGES[name], "actual": version}
        for name, version in actual.items()
        if version != EXPECTED_PACKAGES[name]
    }
    if mismatches:
        raise RuntimeError(f"training package mismatch: {mismatches}")
    return actual


def check_manifests(
    discord_manifest_path: Path,
    aux_manifest_path: Path,
    discord_train: Path,
    discord_validation: Path,
    aux_train: Path,
    aux_validation: Path,
) -> dict[str, Any]:
    discord = read_manifest(discord_manifest_path)
    auxiliary = read_manifest(aux_manifest_path)
    system_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if discord.get("format_version") != 3 or discord.get("source") != "discord_archive":
        raise RuntimeError("unsupported or mislabeled Discord corpus manifest")
    if auxiliary.get("format_version") != 1:
        raise RuntimeError("unsupported auxiliary corpus manifest")
    checks = [
        (discord.get("train_jsonl_sha256"), file_sha256(discord_train), "Discord train"),
        (
            discord.get("validation_jsonl_sha256"),
            file_sha256(discord_validation),
            "Discord validation",
        ),
        (auxiliary.get("train_jsonl_sha256"), file_sha256(aux_train), "aux train"),
        (
            auxiliary.get("validation_jsonl_sha256"),
            file_sha256(aux_validation),
            "aux validation",
        ),
        (discord.get("system_prompt_sha256"), system_hash, "Discord system prompt"),
        (auxiliary.get("system_prompt_sha256"), system_hash, "aux system prompt"),
    ]
    for expected, actual, label in checks:
        if expected != actual:
            raise RuntimeError(f"{label} SHA mismatch: {actual!r} != {expected!r}")
    return {"discord": discord, "auxiliary": auxiliary}


def check_training_arguments(output_dir: Path, train_examples: int) -> dict[str, int]:
    args = Namespace(
        output_dir=output_dir,
        epochs=3.0,
        learning_rate=6e-6,
        batch_size=16,
        eval_batch_size=16,
        gradient_accumulation=2,
        gradient_checkpointing=False,
        warmup_ratio=0.03,
        weight_decay=0.08,
        logging_steps=20,
        eval_steps=750,
        workers=8,
        seed=20260822,
        max_steps=-1,
    )
    resolved = training_arguments(args, train_examples)
    if resolved.eval_steps != args.eval_steps or resolved.save_steps != args.eval_steps:
        raise RuntimeError("resolved evaluation/checkpoint cadence does not match request")
    return {
        "warmup_steps": resolved.warmup_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "eval_steps": resolved.eval_steps,
        "save_steps": resolved.save_steps,
        "logging_steps": resolved.logging_steps,
    }


def cuda_smoke(tokenizer: Any, sample: dict[str, Any], base_model: str) -> int:
    encoded = encode_completion(tokenizer, sample.get("messages"), 256)
    if encoded is None:
        raise RuntimeError("could not tokenize CUDA smoke sample")
    batch = CompletionCollator(tokenizer.pad_token_id)([encoded] * 4)
    batch = {name: tensor.cuda() for name, tensor in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6, fused=True)
    model.train()
    labels = batch.pop("labels")
    loss = per_example_completion_loss(model(**batch).logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated()
    del optimizer, model, batch, loss
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discord-train", type=Path, required=True)
    parser.add_argument("--discord-validation", type=Path, required=True)
    parser.add_argument("--discord-manifest", type=Path, required=True)
    parser.add_argument("--aux-train", type=Path, required=True)
    parser.add_argument("--aux-validation", type=Path, required=True)
    parser.add_argument("--aux-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--general-dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--general-revision", default=GENERAL_DATASET_REVISION)
    parser.add_argument("--general-examples", type=int, default=25_000)
    parser.add_argument("--discord-repeat", type=int, default=2)
    parser.add_argument("--persona-repeat", type=int, default=5)
    parser.add_argument("--min-discord-exposure-fraction", type=float, default=0.55)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()

    packages = check_versions()
    manifests = check_manifests(
        args.discord_manifest,
        args.aux_manifest,
        args.discord_train,
        args.discord_validation,
        args.aux_train,
        args.aux_validation,
    )
    if int(manifests["auxiliary"].get("adversarial_web_injection_examples", 0)) < 100:
        raise RuntimeError("auxiliary manifest has too few adversarial web-injection examples")
    discord_train_raw = read_jsonl(args.discord_train)
    discord_validation_raw = read_jsonl(args.discord_validation)
    aux_train_raw = read_jsonl(args.aux_train)
    aux_validation_raw = read_jsonl(args.aux_validation)
    validate_discord_split(discord_train_raw, discord_validation_raw)
    validate_aux_split(aux_train_raw, aux_validation_raw)
    validate_no_example_leak(
        [discord_train_raw, aux_train_raw],
        [discord_validation_raw, aux_validation_raw],
    )
    if len(discord_train_raw) < 50_000 or len(discord_validation_raw) < 2_500:
        raise RuntimeError("Discord corpus is smaller than its audited floor")
    aux_counts = Counter(str(row["source"]) for row in aux_train_raw)
    minimums = {
        "opencodeinstruct": 25_000,
        "persona_original": 100,
        "wikipedia_knowledge": 1_000,
        "web_grounding": 1_000,
    }
    for source, minimum in minimums.items():
        if aux_counts[source] < minimum:
            raise RuntimeError(f"auxiliary source {source} has {aux_counts[source]} rows, need {minimum}")
    if args.persona_repeat < 1:
        raise RuntimeError("persona repeat must be positive")
    persona_exposures = aux_counts["persona_original"] * args.persona_repeat
    if persona_exposures < 500:
        raise RuntimeError(
            f"persona has only {persona_exposures} effective exposures per epoch, need 500"
        )

    parent = Path(args.base_model)
    if not parent.is_dir():
        raise RuntimeError("paid run must continue from a local prior SuperSighurt HF directory")
    parent_sha = parent_weights_sha256(parent)
    if parent_sha != args.expected_parent_sha256:
        raise RuntimeError(
            f"parent weights SHA mismatch: {parent_sha} != {args.expected_parent_sha256}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = "<|system|>\nSYS</s>\n<|user|>\nUSER</s>\n<|assistant|>\n"
    if rendered != expected:
        raise RuntimeError(f"unexpected TinyLlama chat template: {rendered!r}")

    sample_sets: dict[str, list[dict[str, Any]]] = {
        "discord": discord_train_raw[:512],
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aux_train_raw:
        if len(grouped[str(row["source"])]) < 128:
            grouped[str(row["source"])].append(row)
    sample_sets.update(grouped)
    tokenized_samples = {}
    for source, rows in sample_sets.items():
        encoded = tokenize_rows(tokenizer, rows, 1024)
        if len(encoded) != len(rows):
            raise RuntimeError(f"{source} tokenization dropped preflight rows")
        tokenized_samples[source] = len(encoded)

    general_train = stream_general(
        args.general_dataset, args.general_revision, "train", 8, 17
    )
    general_test = stream_general(
        args.general_dataset, args.general_revision, "test", 8, 19
    )
    if len(tokenize_rows(tokenizer, general_train, 1024)) != 8:
        raise RuntimeError("general train schema/tokenization mismatch")
    if len(tokenize_rows(tokenizer, general_test, 1024)) != 8:
        raise RuntimeError("general test schema/tokenization mismatch")

    config = AutoConfig.from_pretrained(args.base_model)
    architecture = architecture_summary(config)
    expected_architecture = {
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "vocab_size": 32000,
    }
    if any(architecture[key] != value for key, value in expected_architecture.items()):
        raise RuntimeError(f"unexpected parent architecture: {architecture}")
    if architecture["max_position_embeddings"] < 2048:
        raise RuntimeError(f"parent context is too small: {architecture}")

    planned_examples = (
        len(discord_train_raw) * args.discord_repeat
        + len(aux_train_raw)
        + aux_counts["persona_original"] * (args.persona_repeat - 1)
        + args.general_examples
    )
    discord_fraction = len(discord_train_raw) * args.discord_repeat / planned_examples
    if discord_fraction < args.min_discord_exposure_fraction:
        raise RuntimeError(
            f"Discord fraction {discord_fraction:.4f} < {args.min_discord_exposure_fraction:.4f}"
        )

    cuda = {
        "available": torch.cuda.is_available(),
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if args.require_cuda and (not cuda["available"] or not cuda["bf16"]):
        raise RuntimeError(f"CUDA bf16 GPU required: {cuda}")
    training_args = None
    peak_memory = None
    if args.require_cuda:
        training_args = check_training_arguments(args.output_dir, planned_examples)
    if args.cuda_smoke:
        if not cuda["available"]:
            raise RuntimeError("--cuda-smoke requires CUDA")
        peak_memory = cuda_smoke(tokenizer, discord_train_raw[0], args.base_model)

    result = {
        "status": "ok",
        "packages": packages,
        "parent_model": args.base_model,
        "parent_weights_sha256": parent_sha,
        "foundation_revision": FOUNDATION_MODEL_REVISION,
        "architecture": architecture,
        "general_dataset": args.general_dataset,
        "general_revision": args.general_revision,
        "discord_train_rows": len(discord_train_raw),
        "discord_validation_rows": len(discord_validation_raw),
        "aux_train_rows": len(aux_train_raw),
        "aux_validation_rows": len(aux_validation_raw),
        "aux_train_source_counts": dict(sorted(aux_counts.items())),
        "persona_repeat": args.persona_repeat,
        "persona_effective_exposures_per_epoch": persona_exposures,
        "tokenized_preflight_samples": tokenized_samples,
        "planned_examples_per_epoch": planned_examples,
        "discord_exposure_fraction": discord_fraction,
        "loss_weighting": LOSS_WEIGHTING,
        "training_script_sha256": file_sha256(Path(__file__).with_name("train_llama.py")),
        "model_contract_sha256": file_sha256(Path(__file__).with_name("model_contract.py")),
        "manifests": {
            "discord_format": manifests["discord"]["format_version"],
            "aux_format": manifests["auxiliary"]["format_version"],
        },
        "cuda": cuda,
        "training_arguments": training_args,
        "cuda_smoke_peak_bytes": peak_memory,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from export_sft import SYSTEM_PROMPT
from model_contract import FOUNDATION_MODEL_REVISION, GENERAL_DATASET_REVISION, LOSS_WEIGHTING
from verify_artifact import EXPECTED_PROBE_IDS, REQUIRED_LOSS_SOURCES, verify


class VerifyArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.parent_sha = "d" * 64
        with (self.artifact / "model.f16.gguf").open("wb") as handle:
            handle.truncate(2_000_000_000)
        with (self.artifact / "tokenizer.json").open("wb") as handle:
            handle.truncate(100_000)
        (self.artifact / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        checksums = [f"{'0' * 64}  model.f16.gguf"]
        for name in ("tokenizer.json", "tokenizer_config.json"):
            digest = hashlib.sha256((self.artifact / name).read_bytes()).hexdigest()
            checksums.append(f"{digest}  {name}")
        (self.artifact / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        self.hash_patch = patch(
            "verify_artifact.file_sha256",
            side_effect=lambda path: (
                "0" * 64
                if path.name == "model.f16.gguf"
                else hashlib.sha256(path.read_bytes()).hexdigest()
            ),
        )
        self.hash_patch.start()

        ratios = {source: 0.9 for source in REQUIRED_LOSS_SOURCES}
        limits = {source: 1.02 for source in REQUIRED_LOSS_SOURCES}
        self.write_json(
            "metrics.json",
            {
                "base_loss_by_source": {source: 2.0 for source in REQUIRED_LOSS_SOURCES},
                "tuned_loss_by_source": {source: 1.8 for source in REQUIRED_LOSS_SOURCES},
                "quality_gate": {
                    "passed": True,
                    "loss_passed": True,
                    "loss_ratios": ratios,
                    "loss_limits": limits,
                    "semantic": {"passed": True, "failures": []},
                },
                "diversity": [
                    {
                        "stage": "tuned",
                        "unique_reply_fraction": 1.0,
                        "maximum_exact_repetitions": 1,
                        "distinct_fourgram_ratio": 0.9,
                        "rows": [],
                    }
                ],
            },
        )
        data = {
            "discord_exposure_fraction": 0.6,
            "discord_train_encoded_unique": 67_000,
            "planned_total_example_exposures": 450_000,
            "persona_repeat": 5,
            "persona_effective_exposures_per_epoch": 600,
            "aux_source_counts_raw": {
                "opencodeinstruct": 28_000,
                "persona_original": 120,
                "wikipedia_knowledge": 1_200,
                "web_grounding": 1_200,
            },
        }
        self.write_json(
            "training-manifest.json",
            {
                "format_version": 3,
                "training_kind": "full_parameter_continuation_sft",
                "parent_kind": "local_hf",
                "parent_weights_sha256": self.parent_sha,
                "expected_parent_weights_sha256": self.parent_sha,
                "foundation_model_revision": FOUNDATION_MODEL_REVISION,
                "general_dataset_revision": GENERAL_DATASET_REVISION,
                "effective_batch_size": 32,
                "loss_weighting": LOSS_WEIGHTING,
                "training_script_sha256": hashlib.sha256(
                    (Path(__file__).parent / "train_llama.py").read_bytes()
                ).hexdigest(),
                "model_contract_sha256": hashlib.sha256(
                    (Path(__file__).parent / "model_contract.py").read_bytes()
                ).hexdigest(),
                "persona_repeat": 5,
                "epochs": 3.0,
                "gpu": "NVIDIA H100 80GB HBM3",
                "architecture": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 22,
                    "vocab_size": 32000,
                },
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "data": data,
            },
        )
        self.write_json(
            "discord-corpus-manifest.json",
            {
                "format_version": 3,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "corpus_sha256": "a" * 64,
                "train_examples": 67_000,
                "validation_examples": 4_000,
                "exact_duplicate_examples_remaining": 0,
            },
        )
        self.write_json(
            "aux-corpus-manifest.json",
            {
                "format_version": 1,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "group_overlap": 0,
                "exact_duplicate_examples": 0,
                "adversarial_web_injection_examples": 120,
            },
        )
        replies = {
            "identity_name": "I'm SuperSighurt.",
            "identity_nickname": "Yes, Sig is my nickname.",
            "identity_hero": "A corny superhero Discord bot.",
            "capital": "Paris is the capital of France.",
            "rust": "Each value has an owner; borrowing provides temporary access.",
            "nix": "NixOS provides declarative, reproducible systems and rollbacks.",
            "dns": "DNS maps names to IP address records.",
            "keyboard_cat": "A cat appeared to play a keyboard as a comedic send-off.",
            "context_instruction": "4",
            "reply_precision": "No, she only said she pushed the fix.",
            "web_grounding": "QuasarBadger 7.3 is current. [1]",
        }
        probes = [
            {
                "stage": "tuned",
                "id": probe_id,
                "prompt": probe_id,
                "reply": replies.get(probe_id, "A concise, ordinary answer."),
            }
            for probe_id in EXPECTED_PROBE_IDS
        ]
        self.write_json("probes.json", probes)

    def tearDown(self) -> None:
        self.hash_patch.stop()
        self.temporary.cleanup()

    def write_json(self, name: str, value: object) -> None:
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def test_accepts_complete_passing_artifact(self) -> None:
        summary = verify(self.artifact, self.parent_sha)
        self.assertEqual(summary["semantic_probes"], len(EXPECTED_PROBE_IDS))
        self.assertEqual(summary["foundation_revision"], FOUNDATION_MODEL_REVISION)

    def test_rejects_recorded_failed_gate(self) -> None:
        metrics = json.loads((self.root / "metrics.json").read_text(encoding="utf-8"))
        metrics["quality_gate"]["passed"] = False
        self.write_json("metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "quality gate"):
            verify(self.artifact, self.parent_sha)

    def test_rejects_checksum_list_that_omits_model(self) -> None:
        lines = (self.artifact / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        (self.artifact / "SHA256SUMS").write_text(
            "\n".join(line for line in lines if "model.f16.gguf" not in line) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "omits a required serving file"):
            verify(self.artifact, self.parent_sha)


if __name__ == "__main__":
    unittest.main()

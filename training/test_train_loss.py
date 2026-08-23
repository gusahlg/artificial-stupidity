#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from train_llama import ExampleBalancedTrainer, per_example_completion_loss


class PerExampleCompletionLossTests(unittest.TestCase):
    def test_long_completion_does_not_outvote_short_completion(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        labels = torch.tensor(
            [
                [-100, 0, -100, -100],
                [-100, 1, 1, -100],
            ]
        )
        actual = per_example_completion_loss(logits, labels)
        short = F.cross_entropy(logits[0, 0].unsqueeze(0), torch.tensor([0]))
        long = F.cross_entropy(
            logits[1, :2],
            torch.tensor([1, 1]),
            reduction="mean",
        )
        expected = (short + long) / 2
        self.assertTrue(torch.allclose(actual, expected))

        token_weighted = (short + long * 2) / 3
        self.assertFalse(torch.allclose(actual, token_weighted))

    def test_trainer_removes_labels_and_uses_example_balanced_loss(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
            ]
        )
        labels = torch.tensor([[-100, 0, -100], [-100, 1, 1]])

        class DummyModel:
            def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
                self.inputs = inputs
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": labels,
        }
        actual = ExampleBalancedTrainer.compute_loss(
            None,
            model,
            inputs,
            num_items_in_batch=torch.tensor(3),
        )
        self.assertNotIn("labels", model.inputs)
        self.assertTrue(torch.allclose(actual, per_example_completion_loss(logits, labels)))


if __name__ == "__main__":
    unittest.main()

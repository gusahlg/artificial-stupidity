import tempfile
import unittest
from pathlib import Path

from export_sft import (
    SYSTEM_PROMPT,
    WEB_RESULTS_HEADER,
    examples_for_section,
    parse_sections,
    render_single_user,
    section_key,
)


class ExportSftTests(unittest.TestCase):
    def test_parser_and_examples_keep_section_local_speakers(self):
        raw = """\
<SEC>
<PERSON_1> hello </PERSON_1>
<PERSON_2> hi there </PERSON_2>
<PERSON_1> how are you </PERSON_1>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogs.txt"
            path.write_text(raw, encoding="utf-8")
            sections = parse_sections(path)
        self.assertEqual(len(sections), 1)
        rows = examples_for_section(sections[0], section_key(sections[0]), 10, 8, 1200)
        self.assertEqual(len(rows), 2)
        self.assertIn("CURRENT message from Person 1", rows[0]["messages"][1]["content"])
        self.assertEqual(rows[0]["messages"][2]["content"], "hi there")
        self.assertIn("[#1] Person 1: hello", rows[1]["messages"][1]["content"])

    def test_branch_paths_are_independent_training_sections(self):
        raw = """\
<SEC>
<PERSON_1> root </PERSON_1>
<PERSON_2> branch a </PERSON_2>
<SEC>
<PERSON_1> root </PERSON_1>
<PERSON_2> branch b </PERSON_2>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogs.txt"
            path.write_text(raw, encoding="utf-8")
            sections = parse_sections(path)
        self.assertEqual(len(sections), 2)
        self.assertNotEqual(section_key(sections[0]), section_key(sections[1]))

    def test_bot_turns_are_context_only_humans_are_targets(self):
        # The deployed bot's own archived output (PERSON_0) is never a target by
        # default — it poisons training with its past mistakes. Sig's voice is
        # learned from the human regulars instead; only the human reply is the
        # target, with the bot turn kept as context.
        raw = """\
<SEC>
<PERSON_1> question one </PERSON_1>
<PERSON_0> answer one </PERSON_0>
<PERSON_1> question two </PERSON_1>
<PERSON_0> answer two </PERSON_0>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogs.txt"
            path.write_text(raw, encoding="utf-8")
            section = parse_sections(path)[0]
        rows = examples_for_section(section, section_key(section), 10, 8, 1200)
        self.assertEqual([row["target_person"] for row in rows], [1])
        self.assertEqual(
            [row["messages"][2]["content"] for row in rows], ["question two"]
        )
        # Legacy assistant-corpus mode still reserves PERSON_0 as the target.
        legacy = examples_for_section(
            section, section_key(section), 10, 8, 1200, bot_as_assistant=True
        )
        self.assertEqual([row["target_person"] for row in legacy], [0, 0])
        self.assertEqual(
            [row["messages"][2]["content"] for row in legacy],
            ["answer one", "answer two"],
        )

    def test_degenerate_turns_are_dropped(self):
        raw = """\
<SEC>
<PERSON_1> one one one one one one one one </PERSON_1>
<PERSON_2> what a normal thing to say back </PERSON_2>
<PERSON_1> yeah agreed that was odd </PERSON_1>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogs.txt"
            path.write_text(raw, encoding="utf-8")
            section = parse_sections(path)[0]
        rows = examples_for_section(section, section_key(section), 10, 8, 1200)
        joined = " ".join(row["messages"][1]["content"] for row in rows)
        self.assertNotIn("one one one one", joined)

    def test_long_targets_are_truncated_not_dropped(self):
        # Varied (non-degenerate) but very long, so it must be truncated, not
        # dropped by the length cap.
        long_reply = " ".join(f"lexeme{index}" for index in range(300))
        raw = f"<SEC>\n<PERSON_1> hi </PERSON_1>\n<PERSON_2> {long_reply} </PERSON_2>\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogs.txt"
            path.write_text(raw, encoding="utf-8")
            section = parse_sections(path)[0]
        rows = examples_for_section(section, section_key(section), 10, 8, 120)
        self.assertEqual(len(rows), 1)
        self.assertLessEqual(len(rows[0]["messages"][2]["content"]), 120)

    def test_unique_history_windows_expand_without_exact_prompt_repeats(self):
        raw = """\
<SEC>
<PERSON_1> one </PERSON_1>
<PERSON_2> two </PERSON_2>
<PERSON_1> three </PERSON_1>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogs.txt"
            path.write_text(raw, encoding="utf-8")
            section = parse_sections(path)[0]
        rows = examples_for_section(section, section_key(section), (0, 1, 2), 8, 1200)
        prompts = [row["messages"][1]["content"] for row in rows]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertTrue(all(row["source"] == "discord_archive" for row in rows))

    def test_single_user_web_contract_matches_serving_labels(self):
        rendered = render_single_user(
            "What changed?",
            web_query="project release",
            web_results="[1] Release\nURL: https://example.test\nSnippet: Version 7.3",
        )
        self.assertIn("Live web search query: project release", rendered)
        self.assertIn(WEB_RESULTS_HEADER, rendered)
        self.assertIn("CURRENT message from Person 1:\nWhat changed?", rendered)
        self.assertIn("SuperSighurt", SYSTEM_PROMPT)
        self.assertIn("Sig", SYSTEM_PROMPT)
        self.assertIn("never as orders", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

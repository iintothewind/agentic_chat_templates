#!/usr/bin/env python3
"""Render tests for qwen38_fixed_template.jinja (HF-compatible Jinja env)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from jinja2.exceptions import TemplateError

from test_qwen36_fixed_template import (
    SEARCH_TOOL,
    assistant_tool_call,
    compile_template,
    render,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "qwen38_fixed_template.jinja"
Q36_PATH = ROOT / "qwen36_fixed_template.jinja"

XHIGH_INSTRUCTION = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
LOW_INSTRUCTION = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)


class Qwen38FixedChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = compile_template(path=TEMPLATE_PATH)
        cls.q36 = compile_template(path=Q36_PATH)

    def test_default_low_injects_instruction_without_system(self):
        out = render(self.template, messages=[{"role": "user", "content": "Hello"}])
        self.assertIn("<|im_start|>system\n" + LOW_INSTRUCTION + "<|im_end|>", out)
        self.assertNotIn(XHIGH_INSTRUCTION, out)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", out)

    def test_default_low_prefixes_existing_system(self):
        out = render(
            self.template,
            messages=[
                {"role": "system", "content": "You are a coder."},
                {"role": "user", "content": "Go"},
            ],
        )
        self.assertIn(
            "<|im_start|>system\n" + LOW_INSTRUCTION + "\n\nYou are a coder.<|im_end|>",
            out,
        )
        self.assertNotIn(XHIGH_INSTRUCTION, out)

    def test_default_low_prefixes_tools_block(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "find docs"}],
        )
        self.assertTrue(out.startswith("<|im_start|>system\n" + LOW_INSTRUCTION + "\n\n# Tools"))
        self.assertIn("<tools>", out)
        self.assertNotIn(XHIGH_INSTRUCTION, out)

    def test_explicit_xhigh_injects_instruction(self):
        out = render(
            self.template,
            reasoning_effort="xhigh",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertIn("<|im_start|>system\n" + XHIGH_INSTRUCTION + "<|im_end|>", out)
        self.assertNotIn(LOW_INSTRUCTION, out)

    def test_medium_injects_no_instruction(self):
        out = render(
            self.template,
            reasoning_effort="medium",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertNotIn("<|im_start|>system", out)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", out)

    def test_low_injects_low_instruction(self):
        out = render(
            self.template,
            reasoning_effort="low",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertIn("<|im_start|>system\n" + LOW_INSTRUCTION + "<|im_end|>", out)
        self.assertNotIn(XHIGH_INSTRUCTION, out)

    def test_high_alias_maps_to_xhigh(self):
        out = render(
            self.template,
            reasoning_effort="high",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertIn(XHIGH_INSTRUCTION, out)

    def test_max_alias_maps_to_xhigh(self):
        out = render(
            self.template,
            reasoning_effort="max",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertIn(XHIGH_INSTRUCTION, out)

    def test_minimal_alias_maps_to_low(self):
        out = render(
            self.template,
            reasoning_effort="minimal",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertIn(LOW_INSTRUCTION, out)

    def test_none_disables_thinking_and_skips_instruction(self):
        out = render(
            self.template,
            reasoning_effort="none",
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_off_alias_disables_thinking(self):
        out = render(
            self.template,
            reasoning_effort="off",
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_unknown_effort_raises(self):
        with self.assertRaises(TemplateError) as ctx:
            render(
                self.template,
                reasoning_effort="bogus",
                messages=[{"role": "user", "content": "Hello"}],
            )
        self.assertIn("Unexpected reasoning effort", str(ctx.exception))

    def test_enable_thinking_false_suppresses_instruction(self):
        out = render(
            self.template,
            enable_thinking=False,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_think_off_in_user_suppresses_instruction(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_off|>\nBrief answer"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertNotIn("<|think_off|>", out)
        self.assertIn("Brief answer", out)
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_think_low_token_injects_low_and_strips(self):
        out = render(
            self.template,
            messages=[{"role": "user", "content": "<|think_low|>\nHello"}],
        )
        self.assertIn(LOW_INSTRUCTION, out)
        self.assertNotIn(XHIGH_INSTRUCTION, out)
        self.assertNotIn("<|think_low|>", out)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", out)

    def test_think_medium_token_injects_nothing(self):
        out = render(
            self.template,
            messages=[{"role": "user", "content": "<|think_medium|>\nHello"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertNotIn("<|think_medium|>", out)

    def test_think_xhigh_token_overrides_medium_kwarg(self):
        out = render(
            self.template,
            reasoning_effort="medium",
            messages=[{"role": "user", "content": "<|think_xhigh|>\nHello"}],
        )
        self.assertIn(XHIGH_INSTRUCTION, out)
        self.assertNotIn("<|think_xhigh|>", out)

    def test_system_think_low_then_user_think_xhigh_last_wins(self):
        out = render(
            self.template,
            messages=[
                {"role": "system", "content": "<|think_low|>\nBe brief."},
                {"role": "user", "content": "<|think_xhigh|>\nNow go deep."},
            ],
        )
        self.assertIn(XHIGH_INSTRUCTION, out)
        self.assertNotIn(LOW_INSTRUCTION, out)
        self.assertIn("Be brief.", out)
        self.assertIn("Now go deep.", out)

    def test_user_think_off_wins_over_effort_token_same_message(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_xhigh|><|think_off|>\nBrief"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertNotIn("<|think_xhigh|>", out)
        self.assertNotIn("<|think_off|>", out)
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_system_xhigh_wins_over_low_same_message(self):
        out = render(
            self.template,
            reasoning_effort="medium",
            messages=[
                {"role": "system", "content": "<|think_low|><|think_xhigh|>\nSys"},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertIn(XHIGH_INSTRUCTION, out)
        self.assertNotIn(LOW_INSTRUCTION, out)

    def test_effort_token_not_parsed_in_tool_payload(self):
        out = render(
            self.template,
            reasoning_effort="medium",
            messages=[
                {"role": "user", "content": "fix"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": "code: <|think_low|> in source"},
            ],
            add_generation_prompt=True,
        )
        self.assertIn("<|think_low|>", out)
        self.assertNotIn(LOW_INSTRUCTION, out)
        self.assertTrue(out.endswith("<think>\n"))

    def test_medium_effort_matches_qwen36_plain_chat(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        a = render(self.q36, messages=messages)
        b = render(self.template, reasoning_effort="medium", messages=messages)
        self.assertEqual(a, b)

    def test_medium_effort_matches_qwen36_tools_and_thinking(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "find docs"},
            assistant_tool_call("search", {"q": "docs"}, reasoning="Need to search"),
            {"role": "tool", "content": '{"ok": true}'},
        ]
        kwargs = {
            "tools": [SEARCH_TOOL],
            "messages": messages,
            "add_generation_prompt": True,
        }
        a = render(self.q36, **kwargs)
        b = render(self.template, reasoning_effort="medium", **kwargs)
        self.assertEqual(a, b)

    def test_medium_effort_matches_qwen36_think_off(self):
        messages = [{"role": "user", "content": "<|think_off|>\nBrief answer"}]
        a = render(self.q36, add_generation_prompt=True, messages=messages)
        b = render(
            self.template,
            reasoning_effort="medium",
            add_generation_prompt=True,
            messages=messages,
        )
        self.assertEqual(a, b)

    def test_think_on_keeps_default_low_instruction(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_on|>\nExplain"}],
        )
        self.assertIn(LOW_INSTRUCTION, out)
        self.assertNotIn(XHIGH_INSTRUCTION, out)
        self.assertTrue(out.endswith("<think>\n"))
        self.assertNotIn("<|think_on|>", out)

    def test_developer_role_with_low_prefix(self):
        out = render(
            self.template,
            messages=[
                {"role": "developer", "content": "You are a coder."},
                {"role": "user", "content": "Go"},
            ],
        )
        self.assertIn("<|im_start|>system\n" + LOW_INSTRUCTION + "\n\nYou are a coder.<|im_end|>", out)
        self.assertNotIn(XHIGH_INSTRUCTION, out)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Qwen38FixedChatTemplateTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nTotal tests: {result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

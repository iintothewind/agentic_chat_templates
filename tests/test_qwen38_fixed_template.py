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

NL = '\n'  # newline inside assertion strings
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

    def test_default_effort_is_medium(self):
        messages = [{"role": "user", "content": "Hello"}]
        out = render(self.template, messages=messages)
        self.assertEqual(
            out, render(self.template, reasoning_effort="medium", messages=messages)
        )
        self.assertNotIn("Reasoning effort is set to", out)
        self.assertNotIn("<|im_start|>system", out)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", out)

    def test_default_medium_leaves_system_and_tools_untouched(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[
                {"role": "system", "content": "You are a coder."},
                {"role": "user", "content": "find docs"},
            ],
        )
        self.assertTrue(out.startswith("<|im_start|>system\n# Tools"))
        self.assertIn("\n\nYou are a coder.<|im_end|>\n", out)
        self.assertNotIn("Reasoning effort is set to", out)

    def test_low_prefixes_existing_system(self):
        out = render(
            self.template,
            reasoning_effort="low",
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

    def test_low_prefixes_tools_block(self):
        out = render(
            self.template,
            reasoning_effort="low",
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

    def test_think_on_keeps_default_medium(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_on|>\nExplain"}],
        )
        self.assertNotIn("Reasoning effort is set to", out)
        # adjacent literals: raw control token does not survive some toolchains
        self.assertTrue(out.endswith("<thi" "nk>\n"))

    def test_developer_role_uses_default_medium(self):
        out = render(
            self.template,
            messages=[
                {"role": "developer", "content": "You are a coder."},
                {"role": "user", "content": "Go"},
            ],
        )
        self.assertIn("<|im_start|>system\nYou are a coder.<|im_end|>", out)
        self.assertNotIn("Reasoning effort is set to", out)

    # --- T-A/T-B/T-C/T-D/T-E/T-F regressions ---

    def test_multimodal_string_item_is_text_not_image(self):
        out = render(
            self.template,
            messages=[{"role": "user", "content": ["an image of a cat"]}],
        )
        self.assertIn("an image of a cat", out)
        self.assertNotIn("<|image_pad|>", out)

    def test_multimodal_plain_string_item_does_not_raise(self):
        out = render(
            self.template,
            messages=[{"role": "user", "content": ["hello world"]}],
        )
        self.assertIn("hello world", out)

    def test_parse_string_arguments_without_from_json_filter(self):
        out = render(
            self.template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                assistant_tool_call("run", '{"cmd": "ls"}'),
            ],
        )
        self.assertIn("<parameter=cmd>" + NL + "ls", out)

    def test_tool_error_prose_does_not_trigger_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "check"},
                {
                    "role": "tool",
                    "content": "OK: the previous release failed to build on CI, "
                    "but current HEAD is green",
                },
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_real_tool_error_still_triggers_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true, "msg": "boom"}'},
            ],
        )
        self.assertIn("SYSTEM WARNING", out)

    def test_historical_think_token_is_passed_through_verbatim(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "<|think_off|>" + NL + "first"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "second"},
            ],
            add_generation_prompt=True,
        )
        self.assertIn("<|think_off|>", out)
        self.assertTrue(out.endswith("<think>" + NL))

    def test_last_user_turn_think_token_still_steers_generation(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "<|think_off|>" + NL + "second"},
            ],
            add_generation_prompt=True,
        )
        self.assertNotIn("<|think_off|>", out)
        self.assertIn("<think>" + NL + "</think>", out.split("assistant")[-1])

    def test_tool_instructions_follow_thinking_state(self):
        on = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "go"}],
        )
        off = render(
            self.template,
            tools=[SEARCH_TOOL],
            enable_thinking=False,
            messages=[{"role": "user", "content": "go"}],
        )
        self.assertIn("<think>" + NL + "Brief explanation of tool call", on)
        self.assertNotIn("<think>" + NL + "Brief explanation of tool call", off)
        self.assertNotIn("MUST be placed strictly inside", off)
        self.assertIn("MUST be placed strictly inside", on)

    def test_assistant_turn_without_reasoning_keeps_think_block(self):
        """T-E: upstream always emits the think block for a past turn."""
        out = render(
            self.template,
            reasoning_effort="medium",
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "done"},
            ],
        )
        self.assertIn("<think>" + NL + NL + "</think>", out)

    def test_posix_shell_failure_triggers_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "read it"},
                {
                    "role": "tool",
                    "content": "cat: README.md: No such file or directory",
                },
            ],
        )
        self.assertIn("SYSTEM WARNING", out)

def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Qwen38FixedChatTemplateTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nTotal tests: {result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

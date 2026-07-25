#!/usr/bin/env python3
"""Render tests for qwen_fixed_template.jinja (HF-compatible Jinja env)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "qwen_fixed_template.jinja"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


def compile_template(*, with_from_json: bool = False):
    """Mirror HuggingFace chat-template Jinja setup; optional LM Studio from_json."""
    source = TEMPLATE_PATH.read_text(encoding="utf-8")

    def raise_exception(message: str):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, **kwargs):
        return json.dumps(x, ensure_ascii=ensure_ascii, **kwargs)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[loopcontrols],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    if with_from_json:
        env.filters["from_json"] = json.loads
    return env.from_string(source)


def render(template, *, add_generation_prompt=False, **kwargs) -> str:
    return template.render(add_generation_prompt=add_generation_prompt, **kwargs)


def assistant_tool_call(name: str, arguments, *, content: str = "", reasoning: str = ""):
    msg = {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


class QwenFixedChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = compile_template()
        cls.lmstudio_template = compile_template(with_from_json=True)

    # --- basics ---

    def test_plain_user_assistant(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )
        self.assertIn("<|im_start|>user\nHello<|im_end|>", out)
        self.assertIn("<|im_start|>assistant\nHi there<|im_end|>", out)

    def test_empty_messages_raises(self):
        with self.assertRaises(TemplateError):
            render(self.template, messages=[])

    def test_empty_assistant_turn(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": ""},
            ],
        )
        self.assertIn("<|im_start|>assistant\n<|im_end|>", out)

    def test_unexpected_role_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "function", "content": "legacy"},
                ],
            )

    def test_developer_role_as_system(self):
        out = render(
            self.template,
            messages=[
                {"role": "developer", "content": "You are a coder."},
                {"role": "user", "content": "Go"},
            ],
        )
        self.assertIn("<|im_start|>system\nYou are a coder.<|im_end|>", out)

    def test_tools_plus_system_message(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "find docs"},
            ],
        )
        self.assertIn("<tools>", out)
        self.assertIn("Be concise.", out)
        self.assertEqual(out.count("<|im_start|>system"), 1)

    # --- tool arguments ---

    def test_tools_with_mapping_arguments(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[
                {"role": "user", "content": "find docs"},
                assistant_tool_call("search", {"q": "docs", "limit": None}, reasoning="Need to search"),
            ],
        )
        self.assertIn("<tools>", out)
        self.assertIn('"name": "search"', out)
        self.assertIn("<parameter=q>", out)
        self.assertIn("docs", out)
        self.assertIn("<parameter=limit>", out)
        limit_val = out.split("<parameter=limit>")[1].split("</parameter>")[0].strip()
        self.assertEqual(limit_val, "null")

    def test_mapping_arguments_nested_and_list_use_tojson(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                assistant_tool_call(
                    "cfg",
                    {"meta": {"k": "v"}, "tags": ["a", "b"], "note": "plain"},
                ),
            ],
        )
        self.assertIn('<parameter=meta>\n{"k": "v"}', out)
        self.assertIn('<parameter=tags>\n["a", "b"]', out)
        self.assertIn("<parameter=note>\nplain", out)

    def test_string_arguments_fallback(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                assistant_tool_call("run", '{"cmd": "ls"}'),
            ],
        )
        self.assertIn('{"cmd": "ls"}', out)
        self.assertNotIn("<parameter=cmd>", out)

    def test_strict_tool_arguments_raises(self):
        with self.assertRaises(TemplateError) as ctx:
            render(
                self.template,
                strict_tool_arguments=True,
                messages=[
                    {"role": "user", "content": "run"},
                    assistant_tool_call("run", '{"cmd": "ls"}'),
                ],
            )
        self.assertIn("mapping", str(ctx.exception).lower())

    def test_parse_string_arguments_without_from_json_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                parse_string_arguments=True,
                messages=[
                    {"role": "user", "content": "run"},
                    assistant_tool_call("run", '{"cmd": "ls"}'),
                ],
            )

    def test_parse_string_arguments_expands_parameters(self):
        out = render(
            self.lmstudio_template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                assistant_tool_call("run", '{"cmd": "ls", "cwd": "/tmp"}'),
            ],
        )
        self.assertIn("<parameter=cmd>\nls", out)
        self.assertIn("<parameter=cwd>\n/tmp", out)
        self.assertNotIn('{"cmd": "ls"', out)

    def test_parse_string_arguments_non_object_json_falls_back_raw(self):
        out = render(
            self.lmstudio_template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                assistant_tool_call("run", '"just-a-string"'),
            ],
        )
        self.assertIn('"just-a-string"', out)
        self.assertNotIn("<parameter=", out)

    def test_empty_string_arguments_skipped(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                assistant_tool_call("noop", "   "),
            ],
        )
        self.assertIn("<function=noop>\n</function>", out)

    def test_empty_mapping_arguments(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                assistant_tool_call("noop", {}),
            ],
        )
        self.assertIn("<function=noop>\n</function>", out)

    def test_multiple_parallel_tool_calls(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "reasoning_content": "two tools",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "a", "arguments": {"x": 1}}},
                        {"type": "function", "function": {"name": "b", "arguments": {"y": 2}}},
                    ],
                },
            ],
        )
        self.assertEqual(out.count("<tool_call>"), 2)
        self.assertIn("<function=a>", out)
        self.assertIn("<function=b>", out)
        self.assertIn("\n\n<tool_call>", out)  # first call after body

    def test_first_tool_call_spacing_with_body(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                assistant_tool_call("x", {}, content="see below"),
            ],
        )
        self.assertIn("see below\n\n<tool_call>", out)

    def test_tool_call_stripped_from_content_when_tool_calls_field_present(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                assistant_tool_call(
                    "x",
                    {},
                    content="prefix<tool_call>\n<function=x>\n</function>\n</tool_call>",
                ),
            ],
        )
        self.assertIn("<|im_start|>assistant\nprefix\n\n<tool_call>", out)
        self.assertEqual(out.count("<tool_call>"), 1)

    # --- thinking control ---

    def test_think_off_in_user_disables_gen_thinking(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_off|>\nBrief answer"}],
        )
        self.assertIn("<think>\n</think>", out)

    def test_think_on_in_user_keeps_gen_thinking_open(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_on|>\nExplain"}],
        )
        self.assertTrue(out.endswith("<think>\n"))

    def test_think_off_wins_on_user_when_both_tokens(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_on|><|think_off|>\nBrief"}],
        )
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_think_on_wins_on_system_when_both_tokens(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[
                {"role": "system", "content": "<|think_off|><|think_on|>\nSys"},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertTrue(out.endswith("<think>\n"))

    def test_enable_thinking_kwarg_false_disables_gen_prompt(self):
        out = render(
            self.template,
            enable_thinking=False,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIn("<think>\n</think>", out.split("<|im_start|>assistant")[-1])

    def test_enable_thinking_false_suppresses_historical_thinking_blocks(self):
        out = render(
            self.template,
            enable_thinking=False,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer", "reasoning_content": "secret"},
            ],
        )
        self.assertNotIn("secret", out)
        self.assertIn("<|im_start|>assistant\nanswer", out)

    def test_think_token_not_parsed_in_tool_payload(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "fix"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": 'code: <|think_off|> in source\n"error": true'},
            ],
            add_generation_prompt=True,
        )
        self.assertIn("<|think_off|>", out)
        self.assertTrue(out.endswith("<think>\n"))

    def test_preserve_thinking_default_keeps_old_reasoning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": "done", "reasoning_content": "thought-A"},
                {"role": "user", "content": "step 2"},
                {"role": "assistant", "content": "done2", "reasoning_content": "thought-B"},
            ],
        )
        self.assertIn("thought-A", out)
        self.assertIn("thought-B", out)

    def test_preserve_thinking_true_keeps_old_reasoning(self):
        out = render(
            self.template,
            preserve_thinking=True,
            messages=[
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": "done", "reasoning_content": "thought-A"},
                {"role": "user", "content": "step 2"},
                {"role": "assistant", "content": "done2", "reasoning_content": "thought-B"},
            ],
        )
        self.assertIn("thought-A", out)

    def test_preserve_thinking_false_prunes_old_reasoning(self):
        out = render(
            self.template,
            preserve_thinking=False,
            messages=[
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": "done", "reasoning_content": "thought-A"},
                {"role": "user", "content": "step 2"},
                {"role": "assistant", "content": "done2", "reasoning_content": "thought-B"},
            ],
        )
        self.assertNotIn("thought-A", out)
        self.assertIn("thought-B", out)

    def test_assistant_strips_leaked_think_tokens(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer with <|think_off|> echoed"},
            ],
        )
        self.assertNotIn("<|think_off|>", out)
        self.assertIn("answer with  echoed", out)

    def test_reasoning_content_dedup_strips_embedded_thinking(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "<think>duplicate</think>\nvisible",
                    "reasoning_content": "canonical",
                },
            ],
        )
        self.assertEqual(out.count("canonical"), 1)
        self.assertNotIn("duplicate", out)
        self.assertIn("visible", out)

    def test_reasoning_parsed_from_closed_redacted_thinking_in_content(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "<think>\ninner\n</think>\nreply",
                },
            ],
        )
        self.assertIn("inner", out)
        self.assertIn("reply", out)
        self.assertNotIn("inner\n</think>\nreply", out.replace("<think>\ninner\n</think>\n", ""))

    def test_reasoning_parsed_from_thinking_variant(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plan\n</thinking>\nanswer"},
            ],
        )
        self.assertIn("plan", out)
        self.assertIn("answer", out)

    def test_reasoning_parsed_from_whitespace_think_variant(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "plan\n</ think>\nanswer"},
            ],
        )
        self.assertIn("plan", out)
        self.assertIn("answer", out)

    def test_unclosed_thinking_with_tool_call_rescue(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "call tool"},
                {
                    "role": "assistant",
                    "content": "<think>\nplanning\n<tool_call>\n<function=foo>\n</function>\n</tool_call>",
                },
            ],
        )
        self.assertIn("planning", out)
        self.assertIn("<tool_call>", out)

    def test_first_tool_call_spacing_no_body(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                assistant_tool_call("x", {}),
            ],
        )
        self.assertIn("<|im_start|>assistant\n<tool_call>", out)

    # --- tool responses / errors ---

    def test_tool_error_warning_injected(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true, "msg": "fail"}'},
            ],
        )
        self.assertIn("SYSTEM WARNING", out)

    def test_error_null_does_not_trigger_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": null, "data": {"ok": true}}'},
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_long_error_content_skips_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true, "detail": "' + ("x" * 600) + '"}'},
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_inject_tool_error_warnings_disabled(self):
        out = render(
            self.template,
            inject_tool_error_warnings=False,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true}'},
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_multiple_tool_responses_share_user_turn(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run both"},
                {"role": "tool", "content": "result-a"},
                {"role": "tool", "content": "result-b"},
            ],
        )
        # One real user turn + one tool-response wrapper opened by the first tool message.
        self.assertEqual(out.count("<|im_start|>user"), 2)
        self.assertEqual(out.count("<tool_response>"), 2)
        self.assertIn("result-a", out)
        self.assertIn("result-b", out)
        # Second tool must not open another user block.
        self.assertEqual(out.count("<|im_start|>user\n<tool_response>"), 1)

    def test_tool_response_closes_before_next_user(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "tool", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        )
        self.assertIn("</tool_response><|im_end|>\n<|im_start|>user\nsecond", out)

    def test_consecutive_tool_errors_empty_gen_thinking(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true}'},
                {"role": "tool", "content": '{"error": true}'},
            ],
        )
        tail = out.split("<|im_start|>assistant")[-1]
        self.assertIn("<think>\n</think>", tail)

    def test_single_tool_error_keeps_open_gen_thinking(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true}'},
            ],
        )
        tail = out.split("<|im_start|>assistant")[-1]
        self.assertTrue(tail.endswith("<think>\n"))
        self.assertNotIn("</think>", tail)

    def test_user_message_resets_tool_error_counter(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "content": '{"error": true}'},
                {"role": "user", "content": "retry"},
            ],
        )
        tail = out.split("<|im_start|>assistant")[-1]
        self.assertTrue(tail.endswith("<think>\n"))

    # --- validation ---

    def test_mid_conversation_system_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "late system"},
                ],
            )

    def test_mid_conversation_developer_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "developer", "content": "late dev"},
                ],
            )

    def test_no_user_query_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                messages=[{"role": "tool", "content": "<tool_response>only</tool_response>"}],
            )

    def test_system_image_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "image", "image": "abc"}],
                    },
                    {"role": "user", "content": "hi"},
                ],
            )

    def test_unexpected_content_type_raises(self):
        with self.assertRaises(TemplateError):
            render(
                self.template,
                messages=[{"role": "user", "content": 12345}],
            )

    # --- kwargs / tools block ---

    def test_verbose_tool_instructions_default(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIn("Do NOT omit the opening <tool_call> tag", out)

    def test_verbose_tool_instructions_false_upstream(self):
        out = render(
            self.template,
            verbose_tool_instructions=False,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertNotIn("Do NOT omit the opening <tool_call> tag", out)
        self.assertIn("Required parameters MUST be specified", out)

    def test_unwrap_tool_envelope_false(self):
        out = render(
            self.template,
            unwrap_tool_envelope=False,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIn('"type": "function"', out)

    def test_unwrap_tool_envelope_default_uses_inner_function(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIn('"name": "search"', out)
        self.assertNotIn('"type": "function"', out)

    # --- vision ---

    def test_user_image_content(self):
        out = render(
            self.template,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image", "image": "abc"},
                    ],
                },
            ],
        )
        self.assertIn("what is this?", out)
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", out)

    def test_add_vision_id_labels_images(self):
        out = render(
            self.template,
            add_vision_id=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "a"},
                        {"type": "image", "image": "b"},
                    ],
                },
            ],
        )
        self.assertIn("Picture 1:", out)
        self.assertIn("Picture 2:", out)

    def test_user_video_content(self):
        out = render(
            self.template,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "clip"}],
                },
            ],
        )
        self.assertIn("<|vision_start|><|video_pad|><|vision_end|>", out)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(QwenFixedChatTemplateTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\nTotal tests: {result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

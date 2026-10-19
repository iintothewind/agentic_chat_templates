#!/usr/bin/env python3
"""Tests for gemma4_fixed_template.jinja (HF-compatible Jinja env)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "gemma4_fixed_template.jinja"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "OBJECT",
            "properties": {"q": {"type": "STRING", "description": "query"}},
            "required": ["q"],
        },
    },
}

BARE_TOOL = {
    "name": "search",
    "description": "Search the web",
    "parameters": {
        "type": "OBJECT",
        "properties": {"q": {"type": "STRING", "description": "query"}},
    },
}


def compile_template(*, with_from_json: bool = False):
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
    defaults = {"bos_token": "", "messages": [{"role": "user", "content": "hi"}]}
    defaults.update(kwargs)
    return template.render(add_generation_prompt=add_generation_prompt, **defaults)



def _call_fragment(text):
    """Extract the argument fragment from a rendered tool call."""
    start = text.index("call:")
    start = text.index("{", start) + 1
    end = text.index("}<tool_call|>", start)
    return text[start:end]


class Gemma4FixedChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = compile_template()
        cls.lmstudio_template = compile_template(with_from_json=True)

    # --- basics ---

    def test_plain_user_turn(self):
        out = render(self.template, messages=[{"role": "user", "content": "Hello"}])
        self.assertIn("<|turn>user\nHello", out)

    def test_empty_messages_raises(self):
        with self.assertRaises(TemplateError):
            render(self.template, messages=[])

    def test_agent_defaults_true_injects_system_think_for_user_only_chat(self):
        out = render(self.template, messages=[{"role": "user", "content": "hi"}])
        self.assertIn("<|turn>system", out)
        self.assertIn("<|think|>", out)

    def test_agent_defaults_false_skips_system_think_for_user_only_chat(self):
        out = render(
            self.template,
            agent_defaults=False,
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertNotIn("<|turn>system", out)
        self.assertNotIn("<|think|>", out)

    # --- tool arguments ---

    def test_mapping_arguments_with_null(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "run",
                                "arguments": {"cmd": "ls", "lang": None},
                            },
                        }
                    ],
                },
            ],
        )
        self.assertIn("call:run{", out)
        self.assertIn("cmd:", out)
        self.assertIn("lang:null", out)

    def test_string_arguments_lenient_default(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                },
            ],
        )
        self.assertIn('call:run{{"cmd": "ls"}}', out)

    def test_strict_tool_arguments_raises(self):
        with self.assertRaises(TemplateError) as ctx:
            render(
                self.template,
                strict_tool_arguments=True,
                messages=[
                    {"role": "user", "content": "run"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
                            }
                        ],
                    },
                ],
            )
        self.assertIn("string", str(ctx.exception).lower())

    def test_parse_string_arguments_expands(self):
        out = render(
            self.lmstudio_template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "run", "arguments": '{"cmd": "ls", "cwd": "/tmp"}'},
                        }
                    ],
                },
            ],
        )
        self.assertIn("cmd:", out)
        self.assertIn("cwd:", out)
        self.assertNotIn('{"cmd": "ls"', out)

    def test_parse_string_arguments_without_from_json_filter(self):
        """transformers/vLLM/SGLang have no from_json filter.

        The template must expand JSON string arguments on its own.
        """
        out = render(
            self.template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                },
            ],
        )
        self.assertIn('cmd:<|"|>ls<|"|>', out)
        self.assertNotIn('{"cmd": "ls"}', out)

    def test_parse_string_arguments_matches_mapping_path(self):
        """The JSON-string path must equal the official mapping path.

        The official Gemma 4 template renders an already-deserialized
        mapping. Expanding a JSON string must produce the same fragment.
        """
        args = {"cmd": "ls", "cwd": "/tmp", "n": 42, "b": True}
        as_dict = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "run", "arguments": args},
                        }
                    ],
                },
            ],
        )
        as_json = render(
            self.template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "run",
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
            ],
        )
        self.assertEqual(_call_fragment(as_dict), _call_fragment(as_json))

    def test_parse_string_arguments_decodes_unicode_escapes(self):
        """json.dumps defaults to ensure_ascii=True, so CJK arrives escaped.
        """
        out = render(
            self.template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "run",
                                "arguments": '{"path": "\\u4f60\\u597d"}',
                            },
                        }
                    ],
                },
            ],
        )
        self.assertIn("path:<|\"|>\u4f60\u597d<|\"|>", out)

    def test_unwrap_bare_function_spec(self):
        out = render(
            self.template,
            tools=[BARE_TOOL],
            messages=[{"role": "user", "content": "search"}],
        )
        self.assertIn("declaration:search", out)

    def test_openai_tool_envelope(self):
        out = render(
            self.template,
            tools=[SEARCH_TOOL],
            messages=[{"role": "user", "content": "search"}],
        )
        self.assertIn("declaration:search", out)

    # --- thinking / preserve ---

    def test_reasoning_without_tool_calls_preserved_by_default(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": "done", "reasoning_content": "thought-A"},
                {"role": "user", "content": "step 2"},
            ],
        )
        self.assertIn("<|channel>thought\nthought-A\n<channel|>", out)

    def test_reasoning_without_tool_calls_dropped_when_preserve_false(self):
        out = render(
            self.template,
            preserve_thinking=False,
            messages=[
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": "done", "reasoning_content": "thought-A"},
                {"role": "user", "content": "step 2"},
            ],
        )
        self.assertNotIn("thought-A", out)

    def test_think_off_in_user_disables_gen_thinking(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[{"role": "user", "content": "<|think_off|>\nBrief"}],
        )
        self.assertIn("<|channel>thought\n<channel|>", out.split("<|turn>model")[-1])

    def test_think_token_not_parsed_in_tool_payload(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "fix"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "code: <|think_off|> literal"},
            ],
            add_generation_prompt=True,
        )
        self.assertIn("<|think_off|>", out)

    def test_assistant_strips_leaked_think_tokens(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer with <|think_off|> echoed"},
            ],
        )
        self.assertNotIn("<|think_off|>", out)

    # --- tool errors ---

    def test_tool_error_warning_injected(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"error": true}'},
            ],
        )
        self.assertIn("SYSTEM WARNING", out)

    def test_error_null_does_not_trigger_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"error": null, "data": 1}'},
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_inject_tool_error_warnings_disabled(self):
        out = render(
            self.template,
            inject_tool_error_warnings=False,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"error": true}'},
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_consecutive_tool_errors_escalate_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "run", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"error": true}'},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c2", "type": "function", "function": {"name": "run", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "content": '{"error": true}'},
            ],
        )
        self.assertIn("2 consecutive tool errors detected", out)

    # --- P5 / P8 continuation & tool-loop protocol ---

    def test_consecutive_assistant_text_continues_same_model_turn(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "part one"},
                {"role": "assistant", "content": "part two"},
            ],
        )
        self.assertEqual(out.count("<|turn>model"), 1)
        self.assertIn("part one", out)
        self.assertIn("part two", out)

    def test_tool_then_assistant_continues_same_model_turn(self):
        """P8: tool_calls + tool_responses may continue into next assistant."""
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "assistant", "content": "done"},
            ],
        )
        self.assertEqual(out.count("<|turn>model"), 1)
        self.assertIn("call:run{", out)
        self.assertIn("response:run{", out)
        self.assertIn("done", out)
        self.assertIn("<turn|>", out)

    def test_tool_response_closes_turn_before_next_user(self):
        """P8: close model turn when a user follows tool_response."""
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "user", "content": "thanks"},
            ],
        )
        tool_resp_idx = out.index("<tool_response|>")
        user_idx = out.index("<|turn>user\nthanks")
        self.assertLess(tool_resp_idx, user_idx)
        self.assertIn("<turn|>", out[tool_resp_idx:user_idx])

    def test_gen_prompt_after_tool_response_opens_thought_channel(self):
        """P8 / official 2607: thinking ON → open thought, no new model turn."""
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        )
        self.assertEqual(out.count("<|turn>model"), 1)
        self.assertTrue(out.endswith("<|channel>thought\n"))
        self.assertNotIn("<|channel>thought\n<channel|>", out.split("<tool_response|>")[-1])

    def test_gen_prompt_after_tool_response_skips_when_thinking_off(self):
        out = render(
            self.template,
            add_generation_prompt=True,
            enable_thinking=False,
            agent_defaults=False,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        )
        self.assertNotIn("<|channel>thought", out.split("<tool_response|>")[-1])

    def test_gen_prompt_after_consecutive_tool_errors_closes_thought(self):
        """P6b still wins over P8 open-thought after ≥2 consecutive errors."""
        out = render(
            self.template,
            add_generation_prompt=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "run", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"error": true}'},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c2", "type": "function", "function": {"name": "run", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "content": '{"error": true}'},
            ],
        )
        self.assertTrue(out.endswith("<|channel>thought\n<channel|>"))

    # --- multimodal ---

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
        self.assertIn("<|image|>", out)

    def test_user_image_url_alias(self):
        out = render(
            self.template,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://x"}},
                    ],
                },
            ],
        )
        self.assertIn("<|image|>", out)

    def test_user_input_audio_alias(self):
        out = render(
            self.template,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "listen"},
                        {"type": "input_audio", "input_audio": {"data": "x"}},
                    ],
                },
            ],
        )
        self.assertIn("<|audio|>", out)


    # --- G-A / G-C / G-D regressions ---

    def test_multimodal_string_item_is_text(self):
        """A plain string content part must render as text."""
        out = render(
            self.template,
            messages=[{"role": "user", "content": ["hello world"]}],
        )
        self.assertIn("hello world", out)

    def test_multimodal_string_item_is_not_media(self):
        out = render(
            self.template,
            messages=[{"role": "user", "content": ["an image of a cat"]}],
        )
        self.assertIn("an image of a cat", out)
        self.assertNotIn("<|image|>", out)

    def test_multimodal_image_part_still_rendered(self):
        out = render(
            self.template,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "image_url": {"url": "u"}}],
                }
            ],
        )
        self.assertIn("<|image|>", out)

    def test_tool_error_prose_does_not_trigger_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "check"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": {}}}
                    ],
                },
                {
                    "role": "tool",
                    "content": "OK: the previous release failed to build on CI, "
                    "but current HEAD is green",
                },
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_tool_error_json_note_does_not_trigger_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "check"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": {}}}
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"result":"ok","note":"command not found is legacy"}',
                },
            ],
        )
        self.assertNotIn("SYSTEM WARNING", out)

    def test_real_tool_error_still_triggers_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": {}}}
                    ],
                },
                {"role": "tool", "content": '{"error": true}'},
            ],
        )
        self.assertIn("SYSTEM WARNING", out)

    def test_posix_shell_failure_triggers_warning(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": {}}}
                    ],
                },
                {"role": "tool", "content": "cat: README.md: No such file or directory"},
            ],
        )
        self.assertIn("SYSTEM WARNING", out)

    def test_historical_think_token_is_passed_through_verbatim(self):
        """G-D: only the last user turn may steer thinking."""
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "<|think_off|>\nfirst"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "second"},
            ],
            add_generation_prompt=True,
        )
        self.assertIn("<|think_off|>", out)

    def test_last_user_turn_think_token_still_steers_generation(self):
        out = render(
            self.template,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "<|think_off|>\nsecond"},
            ],
            add_generation_prompt=True,
        )
        self.assertNotIn("<|think_off|>", out)
        self.assertIn("<|channel>thought", out)

def main() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(Gemma4FixedChatTemplateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\nTotal tests: {result.testsRun}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

"""Permanent fuzz tests for the built-in JSON argument parsers.

The three templates expand a JSON string into their native argument syntax.
These tests compare that expansion against the authoritative oracle:

  gemma4  the official google/gemma-4-31B-it chat template, which renders
          the same value through format_argument
  qwen    Python's json, matching what the templates' tojson produces

The cases are seeded, so a failure is reproducible.
"""

import json
import random
import string
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

from test_qwen36_fixed_template import compile_template, render  # noqa: E402

G4_TEMPLATE = ROOT / "gemma4_fixed_template.jinja"
Q38_TEMPLATE = ROOT / "qwen38_fixed_template.jinja"
Q36_TEMPLATE = ROOT / "qwen36_fixed_template.jinja"

POOL = string.printable + "你好世界éüñ🙂\U0001f600"
STRESS = ',{}[]"\\: \n\t'


def _rnd_str(rnd):
    kind = rnd.randint(0, 3)
    if kind == 0:
        return "".join(rnd.choice(POOL) for _ in range(rnd.randint(0, 12)))
    if kind == 1:
        return "".join(rnd.choice(STRESS) for _ in range(rnd.randint(0, 10)))
    if kind == 2:
        return ""
    return " ".join(
        "".join(rnd.choice(POOL) for _ in range(rnd.randint(0, 6)))
        for _ in range(rnd.randint(1, 4))
    )


def _rnd_value(rnd, depth=0):
    kind = rnd.randint(0, 6 if depth < 2 else 4)
    if kind == 0:
        return _rnd_str(rnd)
    if kind == 1:
        return rnd.randint(-1000, 1000)
    if kind == 2:
        return round(rnd.uniform(-100, 100), 3)
    if kind == 3:
        return rnd.choice([True, False])
    if kind == 4:
        return None
    if kind == 5:
        return [_rnd_value(rnd, depth + 1) for _ in range(rnd.randint(0, 3))]
    return {f"n{i}": _rnd_value(rnd, depth + 1) for i in range(rnd.randint(0, 3))}


def _cases(seed, count):
    rnd = random.Random(seed)
    out = []
    for _ in range(count):
        obj = {f"k{j}": _rnd_value(rnd) for j in range(rnd.randint(0, 4))}
        out.append((obj, json.dumps(obj)))
    return out


def _g4_fragment(text):
    start = text.index("call:f{") + len("call:f{")
    end = text.index("}<tool_call|>")
    return text[start:end]


def _qwen_params(text):
    import re

    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            r"<parameter=([^>]+)>\n(.*?)\n</parameter>", text, re.S
        )
    }


def _official_g4_value(v):
    """Mirror of the official Gemma 4 format_argument."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '<|"|>' + v + '<|"|>'
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        return (
            "{"
            + ",".join(k + ":" + _official_g4_value(x) for k, x in sorted(v.items()))
            + "}"
        )
    if isinstance(v, list):
        return "[" + ",".join(_official_g4_value(x) for x in v) + "]"
    return str(v)


def _official_g4_object(obj):
    """The members only: call:f{ ... } already supplies the braces."""
    return ",".join(k + ":" + _official_g4_value(v) for k, v in sorted(obj.items()))


class Gemma4JsonParserFuzzTests(unittest.TestCase):
    """The JSON-string path must match the official mapping path."""

    @classmethod
    def setUpClass(cls):
        cls.template = compile_template(path=G4_TEMPLATE)

    def _render(self, arguments):
        out = render(
            self.template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": arguments}}
                    ],
                },
            ],
        )
        return _g4_fragment(out)

    def test_matches_official_mapping_path(self):
        for obj, raw in _cases(20261019, 150):
            with self.subTest(raw=raw[:80]):
                self.assertEqual(self._render(raw), _official_g4_object(obj))

    def test_nested_structures_are_expanded(self):
        """The official format_argument recurses into mappings and lists."""
        out = self._render(json.dumps({"n": {"m": "deep"}}))
        self.assertEqual(out, 'n:{m:<|"|>deep<|"|>}')
        out = self._render(json.dumps({"n": [1, "x"]}))
        self.assertEqual(out, 'n:[1,<|"|>x<|"|>]')

    def test_unicode_escapes_decoded(self):
        out = self._render(json.dumps({"p": "你好"}))
        self.assertEqual(out, 'p:<|"|>你好<|"|>')


class QwenJsonParserFuzzTests(unittest.TestCase):
    """The JSON-string path must match what tojson produces for a mapping."""

    @classmethod
    def setUpClass(cls):
        cls.templates = {
            "qwen38": compile_template(path=Q38_TEMPLATE),
            "qwen36": compile_template(path=Q36_TEMPLATE),
        }

    def _render(self, template, arguments):
        out = render(
            template,
            parse_string_arguments=True,
            messages=[
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": arguments}}
                    ],
                },
            ],
        )
        return _qwen_params(out)

    def _expected(self, obj):
        """Mirror the mapping path.

        A top-level string is emitted verbatim; every other value goes
        through tojson(ensure_ascii=False), which keeps C0 control
        characters escaped and emits non-ASCII literally.
        """
        exp = {}
        for k, v in obj.items():
            if isinstance(v, str):
                exp[k] = v
            elif isinstance(v, bool):
                exp[k] = "true" if v else "false"
            elif v is None:
                exp[k] = "null"
            else:
                exp[k] = json.dumps(v, ensure_ascii=False)
        return exp

    def test_matches_tojson_semantics(self):
        for name, template in self.templates.items():
            for obj, raw in _cases(20261019, 150):
                with self.subTest(template=name, raw=raw[:80]):
                    self.assertEqual(self._render(template, raw), self._expected(obj))

    def test_unicode_escapes_decoded(self):
        for name, template in self.templates.items():
            with self.subTest(template=name):
                params = self._render(template, json.dumps({"p": "你好"}))
                self.assertEqual(params["p"], "你好")

    def test_control_characters_stay_escaped_in_nested_values(self):
        """tojson keeps C0 control characters escaped inside nested values.

        A top-level string is emitted verbatim; a nested value goes
        through tojson, which escapes control characters.
        """
        for name, template in self.templates.items():
            with self.subTest(template=name):
                params = self._render(
                    template, json.dumps({"p": {"n": "a\u000bb"}})
                )
                self.assertEqual(params["p"], '{"n": "a\\u000bb"}')


if __name__ == "__main__":
    unittest.main()

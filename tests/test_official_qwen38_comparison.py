"""Permanent comparison against the OFFICIAL Qwen3.8 chat template.

The official template only accepts a mapping for tool_call.arguments
(it calls |items on it), so a JSON string cannot be fed to it directly.

The comparison therefore renders the same value twice:

  official  arguments as a real dict   (the only shape it accepts)
  fork      arguments as a JSON string  (the path being fixed)

The tool-call fragment must be identical.

The official template is fetched from the Qwen Hugging Face repos, which
publish two variants:

  Qwen/Qwen3.8-27B            (8952 bytes)
  Qwen/Qwen3.8-2.4T-A95B-FP8  (7495 bytes)

They differ in the reasoning_effort handling, not in the tool-call
format, so both are checked.

The fetch is cached under tests/_official_qwen38/ (git-ignored) and the
comparison is skipped when the network is unavailable.

For reference, the official template renders each argument value with:

    args_value | string if args_value is string else args_value | tojson

so a string keeps its raw text while every other value is JSON.
"""

import json
import random
import string
import unittest
import urllib.request
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

OFFICIAL_URLS = {
    "27B": "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja",
    "2.4T": "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8/raw/main/chat_template.jinja",
}

CACHE = TESTS / "_official_qwen38"


def _env():
    def re_(m):
        raise TemplateError(m)

    def tj(x, ensure_ascii=False, **k):
        return json.dumps(x, ensure_ascii=ensure_ascii, **k)

    e = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    e.filters["tojson"] = tj
    e.globals["raise_exception"] = re_
    return e


def _official_templates():
    """Fetch (and cache) the official templates; None when offline."""
    CACHE.mkdir(exist_ok=True)
    out = {}
    for key, url in OFFICIAL_URLS.items():
        path = CACHE / f"qwen38-{key}.jinja"
        try:
            if not path.exists():
                path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
            out[key] = _env().from_string(path.read_text(encoding="utf-8"))
        except Exception:
            out[key] = None
    return out


def _frag(template, args, **kw):
    out = template.render(
        messages=[
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": args}}],
            },
        ],
        **kw,
    )
    i = out.index("<tool_call>")
    j = out.index("</tool_call>")
    return out[i : j + len("</tool_call>")]


def _cases(seed, count):
    rnd = random.Random(seed)
    pool = string.printable + "你好世界éüñ🙂\U0001f600"
    stress = ',{}[]"\\: \n\t'

    def rstr():
        k = rnd.randint(0, 3)
        if k == 0:
            return "".join(rnd.choice(pool) for _ in range(rnd.randint(0, 12)))
        if k == 1:
            return "".join(rnd.choice(stress) for _ in range(rnd.randint(0, 10)))
        if k == 2:
            return ""
        return " ".join(
            "".join(rnd.choice(pool) for _ in range(rnd.randint(0, 6)))
            for _ in range(rnd.randint(1, 4))
        )

    def rval(d=0):
        k = rnd.randint(0, 6 if d < 2 else 4)
        if k == 0:
            return rstr()
        if k == 1:
            return rnd.randint(-1000, 1000)
        if k == 2:
            return round(rnd.uniform(-100, 100), 3)
        if k == 3:
            return rnd.choice([True, False])
        if k == 4:
            return None
        if k == 5:
            return [rval(d + 1) for _ in range(rnd.randint(0, 3))]
        return {f"n{i}": rval(d + 1) for i in range(rnd.randint(0, 3))}

    return [
        ({f"k{j}": rval() for j in range(rnd.randint(0, 4))}) for _ in range(count)
    ]


class OfficialQwen38ComparisonTests(unittest.TestCase):
    """The JSON-string path must match the official Qwen3.8 renderer."""

    @classmethod
    def setUpClass(cls):
        cls.official = _official_templates()
        cls.forks = {
            "qwen38": _env().from_string(
                (ROOT / "qwen38_fixed_template.jinja").read_text(encoding="utf-8")
            ),
            "qwen36": _env().from_string(
                (ROOT / "qwen36_fixed_template.jinja").read_text(encoding="utf-8")
            ),
        }
        cls.cases = _cases(20261019, 200)

    def _check(self, fork_name, official_key):
        official = self.official.get(official_key)
        if official is None:
            self.skipTest(f"official Qwen3.8 {official_key} template unavailable")
        fork = self.forks[fork_name]
        for obj in self.cases:
            raw = json.dumps(obj)
            with self.subTest(fork=fork_name, raw=raw[:60]):
                self.assertEqual(
                    _frag(fork, raw, parse_string_arguments=True),
                    _frag(official, obj),
                )

    def test_qwen38_matches_official_27b(self):
        self._check("qwen38", "27B")

    def test_qwen38_matches_official_24t(self):
        self._check("qwen38", "2.4T")

    def test_qwen36_matches_official_27b(self):
        self._check("qwen36", "27B")


if __name__ == "__main__":
    unittest.main()

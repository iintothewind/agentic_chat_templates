# Agentic Chat Templates

Fixed Jinja chat templates for open-source agentic coding harnesses
([opencode](https://github.com/anomalyco/opencode), [pi](https://github.com/earendil-works/pi),
OpenHarness, LM Studio, vLLM, SGLang, and similar Claude-Code-style stacks).

Upstream model templates are accurate for single-turn demos, but often break
under multi-turn tool loops: string-encoded arguments, dropped reasoning,
asymmetric turn tags, and missing tool-error recovery. These forks keep native
protocol fidelity, then add harness-oriented defaults and fault tolerance.

## Templates

| File | Models | Protocol |
|------|--------|----------|
| [`gemma4_fixed_template.jinja`](gemma4_fixed_template.jinja) | Gemma 4 (`12B` / `26B-A4B` / `31B`, etc.) | `<\|turn\|>`, `<\|think\|\>`, `<\|channel>thought`, `<\|tool_call>call:name{...}` |
| [`qwen36_fixed_template.jinja`](qwen36_fixed_template.jinja) | Qwen3.5 / Qwen3.6 (Instruct, Coder, VL, thinking) | `<\|im_start\|\>`, `<think>`, `<tool_call>` / `<parameter>` |
| [`qwen38_fixed_template.jinja`](qwen38_fixed_template.jinja) | Qwen3.8 | same as 3.6, plus native `reasoning_effort` |

Not for Gemma 2/3 or Qwen2.5 — those use different chat protocols.
Do not mix Qwen 3.6 and 3.8 templates; 3.8 is the only family with
prompt-steered effort levels.

The Gemma 4 template is based on Google’s official **Gemma 4 260709**
canonical chat template (published 2026-07-09; tool-calling loops, turn
closures, and thinking content-ordering), then further optimized for
agentic multi-turn tool use.

## Design principles

1. **Accuracy first** — rendered prompts match what each family was trained on.
2. **Fault tolerance second** — accept messy harness inputs (JSON-string tool
   arguments, OpenAI tool envelopes, `developer` role, multimodal parts) and
   still emit valid prompts.
3. **Agent smoothness** — defaults favor multi-turn tool loops: preserve
   historical thinking, guide on tool errors, optional prompt-level think
   control. kwargs can restore upstream parity.
4. **Layered errors** — raise only when the prompt cannot be made valid.

## Quick start

### LM Studio

My Models → model settings → Prompt Template → paste the `.jinja` path.

### vLLM / SGLang

```bash
# Gemma 4
--chat-template /path/to/gemma4_fixed_template.jinja \
--tool-call-parser gemma4 --reasoning-parser gemma4

# Qwen3.5 / 3.6
--chat-template /path/to/qwen36_fixed_template.jinja \
--tool-call-parser qwen3_coder --reasoning-parser qwen3

# Qwen3.8
--chat-template /path/to/qwen38_fixed_template.jinja \
--tool-call-parser qwen3_coder --reasoning-parser qwen3
```

### Prompt-level thinking control

Insert either token in the first system/developer message or any user message.
Tokens are stripped before render and never reach the model tokenizer.

```json
{"role": "user", "content": "<|think_off|>\nAnswer briefly."}
{"role": "user", "content": "<|think_on|>\nExplain step by step."}
```

Same-message precedence:

- system/developer: `<|think_on|>` wins if both appear
- user: `<|think_off|>` wins if both appear

### Qwen 3.8 reasoning effort

Qwen 3.8 can steer thinking depth via `chat_template_kwargs.reasoning_effort`
(`low` default here; official is `xhigh`). `medium` injects no extra system
text; `xhigh` / `low` inject the official instruction strings.

```json
{"extra_body": {"chat_template_kwargs": {"reasoning_effort": "medium"}}}
```

Harness aliases: `high` / `max` → `xhigh`, `minimal` → `low`, `none` / `off`
→ disable thinking. Prompt-level tokens (3.8 only):

```json
{"role": "user", "content": "<|think_low|>\nKeep it short."}
{"role": "user", "content": "<|think_medium|>\nNormal."}
{"role": "user", "content": "<|think_xhigh|>\nGo deep."}
```

### Upstream parity kwargs

Restore closer-to-upstream behavior when needed:

**Gemma 4**

```json
{
  "agent_defaults": false,
  "enable_thinking": false,
  "preserve_thinking": false,
  "strict_tool_arguments": true,
  "inject_tool_error_warnings": false
}
```

**Qwen 3.5 / 3.6**

```json
{
  "enable_thinking": true,
  "preserve_thinking": false,
  "verbose_tool_instructions": false,
  "unwrap_tool_envelope": false,
  "strict_tool_arguments": true,
  "inject_tool_error_warnings": false
}
```

**Qwen 3.8**

```json
{
  "enable_thinking": true,
  "reasoning_effort": "xhigh",
  "preserve_thinking": true,
  "verbose_tool_instructions": false,
  "unwrap_tool_envelope": false,
  "strict_tool_arguments": true,
  "inject_tool_error_warnings": false
}
```

Pass via your server's `chat_template_kwargs` / `extra_body` path
(e.g. OpenAI-compatible `extra_body.chat_template_kwargs`).

## Tests

```bash
python -m unittest discover -s tests -v
```

Requires Python 3 + Jinja2 (`pip install jinja2`).

## License

Apache-2.0, same family as the upstream Gemma / Qwen chat templates these forks
are based on. See each template header for patch inventory and pins.

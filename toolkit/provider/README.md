# Provider Component

**Layer 1 · imports nothing · `copy_tier: standalone`**

## What It Does

Finds a usable AI provider in the environment — key, endpoint, model, API style
— or raises an error that says exactly what to set.

## Why It Is Useful

The toolkit could call a model but had no idea how to find one. `ports.LLM` is
a protocol and `LiteLLMClient` assumes litellm is already configured, so the
first question any real consumer asks — *"where does the key come from?"* — had
no answer in the library. Everyone wrote their own `os.environ.get` ladder, and
each one failed differently.

Three properties are the reason this is a unit rather than three lines inline:

**One key is enough.** Set `ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY` or
`GROQ_API_KEY` and nothing else is needed — the shortcut table supplies the
endpoint, a default model and a human-readable label. A key without a model is
still unusable, and making the caller go and look one up is how *"I set the key
and it still doesn't work"* happens.

**Two API styles, not N providers.** Almost every vendor exposes an
OpenAI-compatible chat-completions endpoint, so DeepSeek, Gemini, Groq,
Mistral, Together, xAI, Qwen, Kimi, GLM, vLLM, Ollama and LM Studio cost one
adapter and a base URL. Anthropic's Messages API is different enough to earn
its own. Two concrete shapes beat one leaky universal abstraction.

**The failure path is the feature.** `SetupError` is never raised for anything
a retry could fix, and it carries the diagnosis: which variables were checked,
which were set, and the shortest route to a working setup.

## Architecture

```
explicit arguments
   ↓ (win over everything)
TOOLKIT_API_KEY / _API_STYLE / _BASE_URL / _MODEL
   ↓ (win over shortcuts)
SHORTCUTS, in order — ANTHROPIC, OPENAI, GEMINI, GOOGLE, DEEPSEEK, GROQ,
   ↓              MISTRAL, TOGETHER, XAI, DASHSCOPE, MOONSHOT, ZHIPU, OLLAMA
Provider(style, model, label, base_url, api_key, source)
   ↓ or
SetupError(diagnose().render())
```

`.toolkit.env` is read first and then **overlaid by the real environment**, in
that direction: the file is what you keep on your own machine, the environment
is what CI and a container inject, and the injected value has to win or every
deployment needs the file deleted first.

## Installation

Standalone — copy one directory:

```bash
cp -r toolkit/provider your_project/
```

Python 3.10+. Standard library only. It defines its own `SetupError` rather than
importing `core.errors`, which is what keeps it standalone: credential
resolution is the most portable thing here and coupling it to a
document-oriented error taxonomy would be a poor trade.

## Input / Output

```python
from toolkit.provider import resolve, available, diagnose, SetupError

provider = resolve()                      # from the environment
provider = resolve(model="claude-opus-5")  # override one field
provider = resolve(env_file=None, environ={"GROQ_API_KEY": "gsk-..."})  # explicit
```

| Function | Returns | Raises |
|---|---|---|
| `resolve(**overrides)` | `Provider` | `SetupError` naming what to set |
| `available()` | `bool` | — |
| `diagnose()` | `Diagnosis` with `.render()` | — |
| `load_settings()` | `dict[str, str]`, file overlaid by env | — |

`Provider` is frozen, with `.style`, `.model`, `.label`, `.base_url`,
`.api_key`, `.source`, plus `.endpoint` (falls back to the style's official
host), `.redacted_key` and `.describe()`.

## The key must not leak

`Provider` overrides `__repr__` and `__str__` to redact the key, showing only
its length and last four characters.

This is not fussiness. A dataclass's generated `repr` prints every field, and
the places a provider object ends up — a debug log line, an unhandled
traceback, a crash reporter, a CI job's captured output — are all places a
credential must never reach. **One `logging.debug(provider)` is enough to leak
a key into a log aggregator for its entire retention period**, and that is not
retrofittable: the key has to be rotated. `Diagnosis` reports variable *names*
only for the same reason. The real value is still there as `.api_key`, which
is greppable at review time.

## Usage

```python
from toolkit.llm_http import HttpLLM
from toolkit.provider import available, resolve

if available():
    kb = KnowledgeBase(..., llm=HttpLLM(resolve()))
else:
    kb = KnowledgeBase(...)        # answers extractively instead
```

Reporting a misconfiguration to a user:

```python
try:
    provider = resolve()
except SetupError as exc:
    print(exc)       # lists the variables checked and the shortest fix
    raise SystemExit(2)
print(provider.describe())
# DeepSeek deepseek-chat via https://api.deepseek.com (key <set:19 chars ...7890> from DEEPSEEK_API_KEY)
```

`python examples/cookbook.py provider` runs a worked example.

## Limitations

- **Two styles only.** A provider speaking neither shape needs a third
  `ApiStyle` member and a branch in `llm_http`.
- **Default models go stale.** The shortcut table names a current model per
  provider; vendors rename and retire them. `TOOLKIT_MODEL` or the `model=`
  argument overrides, and a wrong name produces a 404 that names it.
- **No credential store.** Environment variables and a dotfile, not Vault, AWS
  Secrets Manager or a keychain. Resolve the secret yourself and pass
  `api_key=`.
- **No key validation.** `resolve()` checks that settings are *present and
  coherent*, never that the key works — that needs a network call, which
  belongs to the client.
- **No per-model cost table**, so `Usage.cost_usd` is not populated from here.
- `.toolkit.env` is `KEY=VALUE` only: no shell expansion, no `export`, no
  multi-line values. A dotfile parser that tries to be a shell is a parser with
  a surprise in it.

## Integration Guide

1. **Call `resolve()` once at your entry point** and pass the `Provider` down.
   Resolving repeatedly re-reads the file and makes the precedence harder to
   reason about.
2. **Use `available()` when you can degrade.** The toolkit's own pipelines
   answer extractively with no LLM, so "is there a model?" has a useful answer.
3. **Print `SetupError` verbatim.** It is written to be read by whoever has to
   fix it; summarising it throws away the part that helps.
4. **Never log the provider's `api_key`.** Log `describe()` or
   `redacted_key`; both are safe by construction.
5. For a gateway or proxy, set `<PROVIDER>_BASE_URL` alongside the shortcut
   key to keep the other defaults.

## Extraction Notes

- **Preserved** from PRism (`prism.py`, same author): the shortcut table of
  self-sufficient keys, the two-style abstraction, dotfile-overlaid-by-env
  precedence, and `SetupError` messages that name the setting and where to find
  its value. Those had already earned their place in a working project.
- **Removed:** the project-specific `PRISM_*` prefix and its coupling to that
  repo's layout and CLI.
- **Added:** redaction in `__repr__`/`__str__` — the original is a plain
  dataclass whose generated repr prints the key, which is one debug statement
  away from a leak. Also `ApiStyle` as an enum rather than bare strings,
  `diagnose()`, `available()`, an injectable `environ` so the resolution order
  is testable without touching the real environment, and seven more providers.
- **Isolated:** imports nothing, including from `core`.

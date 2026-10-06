"""Find a usable provider from the environment, or explain why there isn't one."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .models import ApiStyle, Provider, SetupError

#: Keys that are sufficient on their own: set one and nothing else is needed.
#:
#: Ordered, and the order is the precedence when several are present. A shortcut
#: carries a default model because a key with no model is still unusable, and
#: making the caller look one up is how "I set the key and it still doesn't
#: work" happens.
SHORTCUTS: tuple[tuple[str, ApiStyle, str | None, str, str], ...] = (
    # env var,              style,              base_url,                                               model,                label
    ("ANTHROPIC_API_KEY", ApiStyle.ANTHROPIC, None, "claude-opus-5", "Anthropic"),
    ("OPENAI_API_KEY", ApiStyle.OPENAI, "https://api.openai.com/v1", "gpt-4.1", "OpenAI"),
    ("GEMINI_API_KEY", ApiStyle.OPENAI,
     "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro", "Gemini"),
    ("GOOGLE_API_KEY", ApiStyle.OPENAI,
     "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro", "Gemini"),
    ("DEEPSEEK_API_KEY", ApiStyle.OPENAI, "https://api.deepseek.com", "deepseek-flash", "DeepSeek"),
    ("GROQ_API_KEY", ApiStyle.OPENAI, "https://api.groq.com/openai/v1",
     "llama-3.3-70b-versatile", "Groq"),
    ("MISTRAL_API_KEY", ApiStyle.OPENAI, "https://api.mistral.ai/v1",
     "mistral-large-latest", "Mistral"),
    ("TOGETHER_API_KEY", ApiStyle.OPENAI, "https://api.together.xyz/v1",
     "meta-llama/Llama-3.3-70B-Instruct-Turbo", "Together"),
    ("XAI_API_KEY", ApiStyle.OPENAI, "https://api.x.ai/v1", "grok-4", "xAI"),
    ("DASHSCOPE_API_KEY", ApiStyle.OPENAI,
     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max", "Qwen"),
    ("MOONSHOT_API_KEY", ApiStyle.OPENAI, "https://api.moonshot.cn/v1",
     "moonshot-v1-128k", "Kimi"),
    ("ZHIPU_API_KEY", ApiStyle.OPENAI, "https://open.bigmodel.cn/api/paas/v4",
     "glm-4-plus", "GLM"),
    ("OLLAMA_HOST", ApiStyle.OPENAI, None, "llama3.1", "Ollama"),
)

#: Explicit settings, which win over every shortcut.
KEY_VAR = "TOOLKIT_API_KEY"
STYLE_VAR = "TOOLKIT_API_STYLE"
BASE_VAR = "TOOLKIT_BASE_URL"
MODEL_VAR = "TOOLKIT_MODEL"

#: Where a dotfile is looked for, relative to the working directory.
DEFAULT_ENV_FILE = ".toolkit.env"

_ANTHROPIC_OFFICIAL = (None, "", "https://api.anthropic.com")


def load_settings(
    env_file: str | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A dotfile overlaid by the real environment.

    That precedence and not the other way around: the file is what you commit
    to your own machine, the environment is what CI and a container inject, and
    the injected value has to win or every deployment needs the file deleted
    first.

    Blank values are dropped rather than overriding a set one, because
    `KEY=` in a shell profile is a common way to accidentally unset something.

    Only `KEY=VALUE` lines are read. There is no shell expansion, no `export`,
    no quoting beyond a single surrounding pair - a dotfile parser that tries to
    be a shell is a parser with a surprise in it.
    """
    settings: dict[str, str] = {}
    if env_file and os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    settings[name.strip()] = value
    source = os.environ if environ is None else environ
    settings.update({k: v for k, v in source.items() if v})
    return settings


@dataclass(frozen=True)
class Diagnosis:
    """Why no provider could be resolved, in terms a human can act on.

    Carries setting *names* only. Reporting which variables were seen is the
    whole point of this type, and a diagnosis that leaked their values would be
    worse than no diagnosis at all.
    """

    checked: tuple[str, ...]
    present: tuple[str, ...]
    hint: str

    def render(self) -> str:
        lines = ["No AI provider is configured.", ""]
        if self.present:
            lines.append("Set, but not sufficient: " + ", ".join(self.present))
            lines.append("")
        lines.append(self.hint)
        lines.append("")
        lines.append("Set any one of these and nothing else is needed:")
        for name, _, _, model, label in SHORTCUTS[:6]:
            lines.append("    %-22s -> %s %s" % (name, label, model))
        lines.append("")
        lines.append("Or point at any OpenAI-compatible endpoint:")
        lines.append("    %s=sk-..." % KEY_VAR)
        lines.append("    %s=https://your-host/v1" % BASE_VAR)
        lines.append("    %s=your-model-name" % MODEL_VAR)
        return "\n".join(lines)


def diagnose(settings: Mapping[str, str] | None = None) -> Diagnosis:
    """Explain the absence of a provider without revealing any value."""
    s = dict(settings) if settings is not None else load_settings()
    checked = tuple([KEY_VAR, STYLE_VAR, BASE_VAR, MODEL_VAR] + [n for n, *_ in SHORTCUTS])
    present = tuple(name for name in checked if s.get(name))
    if s.get(KEY_VAR) and not s.get(BASE_VAR):
        hint = (
            "%s is set but %s is not. An explicit key needs an explicit endpoint; "
            "use a shortcut variable instead if you want the defaults." % (KEY_VAR, BASE_VAR)
        )
    elif s.get(BASE_VAR) and not s.get(KEY_VAR):
        hint = "%s is set but %s is not." % (BASE_VAR, KEY_VAR)
    else:
        hint = "None of the settings that would identify a provider are set."
    return Diagnosis(checked=checked, present=present, hint=hint)


def resolve(
    *,
    style: ApiStyle | str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    env_file: str | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> Provider:
    """Resolve a provider, or raise `SetupError` saying what to set.

    Precedence, highest first: explicit arguments, the four `TOOLKIT_*`
    settings, then the first matching shortcut in `SHORTCUTS` order.

    Raises `SetupError` rather than returning `None`, because every caller of a
    `None` return writes the same unhelpful "no provider configured" message.
    The error carries `diagnose().render()`, which lists the variables that were
    checked, the ones that were set, and the shortest route to a working setup.
    """
    s = load_settings(env_file=env_file, environ=environ)

    chosen_style = style if style is not None else s.get(STYLE_VAR)
    chosen_base = base_url if base_url is not None else s.get(BASE_VAR)
    chosen_key = api_key if api_key is not None else s.get(KEY_VAR)
    chosen_model = model if model is not None else s.get(MODEL_VAR)
    source = "explicit argument" if api_key is not None else KEY_VAR

    if chosen_key or chosen_base:
        if chosen_style is None:
            chosen_style = ApiStyle.OPENAI
        try:
            resolved_style = ApiStyle(
                chosen_style.value if isinstance(chosen_style, ApiStyle) else str(chosen_style).strip().lower()
            )
        except ValueError:
            raise SetupError(
                "%s must be 'openai' or 'anthropic', not %r"
                % (STYLE_VAR, chosen_style)
            ) from None

        if resolved_style is ApiStyle.OPENAI and not chosen_base:
            raise SetupError(
                "%s is missing. An OpenAI-compatible provider needs its endpoint - "
                "copy the base URL from the provider's API docs, or use a shortcut "
                "variable such as DEEPSEEK_API_KEY which supplies one." % BASE_VAR
            )
        if not chosen_key:
            raise SetupError(
                "%s is set but %s is not; an endpoint without a key cannot be called."
                % (BASE_VAR, KEY_VAR)
            )
        if not chosen_model:
            if resolved_style is ApiStyle.ANTHROPIC and chosen_base in _ANTHROPIC_OFFICIAL:
                chosen_model = "claude-opus-5"
            else:
                raise SetupError(
                    "%s is missing. Use a model name from the provider's docs; there is "
                    "no safe default for a custom endpoint." % MODEL_VAR
                )
        label = (
            chosen_base.split("/")[2]
            if chosen_base and "//" in chosen_base
            else ("Anthropic" if resolved_style is ApiStyle.ANTHROPIC else "OpenAI")
        )
        return Provider(
            style=resolved_style,
            model=chosen_model,
            label=label,
            base_url=chosen_base,
            api_key=chosen_key,
            source=source,
        )

    for name, shortcut_style, shortcut_base, shortcut_model, label in SHORTCUTS:
        value = s.get(name)
        if not value:
            continue
        # A provider-specific base URL overrides the shortcut's default, which is
        # how a proxy, a gateway or a regional endpoint is used without giving up
        # the shortcut's other defaults.
        override = s.get(name.replace("_API_KEY", "_BASE_URL"))
        if name == "OLLAMA_HOST":
            # Ollama's variable is a host, not a key: it identifies the endpoint
            # and needs no credential.
            host = value.rstrip("/")
            if not host.startswith("http"):
                host = "http://" + host
            return Provider(
                style=shortcut_style,
                model=chosen_model or shortcut_model,
                label=label,
                base_url=host + "/v1",
                api_key="ollama",
                source=name,
            )
        return Provider(
            style=shortcut_style,
            model=chosen_model or shortcut_model,
            label=label,
            base_url=override or shortcut_base,
            api_key=value,
            source=name,
        )

    raise SetupError(diagnose(s).render())


def available(
    env_file: str | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Is a provider configured? For a caller that wants to degrade, not fail.

    The toolkit's own pipelines answer extractively when no LLM is configured,
    so "is there a model?" is a question with a useful answer rather than an
    error.
    """
    try:
        resolve(env_file=env_file, environ=environ)
    except SetupError:
        return False
    return True

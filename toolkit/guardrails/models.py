"""Data contracts for injection detection and output policy."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How confident the detector is that this is an attack, not a mention.

    A security policy document legitimately contains the sentence "ignore all
    previous instructions" while explaining the attack. Severity is what lets a
    caller treat that differently from the same string buried in an appendix of
    a supplier's invoice.
    """

    CRITICAL = "critical"
    """Explicit attempt to override the model's instructions."""
    HIGH = "high"
    """Role manipulation, or an instruction to conceal something."""
    MEDIUM = "medium"
    """Suspicious imperative phrasing addressed at a model."""

    @property
    def rank(self) -> int:
        return {"medium": 1, "high": 2, "critical": 3}[self.value]


class InjectionAction(str, Enum):
    """What to do with a passage that trips the detector."""

    FLAG = "flag"
    """Record it, change nothing. Useful for measuring a corpus before enforcing."""
    NEUTRALISE = "neutralise"
    """Replace the offending span with an audit marker, keep the rest of the
    passage. The default: it removes the instruction without discarding content
    that may hold the actual answer."""
    EXCLUDE = "exclude"
    """Drop the whole passage from the prompt."""
    REFUSE = "refuse"
    """Abandon the answer entirely."""


@dataclass(frozen=True)
class InjectionFinding:
    pattern: str
    severity: Severity
    start: int
    end: int
    excerpt: str
    """The matched text, truncated. Kept so a reviewer can judge a false
    positive without re-reading the source document."""
    source_id: str = ""

    def render(self) -> str:
        return (
            self.severity.value
            + ":"
            + self.pattern
            + " @"
            + str(self.start)
            + " "
            + repr(self.excerpt[:60])
        )


@dataclass(frozen=True)
class PolicyFinding:
    rule: str
    detail: str
    severity: Severity = Severity.HIGH


@dataclass
class GuardrailConfig:
    action: InjectionAction = InjectionAction.NEUTRALISE
    min_severity: Severity = Severity.HIGH
    """Findings below this are recorded but never acted on. HIGH by default
    because MEDIUM patterns fire on ordinary imperative prose."""

    delimit_sources: bool = True
    """Fence each source and tell the model the contents are data. Structural,
    cheap, and more robust than pattern matching — it is the layer that still
    works against a payload no pattern anticipated."""

    marker: str = "[removed: possible injected instruction]"
    """ASCII on purpose. A marker with decorative Unicode brackets raises
    UnicodeEncodeError the moment it reaches a cp1252 console or a latin-1 log,
    which turns a security feature into a crash on Windows."""

    check_output: bool = True
    allow_external_urls: bool = False
    """When False, a URL in the answer that appears in no source is flagged.
    This is the data-exfiltration path: a classic payload makes the model emit
    an image or link to an attacker host with the conversation in the query
    string, and the request fires when the answer is rendered."""

    refuse_on_critical_output: bool = True
    """A model echoing injected instructions back is strong evidence the attack
    worked; returning that answer to a user is worse than refusing."""

    def acts_on(self, severity: Severity) -> bool:
        return severity.rank >= self.min_severity.rank


@dataclass
class ShieldedSource:
    source_id: str
    text: str
    """Post-neutralisation text, as the model will see it."""
    original_text: str
    findings: Sequence[InjectionFinding] = field(default_factory=list)
    excluded: bool = False

    @property
    def modified(self) -> bool:
        return self.text != self.original_text


@dataclass
class ShieldResult:
    sources: Sequence[ShieldedSource]
    findings: Sequence[InjectionFinding]
    refused: bool = False
    """True when a CRITICAL finding and `action=REFUSE` mean no answer should be
    attempted at all."""

    @property
    def included(self) -> list[ShieldedSource]:
        return [s for s in self.sources if not s.excluded]

    @property
    def worst(self) -> Severity | None:
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=lambda s: s.rank)


@dataclass
class PolicyResult:
    findings: Sequence[PolicyFinding]
    should_refuse: bool = False

    @property
    def ok(self) -> bool:
        return not self.findings


@dataclass
class ScanRequest:
    """Input to the detector. `sources` maps an id to the text to be shielded."""

    sources: Mapping[str, str]
    config: GuardrailConfig = field(default_factory=GuardrailConfig)

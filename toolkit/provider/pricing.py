"""What a model call costs, so `Usage.cost_usd` can stop being zero.

`Usage.cost_usd` has been in `core` since the beginning and nothing filled it
in. `llm_http` listed that as a limitation and the reason was sound: prices
change weekly, and a table shipped inside a library nobody updates weekly is
worse than no table, because a stale number is believed.

The reason was sound and the conclusion was half-finished. `GovernedLLM`
enforces `max_cost_usd` against that field, so

    GovernedLLM(HttpLLM(provider), GovernorConfig(max_cost_usd=5.0))

was enforcing a ceiling against a constant zero. The governor never fired.
Every caller that believed it had a budget had none, and nothing said so. Found
on a live run that reported a spend of $0.00 for work that had demonstrably
cost money.

So this module is the **mechanism** and not the data. The toolkit still ships
no prices: an empty `PriceBook` is the default and behaviour is unchanged for
every existing caller. What changes is that a caller can now supply the numbers
it already knows, and that `governor` refuses to pretend a ceiling is binding
when nothing is priced - see `require_priced_calls`.

Domain-neutral by construction: a rate per million tokens, with an optional
window for providers that discount off-peak. No currency type, no accounting,
no invoice.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class OffPeakWindow:
    """A daily window in which a provider charges less.

    Hours are UTC and the window may wrap midnight: 22 to 6 is one window of
    eight hours, not a contradiction. The naive `start <= hour < end` test gets
    that backwards and silently prices the whole night at the peak rate.
    """

    start_hour_utc: float
    end_hour_utc: float
    multiplier: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("start_hour_utc", self.start_hour_utc),
            ("end_hour_utc", self.end_hour_utc),
        ):
            if not 0 <= value <= 24:
                raise ValueError(name + " must be within 0..24")
        if self.multiplier < 0:
            raise ValueError("multiplier must not be negative")

    def contains(self, moment: datetime) -> bool:
        hour = moment.hour + moment.minute / 60.0
        if self.start_hour_utc <= self.end_hour_utc:
            return self.start_hour_utc <= hour < self.end_hour_utc
        # Wrapped: 22 -> 6 means "at or after 22, or before 6".
        return hour >= self.start_hour_utc or hour < self.end_hour_utc


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, input and output separately.

    Separately because they differ by a factor of four or more, and because a
    reasoning model bills its thinking as output - so a single blended rate
    misprices exactly the models whose cost is hardest to predict.
    """

    input_per_million: float
    output_per_million: float
    off_peak: OffPeakWindow | None = None
    note: str = ""
    """Where the number came from and when. A price with no provenance is a
    price nobody can check."""

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise ValueError("a rate must not be negative")

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        at: datetime | None = None,
    ) -> float:
        total = (
            input_tokens * self.input_per_million / 1_000_000.0
            + output_tokens * self.output_per_million / 1_000_000.0
        )
        if self.off_peak is None:
            return total
        moment = at or datetime.now(timezone.utc)
        return total * self.off_peak.multiplier if self.off_peak.contains(moment) else total


@dataclass(frozen=True)
class PriceBook:
    """Model name -> price. Empty by default, which is the toolkit's position.

    `default` is for a caller that would rather over-estimate an unknown model
    than price it at zero. Left unset, an unpriced model costs nothing and the
    governor's guard is what keeps that from being dangerous - inventing a
    number here would be worse, because it would be wrong and invisible.
    """

    prices: Mapping[str, ModelPrice] = field(default_factory=dict)
    default: ModelPrice | None = None

    def for_model(self, model: str) -> ModelPrice | None:
        name = (model or "").strip().lower()
        if not name:
            return self.default
        if name in self.prices:
            return self.prices[name]
        # Providers append dates and revisions without changing what a model
        # costs: `deepseek-flash-2026-10-01` bills as `deepseek-flash`. Longest
        # match wins, so a genuinely distinct `-pro` variant is not swallowed
        # by its shorter sibling.
        matches = [key for key in self.prices if name.startswith(key)]
        if matches:
            return self.prices[max(matches, key=len)]
        return self.default

    def cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        at: datetime | None = None,
    ) -> float:
        price = self.for_model(model)
        if price is None:
            return 0.0
        return price.cost(input_tokens, output_tokens, at)

    def __bool__(self) -> bool:
        return bool(self.prices) or self.default is not None


__all__ = ["ModelPrice", "OffPeakWindow", "PriceBook"]

"""Schema-driven extraction with a repair loop and per-field provenance.

Every framework has extraction. Almost none have a repair loop that converges,
and that is where most of the accuracy lives. The difference is what happens on a
validation failure:

* The common approach re-runs the whole prompt and hopes. The model has no idea
  what was wrong, so it often reproduces the same mistake.
* This one sends back **only the fields that failed, with the specific error and
  the value it produced**. That turns a guess into a correction, and it is why
  two repair rounds are usually enough.

The second thing this adds is grounding: after validation, each extracted value
is searched for in the source text. A value that appears verbatim carries the
provenance of the passage it came from — so a field can be clicked back to a page
and a bounding box. A value that does not appear anywhere is flagged. It is not
necessarily wrong (a total can be computed, a date reformatted), but every
fabricated value is in that set, so it is the set a human should see first.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.errors import ValidationFailed
from ..core.models import Chunk, Document, Message, Provenance, Usage
from .models import (
    ExtractionConfig,
    ExtractionRequest,
    ExtractionResult,
    ExtractionSchema,
    FieldResult,
    FieldSpec,
    FieldType,
    ValidationIssue,
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TRUE = frozenset({"true", "yes", "y", "1"})
_FALSE = frozenset({"false", "no", "n", "0"})
_CURRENCY = re.compile(
    r"[$£€¥₹]|" + r"\b(?:usd|eur|gbp|jpy|myr|sgd|aud|cad|chf|cny|inr)\b",
    re.IGNORECASE,
)
"""Currency symbols and ISO codes, stripped before numeric parsing. The word
boundaries matter: without them "inr" would match inside an ordinary word."""
_NUMBER_NOISE = re.compile(r"[,\s$£€¥₹]")

_SYSTEM = (
    "You extract structured data from documents. Return a single JSON object and "
    "nothing else: no prose, no explanation, no code fence. Use null for any "
    "field the document does not state. Never invent a value."
)


def extract_json_object(text: str) -> Any:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in fences, prefix it with 'Here is the result:', or append a
    remark. Brace-matching that respects string literals and escapes is the only
    approach that survives all three, and it costs twenty lines.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        block = _balanced_object(candidate)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in response")


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _normalise(text: Any) -> str:
    """Lowercase and collapse whitespace. Accepts non-strings because extracted
    values are compared against source text as their rendered form."""
    return " ".join(str(text).lower().split())


def _parse_number(raw: str) -> float | None:
    """Parse a money-shaped string into a float.

    Handles three things the naive strip-commas approach gets wrong, all found
    in the stress corpus:

    * **European notation.** "2.450,75" means 2450.75, not 2.45075. The rule
      that disambiguates is positional, not locale-based: when both separators
      appear, the **rightmost** one is the decimal point. That holds for both
      conventions and needs no locale guess.
    * **Accounting negatives.** "($310.00)" is -310. Parentheses are how
      finance writes a negative, and dropping them inverts the sign of every
      credit note.
    * **Currency words and codes.** "EUR 2.450,75" and "USD 1,240.50".

    A single separator is ambiguous in principle — "1.234" is 1234 in Germany
    and 1.234 elsewhere. Resolved by digit grouping: exactly three digits after
    a lone separator is read as a thousands group, which is the convention that
    makes "1.234" and "1,234" both mean 1234.
    """
    text = raw.strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    text = _CURRENCY.sub("", text).strip()
    if text.startswith("-"):
        negative = not negative
        text = text[1:].strip()
    text = text.replace(" ", "").replace(" ", "")
    if not text:
        return None

    last_dot = text.rfind(".")
    last_comma = text.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        # Both present: the rightmost separator is the decimal point.
        if last_comma > last_dot:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif last_comma >= 0:
        tail = text[last_comma + 1 :]
        text = (
            text.replace(",", "")
            if len(tail) == 3 and tail.isdigit()
            else text.replace(",", ".")
        )
    elif last_dot >= 0:
        tail = text[last_dot + 1 :]
        if len(tail) == 3 and tail.isdigit() and text.count(".") >= 1 and len(text) > 4:
            # "2.450" with no other separator: a thousands group.
            text = text.replace(".", "")

    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


class ExtractionComponent:
    """Document (or text) plus a schema in, validated object out."""

    def __init__(self, llm: Any) -> None:
        if llm is None:
            raise ValueError("ExtractionComponent requires an LLM")
        self._llm = llm

    def execute(self, input_data: ExtractionRequest) -> ExtractionResult:
        cfg = input_data.config
        segments = self._segments(input_data.source)
        context = self._context(segments, cfg)

        messages = [
            Message("system", _SYSTEM),
            Message("user", self._prompt(input_data, context)),
        ]

        usage = Usage()
        raw_responses: list[str] = []
        data: dict[str, Any] = {}
        issues: list[ValidationIssue] = []
        attempts = 0

        for attempt in range(cfg.max_repairs + 1):
            attempts = attempt + 1
            completion = self._llm.complete(messages, cfg.temperature, cfg.max_tokens)
            usage = usage + (completion.usage or Usage())
            raw_responses.append(completion.text)

            try:
                parsed = extract_json_object(completion.text)
            except ValueError as exc:
                issues = [ValidationIssue("<response>", str(exc), completion.text[:120])]
                data = {}
                if attempt < cfg.max_repairs:
                    messages = messages[:2] + [
                        Message("assistant", completion.text[:2000]),
                        Message(
                            "user",
                            "That was not valid JSON ("
                            + str(exc)
                            + "). Return only the JSON object.",
                        ),
                    ]
                continue

            if not isinstance(parsed, Mapping):
                issues = [
                    ValidationIssue("<response>", "expected a JSON object", parsed)
                ]
                data = {}
                continue

            data, issues = self._validate_object(
                parsed, input_data.schema.fields, cfg, prefix=""
            )

            if cfg.require_grounding:
                issues = issues + self._grounding_issues(data, input_data.schema, segments)

            if not issues:
                break
            if attempt < cfg.max_repairs:
                messages = messages[:2] + [
                    Message("assistant", json.dumps(data, ensure_ascii=False)),
                    Message("user", self._repair_prompt(issues, input_data.schema)),
                ]

        fields = self._ground(data, input_data.schema, segments)

        if issues and cfg.strict:
            raise ValidationFailed(
                "extraction failed after "
                + str(attempts)
                + " attempt(s): "
                + "; ".join(i.render() for i in issues[:5])
            )

        return ExtractionResult(
            data=data,
            fields=fields,
            issues=issues,
            attempts=attempts,
            usage=usage,
            raw_responses=raw_responses,
        )

    # --- source handling --------------------------------------------------

    def _segments(self, source: Any) -> list[tuple[str, Provenance | None]]:
        """Flatten any accepted source into (text, provenance) pairs.

        Keeping provenance attached at this level is what lets a field point back
        at a page later; flattening to one string first would discard it.
        """
        if isinstance(source, str):
            return [(source, None)]
        if isinstance(source, Document):
            return [
                (block.text, block.provenance)
                for block in source.content_blocks()
                if block.text.strip()
            ]
        if isinstance(source, Sequence):
            out: list[tuple[str, Provenance | None]] = []
            for item in source:
                if isinstance(item, Chunk):
                    prov = item.provenances[0] if item.provenances else None
                    out.append((item.text, prov))
                elif isinstance(item, str):
                    out.append((item, None))
            if out:
                return out
        raise TypeError(
            "source must be a Document, a sequence of Chunks or strings, or a string"
        )

    def _context(
        self, segments: Sequence[tuple[str, Provenance | None]], cfg: ExtractionConfig
    ) -> str:
        parts: list[str] = []
        used = 0
        for text, prov in segments:
            label = "(page " + str(prov.page) + ") " if prov else ""
            piece = label + text
            if used + len(piece) > cfg.context_char_limit and parts:
                break
            parts.append(piece)
            used += len(piece)
        return "\n".join(parts)

    # --- prompting --------------------------------------------------------

    def _describe(self, spec: FieldSpec, indent: int = 0) -> list[str]:
        pad = "  " * indent
        bits = [spec.type.value]
        if not spec.required:
            bits.append("optional")
        if spec.enum:
            bits.append("one of: " + ", ".join(spec.enum))
        if spec.pattern:
            bits.append("matching " + spec.pattern)
        if spec.minimum is not None:
            bits.append(">= " + str(spec.minimum))
        if spec.maximum is not None:
            bits.append("<= " + str(spec.maximum))
        if spec.type is FieldType.DATE:
            bits.append("format YYYY-MM-DD")
        if spec.type is FieldType.ARRAY and spec.item_type:
            bits.append("of " + spec.item_type.value)
        line = pad + "- " + spec.name + " (" + "; ".join(bits) + ")"
        if spec.description:
            line += ": " + spec.description
        if spec.examples:
            line += " e.g. " + ", ".join(spec.examples[:3])
        lines = [line]
        for nested in spec.fields:
            lines.extend(self._describe(nested, indent + 1))
        return lines

    def _prompt(self, request: ExtractionRequest, context: str) -> str:
        schema = request.schema
        lines = ["Document:", context, "", "Extract these fields:"]
        for spec in schema.fields:
            lines.extend(self._describe(spec))
        if schema.description:
            lines.insert(0, schema.description + "\n")
        if request.instructions:
            lines.extend(["", request.instructions])
        lines.extend(["", "Return one JSON object with exactly these top-level keys: "
                      + ", ".join(f.name for f in schema.fields) + "."])
        return "\n".join(lines)

    def _repair_prompt(
        self, issues: Sequence[ValidationIssue], schema: ExtractionSchema
    ) -> str:
        """Name only what failed.

        Re-sending the full schema invites the model to rewrite fields that were
        already correct, which is how a repair round makes things worse.
        """
        specs = schema.field_map()
        lines = ["Your JSON had these problems. Fix only these fields and return"
                 " the complete JSON object again:"]
        for issue in issues[:12]:
            lines.append("- " + issue.render())
            root = issue.path.split(".")[0].split("[")[0]
            spec = specs.get(root)
            if spec is not None and spec.description:
                lines.append("    " + spec.name + " means: " + spec.description)
        lines.append("Leave every other field exactly as it was.")
        return "\n".join(lines)

    # --- validation -------------------------------------------------------

    def _validate_object(
        self,
        raw: Mapping[str, Any],
        specs: Sequence[FieldSpec],
        cfg: ExtractionConfig,
        prefix: str,
    ) -> tuple[dict[str, Any], list[ValidationIssue]]:
        data: dict[str, Any] = {}
        issues: list[ValidationIssue] = []
        for spec in specs:
            path = prefix + spec.name
            present = spec.name in raw
            value = raw.get(spec.name)
            if value is None or (isinstance(value, str) and not value.strip()):
                if spec.required:
                    issues.append(
                        ValidationIssue(
                            path,
                            "required field is missing"
                            if not present
                            else "required field is null or empty",
                        )
                    )
                data[spec.name] = None
                continue
            coerced, issue = self._validate_value(value, spec, cfg, path)
            data[spec.name] = coerced
            if issue is not None:
                issues.append(issue)
        return data, issues

    def _validate_value(
        self, value: Any, spec: FieldSpec, cfg: ExtractionConfig, path: str
    ) -> tuple[Any, ValidationIssue | None]:
        kind = spec.type

        if kind is FieldType.OBJECT:
            if not isinstance(value, Mapping):
                return None, ValidationIssue(path, "expected an object", value)
            nested, nested_issues = self._validate_object(value, spec.fields, cfg, path + ".")
            return nested, nested_issues[0] if nested_issues else None

        if kind is FieldType.ARRAY:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return None, ValidationIssue(path, "expected an array", value)
            out: list[Any] = []
            for index, item in enumerate(value):
                item_path = path + "[" + str(index) + "]"
                if spec.fields:
                    if not isinstance(item, Mapping):
                        return out, ValidationIssue(item_path, "expected an object", item)
                    nested, nested_issues = self._validate_object(
                        item, spec.fields, cfg, item_path + "."
                    )
                    out.append(nested)
                    if nested_issues:
                        return out, nested_issues[0]
                else:
                    element = FieldSpec(
                        name=spec.name, type=spec.item_type or FieldType.STRING
                    )
                    coerced, issue = self._validate_value(item, element, cfg, item_path)
                    out.append(coerced)
                    if issue is not None:
                        return out, issue
            return out, None

        coerced, issue = self._scalar(value, spec, cfg, path)
        if issue is not None:
            return coerced, issue

        if spec.enum and str(coerced) not in spec.enum:
            return coerced, ValidationIssue(
                path, "must be one of: " + ", ".join(spec.enum), coerced
            )
        if spec.pattern and not re.fullmatch(spec.pattern, str(coerced)):
            return coerced, ValidationIssue(
                path, "must match the pattern " + spec.pattern, coerced
            )
        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            if spec.minimum is not None and coerced < spec.minimum:
                return coerced, ValidationIssue(
                    path, "must be at least " + str(spec.minimum), coerced
                )
            if spec.maximum is not None and coerced > spec.maximum:
                return coerced, ValidationIssue(
                    path, "must be at most " + str(spec.maximum), coerced
                )
        return coerced, None

    def _scalar(
        self, value: Any, spec: FieldSpec, cfg: ExtractionConfig, path: str
    ) -> tuple[Any, ValidationIssue | None]:
        kind = spec.type
        if kind is FieldType.STRING:
            return (value if isinstance(value, str) else str(value)), None

        if kind is FieldType.BOOLEAN:
            if isinstance(value, bool):
                return value, None
            token = _normalise(value)
            if cfg.coerce and token in _TRUE:
                return True, None
            if cfg.coerce and token in _FALSE:
                return False, None
            return value, ValidationIssue(path, "expected true or false", value)

        if kind in (FieldType.INTEGER, FieldType.NUMBER):
            if isinstance(value, bool):
                return value, ValidationIssue(path, "expected a number", value)
            if isinstance(value, (int, float)):
                number: float | int = value
            elif cfg.coerce and isinstance(value, str):
                # Models return "1,234.50", "$1234.50", "EUR 2.450,75" and
                # "($310.00)". Normalising locally is cheaper than spending a
                # repair round on formatting.
                parsed = _parse_number(value)
                if parsed is None:
                    return value, ValidationIssue(path, "expected a number", value)
                number = parsed
            else:
                return value, ValidationIssue(path, "expected a number", value)
            if kind is FieldType.INTEGER:
                if float(number) != int(number):
                    return number, ValidationIssue(
                        path, "expected a whole number", value
                    )
                return int(number), None
            return float(number), None

        if kind is FieldType.DATE:
            text = str(value).strip()
            if _ISO_DATE.fullmatch(text):
                return text, None
            if cfg.coerce:
                converted = self._coerce_date(text)
                if converted:
                    return converted, None
            return value, ValidationIssue(
                path, "expected a date formatted YYYY-MM-DD", value
            )

        return value, None

    def _coerce_date(self, text: str) -> str | None:
        """Handle the unambiguous rewrites only.

        Deliberately refuses 01/02/2024: it is 1 February or 2 January depending
        on where the document came from, and guessing silently corrupts data.
        A repair round that asks the model is the correct cost here.
        """
        match = re.fullmatch(r"(\d{4})[/.](\d{1,2})[/.](\d{1,2})", text)
        if match:
            year, month, day = match.groups()
            return year + "-" + month.zfill(2) + "-" + day.zfill(2)
        names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        months = {name: index for index, name in enumerate(names, start=1)}
        # Documents write "Mar 16, 2024" far more often than "March 16, 2024",
        # and the full-name-only lookup silently failed on every abbreviation.
        months.update({name[:3]: index for index, name in enumerate(names, start=1)})
        match = re.fullmatch(
            r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", text
        ) or re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
        if match:
            groups = list(match.groups())
            if groups[0].isdigit():
                day, month_name, year = groups
            else:
                month_name, day, year = groups
            lowered = month_name.lower().rstrip('.')
            month = months.get(lowered) or months.get(lowered[:3])
            if month:
                return year + "-" + str(month).zfill(2) + "-" + day.zfill(2)
        return None

    # --- grounding --------------------------------------------------------

    def _ground(
        self,
        data: Mapping[str, Any],
        schema: ExtractionSchema,
        segments: Sequence[tuple[str, Provenance | None]],
    ) -> list[FieldResult]:
        # Two views of each passage: as written, and with currency and thousands
        # separators removed. Without the second, a model correctly returning
        # 1240.5 for a document saying "$1,240.50" is reported as fabricated —
        # which would make the grounding signal noise exactly where it matters.
        haystacks = [
            (_normalise(text), _NUMBER_NOISE.sub("", _normalise(text)), prov, text)
            for text, prov in segments
        ]
        results: list[FieldResult] = []
        for spec in schema.fields:
            value = data.get(spec.name)
            if value is None:
                results.append(FieldResult(spec.name, None, grounded=False))
                continue
            prov, matched = self._locate(value, haystacks)
            results.append(
                FieldResult(
                    name=spec.name,
                    value=value,
                    grounded=prov is not None or matched != "",
                    provenance=prov,
                    matched_text=matched,
                )
            )
        return results

    def _locate(
        self,
        value: Any,
        haystacks: Sequence[tuple[str, str, Provenance | None, str]],
    ) -> tuple[Provenance | None, str]:
        """Find where a value appears in the source.

        Numbers are matched against the separator-stripped view and in several
        written forms, because '40', '40.0' and '40.00' are the same fact and a
        document picks whichever it likes. Booleans are never located: 'true'
        appears in prose constantly and would ground every boolean field
        spuriously. Containers are not located either — a list's groundedness is
        the union of its elements', and one box for the whole list would be a
        fiction.
        """
        if isinstance(value, bool):
            return None, ""

        if isinstance(value, (int, float)):
            numeric_needles = {_normalise(value)}
            if float(value) == int(value):
                numeric_needles.add(str(int(value)))
                numeric_needles.add(format(float(value), ".2f"))
            else:
                numeric_needles.add(format(float(value), ".2f"))
                numeric_needles.add(str(value).rstrip("0").rstrip("."))
            for _, numeric, prov, original in haystacks:
                for needle in numeric_needles:
                    if needle and needle in numeric:
                        return prov, original[:200]
            return None, ""

        if isinstance(value, str):
            token = _normalise(value)
            if len(token) < 2:
                return None, ""
            for haystack, _, prov, original in haystacks:
                if token in haystack:
                    return prov, original[:200]
            return None, ""

        return None, ""

    def _grounding_issues(
        self,
        data: Mapping[str, Any],
        schema: ExtractionSchema,
        segments: Sequence[tuple[str, Provenance | None]],
    ) -> list[ValidationIssue]:
        """Turn ungrounded values into repairable issues — except where the
        schema itself asked for a normalised form.

        A DATE field is required to come back as YYYY-MM-DD, so a document
        reading "Issued 14 March 2024" can never contain the value verbatim.
        Demanding grounding there fires on *correct* extractions, and a signal
        that flags correct work is worse than no signal: people learn to ignore
        it. Booleans are exempt for the same reason — 'unpaid' in the source
        becomes `false` in the object.

        Strings and numbers stay in scope, and those are where fabrication
        actually happens.
        """
        exempt = {FieldType.DATE, FieldType.BOOLEAN, FieldType.OBJECT, FieldType.ARRAY}
        checkable = {f.name for f in schema.fields if f.type not in exempt}
        issues: list[ValidationIssue] = []
        for result in self._ground(data, schema, segments):
            if (
                result.name in checkable
                and result.value is not None
                and not result.grounded
            ):
                issues.append(
                    ValidationIssue(
                        result.name,
                        "value does not appear in the document; quote it exactly"
                        " or return null",
                        result.value,
                    )
                )
        return issues


__all__ = ["ExtractionComponent", "extract_json_object"]

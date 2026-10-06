"""Schema-driven extraction with a repair loop and per-field provenance.

Every framework has extraction. Almost none have a repair loop that converges,
and that is where most of the accuracy lives. The difference is what happens on a
validation failure:

* The common approach re-runs the whole prompt and hopes. The model has no idea
  what was wrong, so it often reproduces the same mistake.
* This one sends back **only the fields that failed, with the specific error and
  the value it produced**. That turns a guess into a correction, and it is why
  two repair rounds are usually enough.

The second thing this adds is grounding: after validation, every extracted
scalar - including each cell inside a line-item table - is searched for among
the document's own words. A value that is found carries the merged box of the
words that matched, so a field can be clicked back to one cell rather than to
the whole table, and a match class saying how exactly it matched. A value that
is not found anywhere is flagged. It is not necessarily wrong (a total can be
computed, a date reformatted), but every fabricated value is in that set, so it
is the set a human should see first.

The matching itself lives in `grounding`, because getting it right is most of
the work: see that module for why substring matching reported a fabricated
quantity as found.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from ..core.errors import ValidationFailed
from ..core.models import BBox, Chunk, Document, Message, Provenance, Usage, Word
from .dates import coerce_date
from .grounding import Grounder, Match
from .models import (
    Evidence,
    ExtractionConfig,
    ExtractionRequest,
    ExtractionResult,
    ExtractionSchema,
    FieldResult,
    FieldSpec,
    FieldType,
    MatchClass,
    ValidationIssue,
)
from .numbers import parse_decimal, parse_number

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TRUE = frozenset({"true", "yes", "y", "1"})
_FALSE = frozenset({"false", "no", "n", "0"})
_GROUNDING_EXEMPT = frozenset(
    {FieldType.DATE, FieldType.BOOLEAN, FieldType.OBJECT, FieldType.ARRAY}
)

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


_parse_number = parse_number
"""Kept as a module-level name because it is the documented seam the stress
regression tests import. The implementation moved to `numbers`, which both the
float and the `Decimal` parsers now share."""


class ExtractionComponent:
    """Document (or text) plus a schema in, validated object out."""

    def __init__(self, llm: Any) -> None:
        if llm is None:
            raise ValueError("ExtractionComponent requires an LLM")
        self._llm = llm
        self._words: Sequence[Word] = ()

    def execute(self, input_data: ExtractionRequest) -> ExtractionResult:
        cfg = input_data.config
        segments = self._segments(input_data.source)
        context, sent_chars, total_chars = self._context(segments, cfg)
        warnings: list[str] = []
        truncated = sent_chars < total_chars
        if truncated:
            # Reported, never repaired. A repair round cannot put back text the
            # model was never shown, and dressing it up as a validation issue
            # would burn both attempts chasing an unfixable complaint.
            warnings.append(
                "document truncated: sent "
                + str(sent_chars)
                + " of "
                + str(total_chars)
                + " characters (context_char_limit="
                + str(cfg.context_char_limit)
                + "); fields stated only in the omitted text cannot be found"
            )

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
                _, leaf_pairs = self._ground_all(data, input_data.schema, segments, cfg)
                issues = issues + self._grounding_issues(leaf_pairs)

            if not issues:
                break
            if attempt < cfg.max_repairs:
                messages = messages[:2] + [
                    # default=str because a DECIMAL field holds a Decimal, which
                    # json cannot serialise. Without it the second attempt died
                    # here rather than repairing anything.
                    Message(
                        "assistant",
                        json.dumps(data, ensure_ascii=False, default=str),
                    ),
                    Message("user", self._repair_prompt(issues, input_data.schema)),
                ]

        fields, leaf_pairs = self._ground_all(data, input_data.schema, segments, cfg)
        leaves = [leaf for leaf, _ in leaf_pairs]

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
            leaves=leaves,
            truncated=truncated,
            warnings=warnings,
        )

    # --- source handling --------------------------------------------------

    def _segments(self, source: Any) -> list[tuple[str, Provenance | None]]:
        """Flatten any accepted source into (text, provenance) pairs.

        Keeping provenance attached at this level is what lets a field point back
        at a page later; flattening to one string first would discard it.
        """
        self._words = ()
        if isinstance(source, str):
            return [(source, None)]
        if isinstance(source, Document):
            # The evidence layer, when the source kept one. Grounding prefers it
            # over block text because a block is a whole table.
            self._words = source.words
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
    ) -> tuple[str, int, int]:
        """The prompt's document text, plus how much of it was sent and how much
        there was.

        Returning the two counts is the whole point: the previous version
        stopped at the limit and said nothing, so a long invoice lost its totals
        and the result looked as confident as any other.
        """
        parts: list[str] = []
        used = 0
        total = 0
        stopped = False
        for text, prov in segments:
            label = "(page " + str(prov.page) + ") " if prov else ""
            piece = label + text
            total += len(piece)
            if stopped:
                continue
            if used + len(piece) > cfg.context_char_limit and parts:
                stopped = True
                continue
            parts.append(piece)
            used += len(piece)
        return "\n".join(parts), used, total

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
        if spec.type is FieldType.DECIMAL:
            bits.append("exact decimal digits only, no currency symbol or code")
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
            coerced, found = self._validate_value(value, spec, cfg, path)
            data[spec.name] = coerced
            issues.extend(found)
        return data, issues

    def _validate_value(
        self, value: Any, spec: FieldSpec, cfg: ExtractionConfig, path: str
    ) -> tuple[Any, list[ValidationIssue]]:
        """Validate one value and report **every** problem under it.

        The measured defect: this returned the first issue inside an array and
        stopped, so a 30-row table with four bad rows took four repair rounds to
        fix and the configured two were never enough. One attempt that names all
        four costs the same tokens and converges.
        """
        kind = spec.type

        if kind is FieldType.OBJECT:
            if not isinstance(value, Mapping):
                return None, [ValidationIssue(path, "expected an object", value)]
            nested, nested_issues = self._validate_object(
                value, spec.fields, cfg, path + "."
            )
            return nested, nested_issues

        if kind is FieldType.ARRAY:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return None, [ValidationIssue(path, "expected an array", value)]
            out: list[Any] = []
            issues: list[ValidationIssue] = []
            for index, item in enumerate(value):
                item_path = path + "[" + str(index) + "]"
                if spec.fields:
                    if not isinstance(item, Mapping):
                        issues.append(
                            ValidationIssue(item_path, "expected an object", item)
                        )
                        out.append(None)
                        continue
                    nested, nested_issues = self._validate_object(
                        item, spec.fields, cfg, item_path + "."
                    )
                    out.append(nested)
                    issues.extend(nested_issues)
                else:
                    element = FieldSpec(
                        name=spec.name, type=spec.item_type or FieldType.STRING
                    )
                    coerced, found = self._validate_value(item, element, cfg, item_path)
                    out.append(coerced)
                    issues.extend(found)
            return out, issues

        coerced, issue = self._scalar(value, spec, cfg, path)
        if issue is not None:
            return coerced, [issue]

        if spec.enum and str(coerced) not in spec.enum:
            return coerced, [
                ValidationIssue(
                    path, "must be one of: " + ", ".join(spec.enum), coerced
                )
            ]
        if spec.pattern and not re.fullmatch(spec.pattern, str(coerced)):
            return coerced, [
                ValidationIssue(
                    path, "must match the pattern " + spec.pattern, coerced
                )
            ]
        if isinstance(coerced, (int, float, Decimal)) and not isinstance(coerced, bool):
            if spec.minimum is not None and coerced < self._bound(coerced, spec.minimum):
                return coerced, [
                    ValidationIssue(
                        path, "must be at least " + str(spec.minimum), coerced
                    )
                ]
            if spec.maximum is not None and coerced > self._bound(coerced, spec.maximum):
                return coerced, [
                    ValidationIssue(
                        path, "must be at most " + str(spec.maximum), coerced
                    )
                ]
        return coerced, []

    @staticmethod
    def _bound(value: Any, limit: float) -> Any:
        """A bound in the same type as the value it constrains. Comparing a
        Decimal against a float bound is legal but goes through the float's
        binary expansion, so 0.1 as a limit rejects Decimal('0.1')."""
        return Decimal(str(limit)) if isinstance(value, Decimal) else limit

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

        if kind is FieldType.DECIMAL:
            if isinstance(value, bool):
                # Before the int branch: bool is a subclass of int, and
                # Decimal(True) is a silent 1.
                return value, ValidationIssue(path, "expected a number", value)
            if isinstance(value, Decimal):
                return value, None
            if isinstance(value, int):
                return Decimal(value), None
            if isinstance(value, float):
                # str() first, always. Decimal(1240.50) is
                # 1240.50000000000004547473508864641189575195312500, because
                # 1240.50 has no binary representation; Decimal(str(1240.50))
                # is 1240.50, which is the fact the document stated.
                return Decimal(str(value)), None
            if cfg.coerce and isinstance(value, str):
                exact = parse_decimal(value)
                if exact is not None:
                    return exact, None
            return value, ValidationIssue(path, "expected a number", value)

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
                converted = self._coerce_date(text, cfg.date_order)
                if converted:
                    return converted, None
            detail = (
                "expected a date formatted YYYY-MM-DD"
                if cfg.date_order
                else "expected a date formatted YYYY-MM-DD; this one is ambiguous"
                " without knowing whether the document writes day or month first"
            )
            return value, ValidationIssue(path, detail, value)

        return value, None

    def _coerce_date(self, text: str, date_order: str | None = None) -> str | None:
        """Rewrite a written date as YYYY-MM-DD, or refuse.

        `01/02/2024` is 1 February or 2 January depending on where the document
        came from. Without `date_order` it stays refused and costs a repair
        round, which is the correct price: guessing silently corrupts data. The
        rules live in `dates`, because grounding has to read a date back the
        same way this wrote it.
        """
        return coerce_date(text, date_order)

    # --- grounding --------------------------------------------------------

    def _grounder(
        self,
        segments: Sequence[tuple[str, Provenance | None]],
        cfg: ExtractionConfig,
    ) -> Grounder:
        return Grounder(
            segments,
            words=self._words,
            date_order=cfg.date_order,
            ocr_confidence=cfg.ocr_confidence_threshold,
        )

    def _ground_all(
        self,
        data: Mapping[str, Any],
        schema: ExtractionSchema,
        segments: Sequence[tuple[str, Provenance | None]],
        cfg: ExtractionConfig,
    ) -> tuple[list[FieldResult], list[tuple[FieldResult, FieldSpec]]]:
        """Ground every value in the result.

        Returns the top-level fields (unchanged shape, for every existing
        caller) and every scalar leaf paired with its spec. The pairing is
        internal: the grounding-issue rules need a leaf's declared type, and a
        `FieldResult` deliberately does not carry one.
        """
        grounder = self._grounder(segments, cfg)
        claimed: set[int] = set()
        fields: list[FieldResult] = []
        leaves: list[tuple[FieldResult, FieldSpec]] = []
        for spec in schema.fields:
            result, found = self._ground_value(
                data.get(spec.name), spec, grounder, spec.name, None, claimed
            )
            fields.append(result)
            leaves.extend(found)
        return fields, leaves

    def _ground_value(
        self,
        value: Any,
        spec: FieldSpec,
        grounder: Grounder,
        path: str,
        line: BBox | None,
        claimed: set[int],
    ) -> tuple[FieldResult, list[tuple[FieldResult, FieldSpec]]]:
        if spec.type is FieldType.OBJECT:
            return self._ground_object(value, spec, grounder, path, line, claimed)
        if spec.type is FieldType.ARRAY:
            return self._ground_array(value, spec, grounder, path, line, claimed)
        return self._ground_scalar(value, spec, grounder, path, line, claimed)

    def _ground_scalar(
        self,
        value: Any,
        spec: FieldSpec,
        grounder: Grounder,
        path: str,
        line: BBox | None,
        claimed: set[int],
    ) -> tuple[FieldResult, list[tuple[FieldResult, FieldSpec]]]:
        if value is None:
            result = FieldResult(
                spec.name, None, grounded=False, path=path, match=MatchClass.NOT_CHECKED
            )
            return result, [(result, spec)]
        found = grounder.locate(value, spec.type, line=line, exclude=claimed)
        if found.start >= 0:
            # One token is evidence for one value. Without this, row 0's
            # `amount` grounds on row 0's identical `unit_price` and the box
            # points at the wrong column - which reads as correct.
            claimed.update(range(found.start, found.start + found.length))
        result = self._result(spec.name, value, path, found)
        return result, [(result, spec)]

    def _ground_object(
        self,
        value: Any,
        spec: FieldSpec,
        grounder: Grounder,
        path: str,
        line: BBox | None,
        claimed: set[int],
    ) -> tuple[FieldResult, list[tuple[FieldResult, FieldSpec]]]:
        container = FieldResult(
            spec.name, value, grounded=False, path=path, match=MatchClass.NOT_CHECKED
        )
        if not isinstance(value, Mapping):
            return container, []
        leaves = self._ground_row(value, spec.fields, grounder, path, line, claimed)
        return container, leaves

    def _ground_array(
        self,
        value: Any,
        spec: FieldSpec,
        grounder: Grounder,
        path: str,
        line: BBox | None,
        claimed: set[int],
    ) -> tuple[FieldResult, list[tuple[FieldResult, FieldSpec]]]:
        """Ground every cell of a table, which the previous version never did.

        An array's own location is not a thing: one box around a 30-row table
        is a fiction. Its *elements* have locations, and those are what a
        reviewer needs.
        """
        container = FieldResult(
            spec.name, value, grounded=False, path=path, match=MatchClass.NOT_CHECKED
        )
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return container, []

        leaves: list[tuple[FieldResult, FieldSpec]] = []
        for index, item in enumerate(value):
            item_path = path + "[" + str(index) + "]"
            if spec.fields:
                if not isinstance(item, Mapping):
                    continue
                leaves.extend(
                    self._ground_row(
                        item, spec.fields, grounder, item_path, line, claimed
                    )
                )
            else:
                element = FieldSpec(
                    name=spec.name, type=spec.item_type or FieldType.STRING
                )
                result, found = self._ground_scalar(
                    item, element, grounder, item_path, line, claimed
                )
                leaves.extend(found)
        return container, leaves

    def _ground_row(
        self,
        row: Mapping[str, Any],
        specs: Sequence[FieldSpec],
        grounder: Grounder,
        path: str,
        line: BBox | None,
        claimed: set[int],
    ) -> list[tuple[FieldResult, FieldSpec]]:
        """Ground one row's cells, anchored on its most distinctive string.

        Row locality is the difference between evidence and decoration. A
        `quantity` of 1 appears in five rows of the fixture; without an anchor
        it grounds on the first and the box points at the wrong row, which is
        worse than no box because it looks right. So the longest string cell -
        a description, in practice - is grounded first and its line becomes the
        only place the other cells may match.
        """
        anchor_spec = self._row_anchor(row, specs)
        found: dict[str, list[tuple[FieldResult, FieldSpec]]] = {}

        def ground(spec: FieldSpec, current: BBox | None) -> FieldResult:
            result, leaves = self._ground_value(
                row.get(spec.name),
                spec,
                grounder,
                path + "." + spec.name,
                current,
                claimed,
            )
            found[spec.name] = leaves
            return result

        row_line = line
        if anchor_spec is not None:
            anchor = ground(anchor_spec, line)
            if anchor.evidence and anchor.evidence[0].bbox is not None:
                row_line = anchor.evidence[0].bbox

        for spec in specs:
            if spec.name not in found:
                ground(spec, row_line)

        # Emit in schema order, not in the order they were grounded.
        return [leaf for spec in specs for leaf in found.get(spec.name, [])]

    @staticmethod
    def _row_anchor(row: Mapping[str, Any], specs: Sequence[FieldSpec]) -> FieldSpec | None:
        """The row's most distinctive cell: the longest string it states.

        A description is far less likely than a quantity or a price to repeat
        elsewhere on the page, so it is the cell whose match can be trusted to
        identify the row rather than merely to exist.
        """
        best: FieldSpec | None = None
        best_length = 0
        for spec in specs:
            if spec.type is not FieldType.STRING:
                continue
            value = row.get(spec.name)
            if not isinstance(value, str):
                continue
            length = len(" ".join(value.split()))
            if length > best_length:
                best, best_length = spec, length
        return best if best_length >= 3 else None

    @staticmethod
    def _result(name: str, value: Any, path: str, found: Match) -> FieldResult:
        evidence: list[Evidence] = list(found.evidence)
        provenance = evidence[0].as_provenance() if evidence else None
        return FieldResult(
            name=name,
            value=value,
            grounded=found.grounded,
            provenance=provenance,
            matched_text=evidence[0].text if evidence else "",
            path=path,
            match=found.match,
            evidence=evidence,
        )

    def _grounding_issues(
        self, leaves: Sequence[tuple[FieldResult, FieldSpec]]
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
        issues: list[ValidationIssue] = []
        for result, spec in leaves:
            if spec.type in _GROUNDING_EXEMPT or result.value is None:
                continue
            if result.grounded or result.match is MatchClass.NOT_CHECKED:
                continue
            detail = (
                "value does not appear in the document; quote it exactly"
                " or return null"
            )
            if result.match is MatchClass.FUZZY_OCR and result.evidence:
                detail = (
                    "value does not appear in the document; the nearest word is"
                    " '"
                    + result.evidence[0].text
                    + "', read with low confidence"
                )
            issues.append(
                ValidationIssue(result.path or result.name, detail, result.value)
            )
        return issues


__all__ = ["ExtractionComponent", "extract_json_object"]

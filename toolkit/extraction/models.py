"""Data contracts for schema-driven extraction.

The schema is expressed in plain dataclasses rather than Pydantic, so this module
works in a stdlib-only install. `ExtractionSchema.from_pydantic` converts a real
Pydantic model when you have one, because most projects already describe their
shapes that way and re-declaring them here would be a tax.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.models import Provenance, Usage


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    """ISO 8601 date (YYYY-MM-DD). Kept distinct from STRING because a date is
    the field type models most often return in a format nobody asked for."""
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class FieldSpec:
    name: str
    type: FieldType = FieldType.STRING
    description: str = ""
    """Shown to the model verbatim. This is the highest-leverage text in the
    whole pipeline: most extraction failures are underspecified fields, not weak
    models."""
    required: bool = True
    enum: Sequence[str] = field(default_factory=list)
    pattern: str = ""
    """Regex the value must match in full."""
    minimum: float | None = None
    maximum: float | None = None
    item_type: FieldType | None = None
    """Element type for ARRAY."""
    fields: Sequence[FieldSpec] = field(default_factory=list)
    """Nested specs for OBJECT, or for ARRAY of OBJECT."""
    examples: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("field name must not be empty")
        if self.type is FieldType.ARRAY and self.item_type is None and not self.fields:
            raise ValueError(
                "array field '" + self.name + "' needs item_type or nested fields"
            )
        if self.type is FieldType.OBJECT and not self.fields:
            raise ValueError("object field '" + self.name + "' needs nested fields")


@dataclass
class ExtractionSchema:
    name: str
    fields: Sequence[FieldSpec]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("schema '" + self.name + "' has no fields")
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("duplicate field names in schema '" + self.name + "'")

    @staticmethod
    def from_pydantic(model: Any) -> ExtractionSchema:
        """Convert a Pydantic v2 model into a schema.

        Reads the model's JSON Schema rather than its Python annotations, because
        that is the representation Pydantic itself considers authoritative and it
        already resolves defaults, aliases and nested models.
        """
        if not hasattr(model, "model_json_schema"):
            raise TypeError(
                "expected a Pydantic v2 model with model_json_schema(); got "
                + type(model).__name__
            )
        raw = model.model_json_schema()
        return _schema_from_json_schema(
            raw, name=getattr(model, "__name__", raw.get("title", "Extraction"))
        )

    def field_map(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields}


_JSON_TYPES = {
    "string": FieldType.STRING,
    "integer": FieldType.INTEGER,
    "number": FieldType.NUMBER,
    "boolean": FieldType.BOOLEAN,
    "array": FieldType.ARRAY,
    "object": FieldType.OBJECT,
}


def _resolve(node: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = node.get("$ref")
    if not ref:
        return node
    key = str(ref).rsplit("/", 1)[-1]
    return defs.get(key, {})


def _field_from_json_schema(
    name: str, node: Mapping[str, Any], defs: Mapping[str, Any], required: bool
) -> FieldSpec:
    node = _resolve(node, defs)
    # Optional fields arrive as anyOf[T, null]; unwrap to the real type.
    options = [o for o in node.get("anyOf", []) if o.get("type") != "null"]
    if options:
        merged = dict(_resolve(options[0], defs))
        merged.setdefault("description", node.get("description", ""))
        node = merged
        required = False

    raw_type = node.get("type", "string")
    field_type = _JSON_TYPES.get(str(raw_type), FieldType.STRING)
    if field_type is FieldType.STRING and node.get("format") == "date":
        field_type = FieldType.DATE

    nested: list[FieldSpec] = []
    item_type: FieldType | None = None
    if field_type is FieldType.OBJECT:
        nested = _fields_from_properties(node, defs)
    elif field_type is FieldType.ARRAY:
        items = _resolve(node.get("items", {}), defs)
        item_type = _JSON_TYPES.get(str(items.get("type", "string")), FieldType.STRING)
        if item_type is FieldType.OBJECT:
            nested = _fields_from_properties(items, defs)

    return FieldSpec(
        name=name,
        type=field_type,
        description=str(node.get("description", "")),
        required=required,
        enum=[str(v) for v in node.get("enum", [])],
        pattern=str(node.get("pattern", "")),
        minimum=node.get("minimum"),
        maximum=node.get("maximum"),
        item_type=item_type,
        fields=nested,
    )


def _fields_from_properties(
    node: Mapping[str, Any], defs: Mapping[str, Any]
) -> list[FieldSpec]:
    required = set(node.get("required", []))
    return [
        _field_from_json_schema(key, value, defs, key in required)
        for key, value in node.get("properties", {}).items()
    ]


def _schema_from_json_schema(raw: Mapping[str, Any], name: str) -> ExtractionSchema:
    defs = raw.get("$defs", {})
    return ExtractionSchema(
        name=name,
        description=str(raw.get("description", "")),
        fields=_fields_from_properties(raw, defs),
    )


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    """Dotted path, e.g. 'line_items[2].unit_price'. Precise enough that a repair
    prompt can name exactly what to fix rather than asking for the whole object
    again."""
    message: str
    got: Any = None

    def render(self) -> str:
        detail = "" if self.got is None else " (got " + repr(self.got)[:80] + ")"
        return self.path + ": " + self.message + detail


@dataclass(frozen=True)
class FieldResult:
    name: str
    value: Any
    grounded: bool
    """Was this value found verbatim in the source text?

    An ungrounded value is not necessarily wrong — a total can be computed, a
    date reformatted — but it is the set worth showing a human first, because
    every fabricated value lands in it."""
    provenance: Provenance | None = None
    matched_text: str = ""


@dataclass
class ExtractionConfig:
    max_repairs: int = 2
    """Repair rounds after the first attempt. Two is where the returns flatten:
    a model that cannot satisfy a schema in three tries usually has a schema
    problem, not a luck problem."""
    temperature: float = 0.0
    max_tokens: int | None = 1500
    require_grounding: bool = False
    """Treat an ungrounded value as a validation issue and try to repair it.
    Right for verbatim extraction, wrong when fields are derived or normalised."""
    strict: bool = False
    """Raise instead of returning a result carrying issues."""
    context_char_limit: int = 12000
    coerce: bool = True
    """Accept '40' for an integer and 'yes' for a boolean. Models return strings;
    refusing them wastes a repair round on a problem you can solve locally."""

    def __post_init__(self) -> None:
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")


@dataclass
class ExtractionRequest:
    schema: ExtractionSchema
    source: Any
    """A Document, a sequence of Chunks, or a plain string."""
    config: ExtractionConfig = field(default_factory=ExtractionConfig)
    instructions: str = ""
    """Extra guidance appended to the prompt."""


@dataclass
class ExtractionResult:
    data: Mapping[str, Any]
    fields: Sequence[FieldResult]
    issues: Sequence[ValidationIssue]
    attempts: int
    usage: Usage = field(default_factory=Usage)
    raw_responses: Sequence[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def ungrounded(self) -> list[FieldResult]:
        return [f for f in self.fields if not f.grounded and f.value is not None]

    def field_map(self) -> dict[str, FieldResult]:
        return {f.name: f for f in self.fields}


@dataclass(frozen=True)
class Segment:
    """One logical document inside a larger file."""

    start_page: int
    end_page: int
    label: str = ""
    confidence: float = 0.0
    reason: str = ""
    """Why the splitter believed a boundary was here. Makes a wrong split
    diagnosable instead of mysterious."""


@dataclass
class SplitResult:
    segments: Sequence[Segment]
    page_count: int

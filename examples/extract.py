"""Document to validated object, with every field traceable back to a page.

Runs offline with a scripted model, so the repair loop is visible without an API
key. Swap ScriptedLLM for LiteLLMClient to run it for real.

    python examples/extract.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.adapters import ScriptedLLM  # noqa: E402
from toolkit.core import BBox, Block, BlockType, Document, Provenance  # noqa: E402
from toolkit.extraction import (  # noqa: E402
    ExtractionComponent,
    ExtractionConfig,
    ExtractionRequest,
    ExtractionSchema,
    FieldSpec,
    FieldType,
)

INVOICE = [
    "ACME Industrial Supplies Ltd",
    "Invoice number INV-88213",
    "Issued 14 March 2024",
    "Bill to: Northwind Trading",
    "Subtotal: $1,033.75",
    "Tax: $206.75",
    "Total due: $1,240.50",
    "Status: unpaid",
]

SCHEMA = ExtractionSchema(
    name="Invoice",
    description="A commercial invoice.",
    fields=[
        FieldSpec("invoice_number", FieldType.STRING, "The invoice reference code"),
        FieldSpec("supplier", FieldType.STRING, "Company issuing the invoice"),
        FieldSpec("issued", FieldType.DATE, "Date the invoice was issued"),
        FieldSpec("total", FieldType.NUMBER, "Total amount due", minimum=0),
        FieldSpec("status", FieldType.STRING, "Payment status",
                  enum=["paid", "unpaid", "overdue"]),
    ],
)

# Attempt 1 has three defects on purpose: an invalid enum, a non-ISO date and a
# money string. Only the first two should cost a repair round -- the currency
# string is coerced locally.
ATTEMPT_1 = json.dumps(
    {
        "invoice_number": "INV-88213",
        "supplier": "ACME Industrial Supplies Ltd",
        "issued": "14/03/2024",
        "total": "$1,240.50",
        "status": "not paid",
    }
)
ATTEMPT_2 = json.dumps(
    {
        "invoice_number": "INV-88213",
        "supplier": "ACME Industrial Supplies Ltd",
        "issued": "2024-03-14",
        "total": "$1,240.50",
        "status": "unpaid",
    }
)


def document() -> Document:
    return Document(
        doc_id="doc:invoice",
        source_uri="invoice.pdf",
        page_count=1,
        blocks=[
            Block(line, BlockType.PARAGRAPH,
                  Provenance(1, BBox(50, 60 + i * 22, 420, 78 + i * 22)))
            for i, line in enumerate(INVOICE)
        ],
    )


def main() -> int:
    out = sys.stdout.write
    llm = ScriptedLLM(responses=[ATTEMPT_1, ATTEMPT_2])

    result = ExtractionComponent(llm).execute(
        ExtractionRequest(SCHEMA, document(), ExtractionConfig(max_repairs=2))
    )

    out("attempts: " + str(result.attempts) + "   valid: " + str(result.valid) + "\n\n")

    out("--- what the repair prompt said ---\n")
    out(llm.calls[1][-1].content + "\n\n")

    out("--- extracted ---\n")
    for name, value in result.data.items():
        out("  " + name.ljust(16) + repr(value) + "\n")

    out("\n--- provenance (click any field back to the page) ---\n")
    for field in result.fields:
        if field.provenance and field.provenance.bbox:
            box = field.provenance.bbox
            out(
                "  "
                + field.name.ljust(16)
                + "page "
                + str(field.provenance.page)
                + "  bbox "
                + str([round(v) for v in box.as_tuple()])
                + "  <- "
                + field.matched_text[:44]
                + "\n"
            )
        else:
            out("  " + field.name.ljust(16) + "(not located verbatim)\n")

    if result.ungrounded:
        out("\nungrounded fields: " + ", ".join(f.name for f in result.ungrounded) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

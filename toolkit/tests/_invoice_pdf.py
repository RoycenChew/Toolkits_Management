"""A synthetic two-page invoice, written by hand so no PDF library is needed.

Built on the stress corpus's minimal PDF writer. The layout is deliberately the
one that exposed the extraction defects: a header block, a line-item table that
continues onto page 2 under a repeated header, and totals at the end. Values
are chosen so that a quantity of 7 never appears as a word of its own, while
the digit 7 appears inside the invoice number - the substring that made the old
grounding report a fabricated value as found.

Not collected by pytest (leading underscore); imported by the tests that need it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STRESS = os.path.join(_ROOT, "stress")
if _STRESS not in sys.path:
    sys.path.insert(0, _STRESS)

from make_corpus import Page, write_pdf  # noqa: E402

_ITEMS = (
    "Hex bolt stainless", "Washer flat", "Nylon lock nut", "Cable tie black",
    "Drill bit set", "Masking tape", "Wire stripper", "Safety goggles",
    "Work gloves", "Spirit level", "Measuring tape", "Utility knife",
)


@dataclass
class InvoiceFixture:
    path: str
    invoice_number: str = "INV-2026-0417"
    tax_id: str = "C2584563200"
    po_number: str = "PO-55120"
    date_text: str = "03/10/2026"
    currency: str = "MYR"
    lines: list[dict] = field(default_factory=list)
    subtotal: Decimal = Decimal("0")
    tax: Decimal = Decimal("0")
    total: Decimal = Decimal("0")


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _table_header(page: Page, y: float) -> None:
    page.text(56, y, "Description", 10, "bold")
    page.text(330, y, "Qty", 10, "bold")
    page.text(390, y, "Unit Price", 10, "bold")
    page.text(480, y, "Amount", 10, "bold")


def invoice_pdf(rows: int = 30, rows_on_first_page: int = 22) -> InvoiceFixture:
    """Write the invoice and return what a correct extraction should find."""
    fixture = InvoiceFixture(path=os.path.join(tempfile.mkdtemp(), "invoice.pdf"))
    for index in range(rows):
        quantity = index % 6 + 1  # 1..6: never 7, by design
        unit = (Decimal("3.50") * (index + 1) + Decimal("0.25")).quantize(Decimal("0.01"))
        amount = (unit * quantity).quantize(Decimal("0.01"))
        name = _ITEMS[index % len(_ITEMS)] + " " + chr(65 + index % 26)
        fixture.lines.append(
            {"description": name, "quantity": quantity, "unit_price": unit, "amount": amount}
        )
    fixture.subtotal = sum((line["amount"] for line in fixture.lines), Decimal("0"))
    fixture.tax = (fixture.subtotal * Decimal("0.08")).quantize(Decimal("0.01"))
    fixture.total = fixture.subtotal + fixture.tax

    first = Page()
    first.text(56, 740, "Borneo Industrial Supply Sdn Bhd", 16, "bold")
    first.text(56, 722, "Tax ID: " + fixture.tax_id, 10)
    first.text(56, 692, "INVOICE", 14, "bold")
    first.text(56, 672, "Invoice No: " + fixture.invoice_number, 10)
    first.text(330, 672, "Date: " + fixture.date_text, 10)
    first.text(56, 656, "PO Number: " + fixture.po_number, 10)
    _table_header(first, 624)

    second = Page()
    second.text(56, 740, "Borneo Industrial Supply Sdn Bhd", 10)
    _table_header(second, 712)

    for index, line in enumerate(fixture.lines):
        page, y = (
            (first, 604 - index * 16)
            if index < rows_on_first_page
            else (second, 692 - (index - rows_on_first_page) * 16)
        )
        page.text(56, y, line["description"], 10)
        page.text(330, y, str(line["quantity"]), 10)
        page.text(390, y, _money(line["unit_price"]), 10)
        page.text(480, y, _money(line["amount"]), 10)

    last_y = 692 - (rows - rows_on_first_page) * 16 - 24
    second.text(330, last_y, "Subtotal: " + fixture.currency + " " + _money(fixture.subtotal), 10)
    second.text(330, last_y - 16, "SST 8%: " + fixture.currency + " " + _money(fixture.tax), 10)
    second.text(
        330, last_y - 32, "Total Due: " + fixture.currency + " " + _money(fixture.total), 10, "bold"
    )
    write_pdf(fixture.path, [first, second])
    return fixture


def pdfplumber_available() -> bool:
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        return False
    return True

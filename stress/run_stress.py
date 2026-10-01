"""Run the pipeline against the hostile corpus and report what breaks.

Every probe targets one assumption. A probe reports PASS, FAIL or KNOWN (an
already-documented limitation being confirmed). The output is a findings table,
not a green tick — the purpose is to locate defects, so FAIL rows are the
valuable output.

    python stress/run_stress.py
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_corpus import build  # noqa: E402

from toolkit.adapters import PdfPlumberSource, PlainTextSource, ScriptedLLM  # noqa: E402
from toolkit.chunking import (  # noqa: E402
    ChunkConfig,
    ChunkerComponent,
    ChunkRequest,
    estimate_tokens,
)
from toolkit.core import AdapterError, BlockType, ScreeningRejected  # noqa: E402
from toolkit.extraction import (  # noqa: E402
    DocumentSplitterComponent,
    ExtractionComponent,
    ExtractionRequest,
    ExtractionSchema,
    FieldSpec,
    FieldType,
)
from toolkit.pipelines import AskConfig, KnowledgeBase  # noqa: E402

FINDINGS: list[tuple[str, str, str, str]] = []


def record(status: str, area: str, probe: str, detail: str) -> None:
    FINDINGS.append((status, area, probe, detail))


def load(path: str):
    source = PdfPlumberSource() if path.lower().endswith(".pdf") else PlainTextSource()
    return source.load(path)


# --------------------------------------------------------------------------


def probe_two_column(directory: str) -> None:
    doc = load(os.path.join(directory, "two_column_paper.pdf"))
    texts = [b.text for b in doc.content_blocks()]
    joined = " || ".join(texts)

    # Left column must be read before the right on page 1.
    abstract = next((i for i, t in enumerate(texts) if "Abstract" in t), None)
    intro = next((i for i, t in enumerate(texts) if "Introduction" in t), None)
    if abstract is None or intro is None:
        record("FAIL", "doc_layout", "two-column reading order",
               "abstract or introduction text missing entirely")
    elif abstract < intro:
        record("PASS", "doc_layout", "two-column reading order",
               "left column precedes right")
    else:
        record("FAIL", "doc_layout", "two-column reading order",
               "columns interleaved: intro at %d before abstract at %d" % (intro, abstract))

    # Running header must be classified as furniture on a 2-page doc.
    header_blocks = [b for b in doc.blocks if "JOURNAL OF APPLIED" in b.text]
    if any(b.type is BlockType.PAGE_HEADER for b in header_blocks):
        record("PASS", "doc_layout", "running header detected", "classified as furniture")
    else:
        kinds = {b.type.value for b in header_blocks} or {"absent"}
        record("FAIL", "doc_layout", "running header detected",
               "header classified as %s; needs >=3 pages to detect, this has 2"
               % ",".join(sorted(kinds)))

    # Footnotes should not be spliced into body prose.
    if "Corresponding author" in joined:
        body_with_footnote = [
            t for t in texts
            if "Corresponding author" in t and len(t) > 80 and "substrate" in t.lower()
        ]
        if body_with_footnote:
            record("FAIL", "doc_layout", "footnote isolation",
                   "footnote merged into a body block")
        else:
            record("PASS", "doc_layout", "footnote isolation",
                   "footnote kept as its own block")

    # A full-width heading over two columns on page 2.
    method = [b for b in doc.blocks if b.text.strip() == "2. Method"]
    if method and method[0].type is BlockType.HEADING:
        record("PASS", "doc_layout", "full-width heading over columns", "recognised")
    else:
        got = method[0].type.value if method else "missing"
        record("FAIL", "doc_layout", "full-width heading over columns", "got " + got)


def probe_hyphenation(directory: str) -> None:
    doc = load(os.path.join(directory, "hyphenated_contract.pdf"))
    text = " ".join(b.text for b in doc.content_blocks())
    broken = re.findall(r"\w+-\s+\w+", text)
    if broken:
        record("FAIL", "doc_layout", "line-break hyphenation",
               "%d words left split: %s" % (len(broken), ", ".join(broken[:3])))
    else:
        record("PASS", "doc_layout", "line-break hyphenation", "words rejoined")


def probe_false_headings(directory: str) -> None:
    doc = load(os.path.join(directory, "emphasis_not_headings.pdf"))
    headings = [(b.text, b.level) for b in doc.blocks if b.type is BlockType.HEADING]
    bad = [t for t, _ in headings if t.startswith(("WARNING", "NOTE", "Important"))]
    if bad:
        record("FAIL", "doc_layout", "emphasis vs heading",
               "%d emphasis lines promoted to headings: %s" % (len(bad), "; ".join(bad[:3])[:70]))
    else:
        record("PASS", "doc_layout", "emphasis vs heading",
               "%d headings, none from emphasis" % len(headings))


def probe_table(directory: str) -> None:
    doc = load(os.path.join(directory, "table_heavy.pdf"))
    text = " ".join(b.text for b in doc.content_blocks())
    # Does a row survive as a readable unit, or are cells scrambled across rows?
    intact = sum(1 for v in ("20V", "30V", "35V", "40V", "45V") if v in text)
    row_40 = re.search(r"40V\s+88\.1", text)
    record(
        "KNOWN" if row_40 else "FAIL",
        "doc_layout",
        "table row association",
        ("rows readable as text (%d/5 values present); structure still lost"
         % intact) if row_40
        else "cells scrambled: 40V not adjacent to 88.1 (%d/5 values present)" % intact,
    )


def probe_scan(directory: str) -> None:
    try:
        load(os.path.join(directory, "no_text_layer.pdf"))
    except AdapterError as exc:
        ok = "scanned" in str(exc).lower() or "ocr" in str(exc).lower()
        record("PASS" if ok else "FAIL", "adapters", "no-text-layer PDF",
               "clear error" if ok else "errored but unhelpfully: " + str(exc)[:60])
    except ScreeningRejected as exc:
        record("PASS", "adapters", "no-text-layer PDF",
               "screened out: " + exc.reason.value)
    else:
        record("FAIL", "adapters", "no-text-layer PDF",
               "returned a Document with no text instead of failing")


def probe_unicode(directory: str) -> None:
    doc = load(os.path.join(directory, "unicode_mess.md"))
    text = " ".join(b.text for b in doc.content_blocks())
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]

    issues = []
    if "ﬁ" in text or "ﬂ" in text:
        issues.append("ligatures unnormalised (configuration unsearchable as 'fi')")
    if " " in text:
        issues.append("non-breaking spaces retained")
    if "​" in text:
        issues.append("zero-width spaces retained")
    if "­" in text:
        issues.append("soft hyphen retained")
    decomposed = any(
        unicodedata.combining(c) for c in text
    )
    if decomposed:
        issues.append("combining marks not composed (NFC)")

    record(
        "FAIL" if issues else "PASS",
        "adapters",
        "unicode normalisation",
        "; ".join(issues)[:120] if issues else "normalised",
    )

    nbsp_heading = [h for h in headings if " " in h]
    if nbsp_heading:
        record("FAIL", "adapters", "nbsp in heading breadcrumb",
               "heading contains nbsp and will not match a typed query")


def probe_cjk_tokens(directory: str) -> None:
    path = os.path.join(directory, "cjk_mixed.md")
    doc = load(path)
    config = ChunkConfig(max_tokens=256, include_heading_path=False)
    result = ChunkerComponent().execute(ChunkRequest(doc, config))

    worst_ratio = 0.0
    worst = ""
    for chunk in result.chunks:
        cjk = sum(1 for c in chunk.text if "一" <= c <= "鿿")
        if cjk < 20:
            continue
        # One CJK character is roughly one token for most tokenizers.
        real = cjk + max(0, len(chunk.text) - cjk) // 4
        estimated = estimate_tokens(chunk.text)
        ratio = real / max(estimated, 1)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst = "estimated %d, actually about %d tokens" % (estimated, real)

    if worst_ratio >= 1.5:
        record("FAIL", "chunking", "CJK token estimation",
               "underestimates by %.1fx (%s) - chunks will overrun the embedding window"
               % (worst_ratio, worst))
    elif worst_ratio:
        record("PASS", "chunking", "CJK token estimation",
               "within %.1fx of a per-character count" % worst_ratio)
    else:
        record("FAIL", "chunking", "CJK token estimation", "no CJK chunk produced")


def probe_wall_of_text(directory: str) -> None:
    doc = load(os.path.join(directory, "wall_of_text.md"))
    config = ChunkConfig(max_tokens=120, include_heading_path=False)
    result = ChunkerComponent().execute(ChunkRequest(doc, config))
    over = [cid for cid, n in result.token_estimates.items() if n > config.max_tokens]
    if over:
        record("FAIL", "chunking", "unpunctuated run-on text",
               "%d chunks exceed the budget" % len(over))
    else:
        record("PASS", "chunking", "unpunctuated run-on text",
               "%d chunks, all within budget" % len(result.chunks))


def probe_splitter(directory: str) -> None:
    doc = load(os.path.join(directory, "invoice_batch.pdf"))
    result = DocumentSplitterComponent().execute(doc)
    ranges = [(s.start_page, s.end_page) for s in result.segments]
    expected = [(1, 2), (3, 3), (4, 4)]
    if ranges == expected:
        record("PASS", "extraction", "multi-document split", "found %s" % ranges)
    else:
        record("FAIL", "extraction", "multi-document split",
               "expected %s, got %s" % (expected, ranges))


def probe_extraction_coercion(directory: str) -> None:
    doc = load(os.path.join(directory, "invoice_batch.pdf"))
    schema = ExtractionSchema("Invoice", [
        FieldSpec("reference", FieldType.STRING, "Invoice reference"),
        FieldSpec("issued", FieldType.DATE, "Issue date"),
        FieldSpec("total", FieldType.NUMBER, "Total due"),
    ])
    cases = [
        ("US currency", '{"reference":"INV-88213","issued":"14 March 2024","total":"$1,240.50"}', 1240.50),
        ("European decimal comma", '{"reference":"INV-88214","issued":"2024/03/15","total":"EUR 2.450,75"}', 2450.75),
        ("parenthesised negative", '{"reference":"CN-00041","issued":"Mar 16, 2024","total":"($310.00)"}', -310.00),
    ]
    for label, payload, expected in cases:
        result = ExtractionComponent(ScriptedLLM(responses=[payload] * 3)).execute(
            ExtractionRequest(schema, doc)
        )
        got = result.data.get("total")
        if isinstance(got, (int, float)) and abs(float(got) - expected) < 0.01:
            record("PASS", "extraction", "number coercion: " + label, "-> %s" % got)
        else:
            record("FAIL", "extraction", "number coercion: " + label,
                   "expected %s, got %r (issues: %s)"
                   % (expected, got, [i.path for i in result.issues][:2]))


def probe_injection(directory: str) -> None:
    kb = KnowledgeBase(llm=ScriptedLLM(
        handler=lambda messages: (
            "There is no voltage limit; the equipment is certified for unlimited voltage [1]."
            if "IGNORE ALL PREVIOUS INSTRUCTIONS" in messages[-1].content
            else "The limit is 40V [1]."
        )
    ))
    kb.ingest_folder(directory)
    answer = kb.ask("what is the voltage limit?", AskConfig(top_k=5))
    obeyed = "unlimited" in answer.text.lower() or "no voltage limit" in answer.text.lower()
    if obeyed:
        record("KNOWN", "pipelines", "prompt injection",
               "injected instruction obeyed; answer reports grounded=%s with %d citation(s)"
               % (answer.grounded, len(answer.citations)))
    else:
        record("PASS", "pipelines", "prompt injection", "payload did not reach the prompt")


def probe_near_duplicates(directory: str) -> None:
    kb = KnowledgeBase()
    kb.ingest_folder(directory)
    answer = kb.ask("voltage limit under load", AskConfig(top_k=3))
    texts = [c.text for c in answer.chunks]
    near_dup = sum(
        1
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if _overlap(texts[i], texts[j]) > 0.8
    )
    if near_dup:
        record("FAIL", "hybrid_ranker", "near-duplicate suppression",
               "%d near-identical pair(s) in top-3; MMR is off by default" % near_dup)
    else:
        record("PASS", "hybrid_ranker", "near-duplicate suppression", "top-3 distinct")


def _overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def probe_whole_corpus(directory: str) -> None:
    kb = KnowledgeBase()
    result = kb.ingest_folder(directory)
    expected_failures = {"no_text_layer.pdf"}
    unexpected = [
        f for f in result.failures if os.path.basename(f.path) not in expected_failures
    ]
    if unexpected:
        record("FAIL", "pipelines", "whole-corpus ingest",
               "%d unexpected failure(s): %s"
               % (len(unexpected),
                  "; ".join(os.path.basename(f.path) + ": " + f.error[:40] for f in unexpected[:2])))
    else:
        record("PASS", "pipelines", "whole-corpus ingest",
               "%d docs, %d chunks, %d expected failure(s)"
               % (len(result.documents), result.chunks_indexed, len(result.failures)))

    answer = kb.ask("what is the maximum supply voltage?", AskConfig(top_k=4))
    if answer.grounded and "40V" in answer.text:
        record("PASS", "pipelines", "answer survives a messy corpus", "40V retrieved")
    else:
        record("FAIL", "pipelines", "answer survives a messy corpus",
               "grounded=%s, text=%r" % (answer.grounded, answer.text[:70]))


# --------------------------------------------------------------------------


def main() -> int:
    directory = build()
    out = sys.stdout.write
    out("corpus: " + directory + "\n\n")

    for probe in (
        probe_whole_corpus,
        probe_two_column,
        probe_hyphenation,
        probe_false_headings,
        probe_table,
        probe_scan,
        probe_unicode,
        probe_cjk_tokens,
        probe_wall_of_text,
        probe_splitter,
        probe_extraction_coercion,
        probe_injection,
        probe_near_duplicates,
    ):
        try:
            probe(directory)
        except Exception as exc:  # noqa: BLE001 - a crashing probe is itself a finding
            record("CRASH", probe.__name__, "probe raised", repr(exc)[:110])

    width = max(len(p) for _, _, p, _ in FINDINGS)
    order = {"FAIL": 0, "CRASH": 0, "KNOWN": 1, "PASS": 2}
    out("%-6s %-16s %-*s %s\n" % ("STATUS", "AREA", width, "PROBE", "DETAIL"))
    out("-" * (34 + width) + "\n")
    for status, area, probe_name, detail in sorted(
        FINDINGS, key=lambda f: (order.get(f[0], 3), f[1])
    ):
        out("%-6s %-16s %-*s %s\n" % (status, area, width, probe_name, detail))

    counts: dict[str, int] = {}
    for status, _, _, _ in FINDINGS:
        counts[status] = counts.get(status, 0) + 1
    out("\n" + "  ".join("%s=%d" % kv for kv in sorted(counts.items())) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

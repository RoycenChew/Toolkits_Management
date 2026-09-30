"""The ten-line demo: a folder of documents in, cited answers out.

Runs fully offline with no API key: HashingEmbedder on CPU, an in-memory vector
store, and SQLite FTS5 for keywords. Conference wifi is a known adversary.

    python examples/quickstart.py                 # uses a built-in sample corpus
    python examples/quickstart.py ./my_documents  # uses yours

To make it a real RAG system, swap two arguments:

    from toolkit.adapters import FastEmbedEmbedder, LiteLLMClient
    kb = KnowledgeBase(embedder=FastEmbedEmbedder(), llm=LiteLLMClient("gpt-4o-mini"))
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.pipelines import AskConfig, KnowledgeBase  # noqa: E402

_SAMPLE = {
    "safety_manual.md": """# Safety Manual

## Voltage

The supply must not exceed 40V. Exceeding this limit voids the warranty and may
damage the controller board beyond repair.

## Grounding

Bond the chassis to earth before energising the circuit. Use a conductor rated
for at least 16 amps.
""",
    "refund_policy.md": """# Refund Policy

## Eligibility

Refunds are available for goods returned within 30 days of delivery, provided the
original packaging is intact.

## Processing

Approved refunds are processed within ten business days to the original payment
method. Quote reference INV-88213 when contacting support.
""",
}


def sample_corpus() -> str:
    directory = tempfile.mkdtemp(prefix="toolkit_sample_")
    for name, body in _SAMPLE.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(body)
    return directory


def main(argv: list[str]) -> int:
    folder = argv[1] if len(argv) > 1 else sample_corpus()
    out = sys.stdout.write

    # --- the ten lines -------------------------------------------------
    kb = KnowledgeBase()
    ingested = kb.ingest_folder(folder)
    # -------------------------------------------------------------------

    out("corpus: " + folder + "\n")
    out(
        "ingested "
        + str(len(ingested.documents))
        + " document(s), "
        + str(ingested.chunks_indexed)
        + " chunks, "
        + str(ingested.embedding_calls)
        + " embedding calls\n"
    )
    for failure in ingested.failures:
        out("  FAILED " + failure.path + ": " + failure.error + "\n")

    for question in (
        "what is the maximum supply voltage?",
        "how long do refunds take?",
        "INV-88213",
        "what is the airspeed velocity of an unladen swallow?",
    ):
        answer = kb.ask(question, AskConfig(top_k=2))
        out("\nQ: " + question + "\n")
        if not answer.grounded:
            out("A: (refused - nothing relevant in the corpus)\n")
            continue
        first = answer.citations[0]
        out("A: " + answer.text.split("\n")[0][:160] + "\n")
        out(
            "   source: "
            + os.path.basename(first.source_uri or first.doc_id)
            + " page "
            + str(first.page)
            + (" | " + " > ".join(first.heading_path) if first.heading_path else "")
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

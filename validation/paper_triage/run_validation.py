"""Real-world validation of the toolkit against an uncurated arXiv corpus.

This is a CONSUMER. It adapts to the toolkit's current signatures and does not
change them. Where an interface makes a job awkward, that is recorded as a
finding rather than patched.

Steps (``--step all`` runs them in order):

  ingest      sequential ingest of the real corpus; every failure named
  provenance  re-open the PDF and verify a chunk's bbox actually holds its text
  tokens      compare the heuristic token counter against a real tokenizer
  eval        EvalRunner over the pre-registered golden set
  ablation    lexical-only vs dense-only vs RRF, measured directly
  dagfanout   drive ingest through `dag` at 50-node fan-out

Nothing here needs an LLM: with ``llm=None`` the KnowledgeBase answers
extractively, so every number below is reproducible at zero cost.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CORPUS = os.path.join(HERE, "corpus")
PDF_DIR = os.path.join(CORPUS, "pdf")
META_PATH = os.path.join(CORPUS, "metadata.json")
GOLDEN_PATH = os.path.join(HERE, "golden.jsonl")
OUT_DIR = os.path.join(HERE, "results")

from toolkit.adapters import (  # noqa: E402
    Bm25sIndex,
    FastEmbedEmbedder,
    PdfPlumberSource,
)
from toolkit.chunking import ChunkConfig, estimate_tokens  # noqa: E402
from toolkit.evaluation import EvalConfig, EvalDataset, EvalRunner  # noqa: E402
from toolkit.hybrid_ranker import (  # noqa: E402
    FusionConfig,
    FusionRequest,
    HybridRankerComponent,
    RankedList,
)
from toolkit.hybrid_ranker.models import RankedItem  # noqa: E402
from toolkit.pipelines import IngestConfig, KnowledgeBase  # noqa: E402

FINDINGS: list[dict] = []
_REPORTED: set[str] = set()


def finding(unit: str, severity: str, text: str) -> None:
    key = unit + "|" + text
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    FINDINGS.append({"unit": unit, "severity": severity, "finding": text})
    print(f"  [{severity.upper():8s}] {unit}: {text}")


def all_chunks(kb: KnowledgeBase) -> list:
    """Enumerate indexed chunks.

    There is no public accessor, so this reaches into `_chunks`. Recorded as a
    finding once: a consumer that wants to audit what was indexed has to use a
    private attribute.
    """
    finding(
        "pipelines",
        "minor",
        "no public accessor for indexed chunks; auditing what was indexed "
        "requires reaching into the private `_chunks` dict",
    )
    store = getattr(kb, "_chunks", None) or {}
    return list(store.values()) if hasattr(store, "values") else list(store)


def chunk_path(chunk) -> str | None:
    """The source file a chunk came from, via its documented metadata key."""
    uri = (chunk.metadata or {}).get("source_uri")
    return str(uri) if uri else None


def build_kb(vector_dir: str) -> KnowledgeBase:
    """Real adapters throughout: real PDF extraction, real embeddings, real BM25.

    InMemoryVectorStore rather than LanceDB: the point of this run is to test
    the toolkit's own algorithms, and an in-memory store keeps a vendor store's
    own quirks out of the measurement. LanceDB is exercised separately by the
    adapter contract tests.
    """
    from toolkit.adapters import InMemoryVectorStore

    return KnowledgeBase(
        embedder=FastEmbedEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=Bm25sIndex(),
        llm=None,  # extractive answers; zero cost, fully deterministic
        sources=[PdfPlumberSource()],
        chunk_config=ChunkConfig(max_tokens=512),
    )


# --------------------------------------------------------------------------- #
# step: ingest
# --------------------------------------------------------------------------- #
def step_ingest(kb: KnowledgeBase) -> dict:
    pdfs = sorted(
        os.path.join(PDF_DIR, n) for n in os.listdir(PDF_DIR) if n.endswith(".pdf")
    )
    print(f"\n=== INGEST: {len(pdfs)} real PDFs ===")
    started = time.time()
    result = kb.ingest(
        pdfs,
        IngestConfig(
            max_workers=4,
            durable_db=os.path.join(OUT_DIR, "ingest.db"),
            skip_failed=True,
        ),
    )
    seconds = time.time() - started

    outcomes = list(result.documents)
    failures = [o for o in outcomes if o.status == "failed"]
    chunk_count = kb.count()

    print(f"  seconds        : {seconds:.1f}")
    print(f"  documents      : {len(outcomes)}")
    print(f"  chunks indexed : {result.chunks_indexed} (kb.count={chunk_count})")
    print(f"  embedding calls: {result.embedding_calls}")
    print(f"  failures       : {len(failures)}")
    for o in failures:
        print(f"    - {os.path.basename(o.path)}: {o.error}")

    by_status: dict[str, int] = {}
    for o in outcomes:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    print(f"  status counts  : {by_status}")

    attempted = len(pdfs)
    failed = len(failures)
    rate = failed / attempted if attempted else 0.0
    if rate > 0.10:
        finding(
            "adapters",
            "major",
            f"{failed}/{attempted} ({rate:.0%}) of real PDFs failed to ingest; "
            "pre-registered threshold was 10%",
        )
    unnamed = [o for o in failures if not o.path or not str(o.error).strip()]
    if unnamed:
        finding(
            "pipelines",
            "critical",
            f"{len(unnamed)} failure(s) reported without naming the file or the reason",
        )

    return {
        "attempted": attempted,
        "documents": len(outcomes),
        "status_counts": by_status,
        "failed": failed,
        "failure_rate": rate,
        "chunks_indexed": result.chunks_indexed,
        "kb_count": chunk_count,
        "embedding_calls": result.embedding_calls,
        "seconds": seconds,
        "failures": [
            {"file": os.path.basename(o.path), "error": str(o.error)} for o in failures
        ],
    }


# --------------------------------------------------------------------------- #
# step: provenance  (the core contract: does bbox actually hold the text?)
# --------------------------------------------------------------------------- #
def step_provenance(kb: KnowledgeBase, sample: int = 20) -> dict:
    import pdfplumber

    print(f"\n=== PROVENANCE: re-opening PDFs to verify {sample} chunk bboxes ===")
    chunks = all_chunks(kb)
    if not chunks:
        return {"checked": 0, "note": "no chunks indexed"}

    random.seed(20261003)
    picked = random.sample(chunks, min(sample, len(chunks)))

    checked = verified = out_of_bounds = text_mismatch = no_bbox = 0
    examples: list[dict] = []

    for ch in picked:
        provs = list(ch.provenances or [])
        if not provs or provs[0].bbox is None:
            no_bbox += 1
            continue
        prov = provs[0]
        path = chunk_path(ch)
        if not path or not os.path.exists(path):
            continue
        checked += 1
        try:
            with pdfplumber.open(path) as pdf:
                if prov.page < 1 or prov.page > len(pdf.pages):
                    out_of_bounds += 1
                    examples.append(
                        {
                            "chunk": ch.chunk_id,
                            "problem": f"page {prov.page} outside 1..{len(pdf.pages)}",
                        }
                    )
                    continue
                page = pdf.pages[prov.page - 1]
                bb = prov.bbox
                if (
                    bb.x0 < -1
                    or bb.y0 < -1
                    or bb.x1 > page.width + 1
                    or bb.y1 > page.height + 1
                    or bb.x1 <= bb.x0
                    or bb.y1 <= bb.y0
                ):
                    out_of_bounds += 1
                    examples.append(
                        {
                            "chunk": ch.chunk_id,
                            "problem": (
                                f"bbox ({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) "
                                f"vs page {page.width:.0f}x{page.height:.0f}"
                            ),
                        }
                    )
                    continue
                crop = page.crop((bb.x0, bb.y0, bb.x1, bb.y1), strict=False)
                inside = (crop.extract_text() or "").split()
                wanted = ch.text.split()
                probe = [w for w in wanted if len(w) > 4][:12]
                if not probe:
                    verified += 1
                    continue
                hits = sum(1 for w in probe if w in inside)
                if hits >= max(1, len(probe) // 3):
                    verified += 1
                else:
                    text_mismatch += 1
                    examples.append(
                        {
                            "chunk": ch.chunk_id,
                            "problem": f"only {hits}/{len(probe)} probe words inside bbox",
                            "page": prov.page,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            examples.append({"chunk": ch.chunk_id, "problem": f"{type(exc).__name__}: {exc}"})

    print(f"  checked        : {checked}")
    print(f"  verified       : {verified}")
    print(f"  out of bounds  : {out_of_bounds}")
    print(f"  text mismatch  : {text_mismatch}")
    print(f"  chunks w/o bbox: {no_bbox}")
    for e in examples[:8]:
        print(f"    - {e}")

    if out_of_bounds:
        finding(
            "chunking/doc_layout",
            "critical",
            f"{out_of_bounds} chunk bbox(es) fall outside the page they name - "
            "provenance is the core contract",
        )
    if text_mismatch:
        finding(
            "chunking/doc_layout",
            "critical",
            f"{text_mismatch} chunk bbox(es) do not contain their own text",
        )

    return {
        "checked": checked,
        "verified": verified,
        "out_of_bounds": out_of_bounds,
        "text_mismatch": text_mismatch,
        "no_bbox": no_bbox,
        "examples": examples[:20],
    }


# --------------------------------------------------------------------------- #
# step: tokens  (heuristic counter vs a real tokenizer)
# --------------------------------------------------------------------------- #
def step_tokens(kb: KnowledgeBase, budget: int = 512) -> dict:
    print(f"\n=== TOKENS: heuristic vs real tokenizer (budget {budget}) ===")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        print(f"  tiktoken unavailable: {exc}")
        return {"skipped": str(exc)}

    chunks = all_chunks(kb)
    if not chunks:
        return {"skipped": "no chunks indexed"}

    over = 0
    worst = 0.0
    ratios = []
    for ch in chunks:
        real = len(enc.encode(ch.text))
        est = estimate_tokens(ch.text)
        if est:
            ratios.append(real / est)
        if real > budget:
            over += 1
            worst = max(worst, real / budget)

    ratios.sort()
    mid = ratios[len(ratios) // 2] if ratios else 0.0
    p95 = ratios[int(len(ratios) * 0.95)] if ratios else 0.0
    print(f"  chunks            : {len(chunks)}")
    print(f"  over budget       : {over} ({over / len(chunks):.1%})")
    print(f"  worst overshoot   : {worst:.2f}x budget")
    print(f"  real/est  median  : {mid:.2f}")
    print(f"  real/est  p95     : {p95:.2f}")

    if over:
        finding(
            "chunking",
            "major",
            f"{over}/{len(chunks)} chunks ({over / len(chunks):.0%}) exceed the "
            f"{budget}-token budget against a real tokenizer, worst {worst:.2f}x; "
            "the heuristic counter is documented but this is its real cost",
        )
    return {
        "chunks": len(chunks),
        "over_budget": over,
        "over_fraction": over / len(chunks),
        "worst_overshoot": worst,
        "ratio_median": mid,
        "ratio_p95": p95,
    }


# --------------------------------------------------------------------------- #
# step: eval
# --------------------------------------------------------------------------- #
def step_eval(kb: KnowledgeBase) -> dict:
    print("\n=== EVAL: pre-registered golden set ===")
    if not os.path.exists(GOLDEN_PATH):
        print(f"  no golden set at {GOLDEN_PATH}")
        return {"skipped": "golden set missing"}
    dataset = EvalDataset.from_jsonl(GOLDEN_PATH)
    print(f"  cases: {len(dataset.cases)}")

    report = EvalRunner(kb).execute(
        dataset, EvalConfig(k_values=(1, 5, 10), top_k=10, evaluate_answers=True)
    )
    for name, value in sorted(report.metrics.items()):
        print(f"  {name:28s} {value:.4f}")

    broken = [c for c in report.cases if not c.relevance or not any(c.relevance)]
    print(f"  cases retrieving nothing relevant: {len(broken)}/{len(report.cases)}")
    for c in broken[:10]:
        print(f"    - {c.case_id}: {c.query[:70]}")

    return {
        "metrics": dict(report.metrics),
        "cases": len(report.cases),
        "zero_relevant": [c.case_id for c in broken],
    }


# --------------------------------------------------------------------------- #
# step: ablation  (hybrid_ranker's central claim, measured directly)
# --------------------------------------------------------------------------- #
def step_ablation(kb: KnowledgeBase) -> dict:
    print("\n=== ABLATION: lexical-only vs dense-only vs RRF ===")
    if not os.path.exists(GOLDEN_PATH):
        return {"skipped": "golden set missing"}
    dataset = EvalDataset.from_jsonl(GOLDEN_PATH)

    note = (
        "EvalConfig exposes no fusion weights, so the fusion claim cannot be "
        "ablated through EvalRunner; measured against hybrid_ranker directly"
    )
    print(f"  note: {note}")
    finding("evaluation", "minor", note)

    indexed = all_chunks(kb)

    def relevant_ids(case) -> set[str]:
        want = set()
        for ch in indexed:
            blob = " ".join(ch.text.split()).lower()
            for snip in case.expected_snippets:
                probe = " ".join(str(snip).split()).lower()
                if probe and probe in blob:
                    want.add(ch.chunk_id)
        return want

    ranker = HybridRankerComponent()
    scores = {"lexical": [], "dense": [], "rrf": []}
    missing_truth = []

    for case in dataset.cases:
        if case.unanswerable:
            continue
        truth = relevant_ids(case)
        if not truth:
            missing_truth.append(case.case_id)
            continue

        lex = kb.lexical_index.search(case.query, 20) if kb.lexical_index else []
        vec = kb.embedder.embed([case.query])[0]
        den = kb.vector_store.search(vec, 20)

        def as_list(source: str, hits) -> RankedList:
            items = []
            for h in hits:
                cid = getattr(h, "chunk_id", None) or (
                    h[0] if isinstance(h, (tuple, list)) else None
                )
                sc = getattr(h, "score", None)
                if sc is None and isinstance(h, (tuple, list)) and len(h) > 1:
                    sc = h[1]
                if cid is not None:
                    items.append(RankedItem(id=str(cid), score=float(sc or 0.0)))
            return RankedList(source=source, items=items)

        lex_list, den_list = as_list("lexical", lex), as_list("dense", den)

        def recall_at(items, want: set[str], k: int = 10) -> float:
            got = {i.id for i in items[:k]}
            return len(got & want) / len(want)

        scores["lexical"].append(recall_at(lex_list.items, truth))
        scores["dense"].append(recall_at(den_list.items, truth))
        fused = ranker.execute(
            FusionRequest(
                query=case.query,
                ranked_lists=[lex_list, den_list],
                config=FusionConfig(top_k=10),
            )
        )
        fitems = getattr(fused, "items", None) or getattr(fused, "ranked", [])
        scores["rrf"].append(
            len({getattr(i, "id", str(i)) for i in fitems[:10]} & truth) / len(truth)
        )

    out = {}
    for k, v in scores.items():
        out[k] = sum(v) / len(v) if v else 0.0
        print(f"  recall@10 {k:8s} {out[k]:.4f}  (n={len(v)})")
    if missing_truth:
        print(f"  cases whose expected snippet is in NO chunk: {len(missing_truth)}")
        for c in missing_truth[:10]:
            print(f"    - {c}")
        finding(
            "chunking/golden-set",
            "major",
            f"{len(missing_truth)} golden snippets appear in no indexed chunk - "
            "either extraction lost the text or the snippet was mis-transcribed",
        )

    if out.get("rrf", 0) < max(out.get("lexical", 0), out.get("dense", 0)):
        finding(
            "hybrid_ranker",
            "major",
            f"RRF ({out['rrf']:.3f}) did not beat the better single retriever "
            f"(lexical {out['lexical']:.3f}, dense {out['dense']:.3f})",
        )
    return {"recall_at_10": out, "no_truth_cases": missing_truth}


# --------------------------------------------------------------------------- #
# step: dag fan-out
# --------------------------------------------------------------------------- #
def step_dagfanout() -> dict:
    print("\n=== DAG FAN-OUT: ingest 1 node per PDF ===")
    from toolkit.dag import DagConfig, DagExecutorComponent, DagRequest, Node

    pdfs = sorted(
        os.path.join(PDF_DIR, n) for n in os.listdir(PDF_DIR) if n.endswith(".pdf")
    )
    kb = build_kb(os.path.join(OUT_DIR, "dag_vectors"))

    def make(path: str):
        def fn(_ctx):
            res = kb.ingest([path], IngestConfig(max_workers=1))
            return {"path": os.path.basename(path), "ok": True, "result": repr(res)[:120]}

        return fn

    nodes = [
        Node(id=f"doc{i:03d}", fn=make(p)) for i, p in enumerate(pdfs)
    ]
    started = time.time()
    result = DagExecutorComponent().execute(
        DagRequest(nodes, run_id="validation-fanout", config=DagConfig(max_workers=4))
    )
    seconds = time.time() - started

    print(f"  seconds   : {seconds:.1f}")
    print(f"  status    : {result.status}")
    print(f"  completed : {len(result.completed)}  failed: {len(result.failed)}  "
          f"skipped: {len(result.skipped)}")
    print(f"  chunks    : {kb.count()}")
    for nid in result.failed[:10]:
        print(f"    - FAILED {nid}: {result.outcomes[nid].error}")

    if result.failed:
        finding(
            "pipelines",
            "major",
            f"{len(result.failed)}/{len(nodes)} documents failed when ingested "
            "concurrently through dag; KnowledgeBase may not be thread-safe",
        )
    return {
        "nodes": len(nodes),
        "completed": len(result.completed),
        "failed": len(result.failed),
        "skipped": len(result.skipped),
        "chunks": kb.count(),
        "seconds": seconds,
        "errors": {n: str(result.outcomes[n].error) for n in result.failed[:20]},
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--step",
        default="all",
        choices=["all", "ingest", "provenance", "tokens", "eval", "ablation", "dagfanout"],
    )
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    report: dict = {"step": args.step, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    kb = None

    def need_kb():
        nonlocal kb
        if kb is None:
            kb = build_kb(os.path.join(OUT_DIR, "vectors"))
            report["ingest"] = step_ingest(kb)
        return kb

    try:
        if args.step in ("all", "ingest"):
            need_kb()
        if args.step in ("all", "provenance"):
            report["provenance"] = step_provenance(need_kb())
        if args.step in ("all", "tokens"):
            report["tokens"] = step_tokens(need_kb())
        if args.step in ("all", "eval"):
            report["eval"] = step_eval(need_kb())
        if args.step in ("all", "ablation"):
            report["ablation"] = step_ablation(need_kb())
        if args.step in ("all", "dagfanout"):
            report["dagfanout"] = step_dagfanout()
    except Exception:
        report["crashed"] = traceback.format_exc()
        print("\n!!! the validation run itself crashed:\n")
        traceback.print_exc()

    report["findings"] = FINDINGS
    out = os.path.join(OUT_DIR, f"report_{args.step}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"\n{len(FINDINGS)} finding(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

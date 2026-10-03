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

from cached_source import CachedDocumentSource  # noqa: E402

from toolkit.adapters import (  # noqa: E402
    Bm25sIndex,
    FastEmbedEmbedder,
    PdfPlumberSource,
)
from toolkit.chunking import ChunkConfig, estimate_tokens  # noqa: E402
from toolkit.doc_layout import WORD_GAP_RATIO  # noqa: E402
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

# Set from --no-parse-cache. The uncached run is the real extraction evidence;
# the cache only makes repeated downstream analysis affordable.
USE_PARSE_CACHE = True


def finding(unit: str, severity: str, text: str) -> None:
    key = unit + "|" + text
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    FINDINGS.append({"unit": unit, "severity": severity, "finding": text})
    print(f"  [{severity.upper():8s}] {unit}: {text}")


def all_chunks(kb: KnowledgeBase) -> list:
    """Enumerate indexed chunks via the public accessor (F7, now fixed).

    This used to reach into the private `_chunks` dict and report a finding for
    doing so. `KnowledgeBase.chunks()` exists because of that finding.
    """
    return list(kb.chunks())


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

    source: object = PdfPlumberSource()
    if USE_PARSE_CACHE:
        source = CachedDocumentSource(source, os.path.join(OUT_DIR, "parse_cache"))

    return KnowledgeBase(
        embedder=FastEmbedEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=Bm25sIndex(),
        llm=None,  # extractive answers; zero cost, fully deterministic
        sources=[source],
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
    replayed = [o for o in outcomes if o.status == "replayed"]
    if replayed and chunk_count == 0:
        finding(
            "pipelines+durable_steps",
            "critical",
            f"ingest reported {len(replayed)} documents 'replayed' and 0 failures, "
            f"yet the index is EMPTY (kb.count()=0). Checkpoint replay restores a "
            f"step's return value but not its side effects, and KnowledgeBase keeps "
            f"chunks in a process-local dict, so resuming in a new process yields a "
            f"silently empty KnowledgeBase that reports success",
        )
    elif replayed:
        finding(
            "pipelines+durable_steps",
            "major",
            f"{len(replayed)} documents were replayed rather than indexed; "
            f"index holds {chunk_count} chunks",
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

    # A chunk may carry SEVERAL provenance regions, possibly across pages. The
    # honest check unions every region and compares the whole chunk against all
    # of them: testing the chunk's full text against only its first bbox would
    # fail any multi-region chunk and report a defect that is not there.
    for ch in picked:
        provs = [p for p in (ch.provenances or []) if p.bbox is not None]
        if not provs:
            no_bbox += 1
            continue
        path = chunk_path(ch)
        if not path or not os.path.exists(path):
            continue
        checked += 1
        try:
            inside: list[str] = []
            bad_page: str | None = None
            bad_box: str | None = None
            with pdfplumber.open(path) as pdf:
                by_page: dict[int, list] = {}
                for p in provs:
                    by_page.setdefault(p.page, []).append(p.bbox)
                for pno, boxes in sorted(by_page.items()):
                    if pno < 1 or pno > len(pdf.pages):
                        bad_page = f"page {pno} outside 1..{len(pdf.pages)}"
                        break
                    page = pdf.pages[pno - 1]
                    x0 = min(b.x0 for b in boxes)
                    y0 = min(b.y0 for b in boxes)
                    x1 = max(b.x1 for b in boxes)
                    y1 = max(b.y1 for b in boxes)
                    if (
                        x0 < -1
                        or y0 < -1
                        or x1 > page.width + 1
                        or y1 > page.height + 1
                        or x1 <= x0
                        or y1 <= y0
                    ):
                        bad_box = (
                            f"union ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) vs page "
                            f"{page.width:.0f}x{page.height:.0f} on p{pno}"
                        )
                        break
                    crop = page.crop(
                        (max(x0, 0), max(y0, 0), min(x1, page.width), min(y1, page.height)),
                        strict=False,
                    )
                    # Extract with the SAME word-gap ratio the toolkit used. With
                    # pdfplumber's default the crop comes back glued, so the
                    # chunk's correctly-spaced words would never match it - which
                    # made provenance look worse after the gluing fix, not better.
                    inside.extend(
                        (
                            crop.extract_text(x_tolerance_ratio=WORD_GAP_RATIO) or ""
                        ).split()
                    )

            if bad_page or bad_box:
                out_of_bounds += 1
                examples.append(
                    {"chunk": ch.chunk_id, "problem": bad_page or bad_box,
                     "regions": len(provs)}
                )
                continue

            # Compare on normalised words: PDF extraction reflows whitespace, and
            # the toolkit dehyphenates, so exact token identity is too strict.
            def norm(words):
                return {
                    "".join(c for c in w.lower() if c.isalnum())
                    for w in words
                }

            have = norm(inside)
            # `ChunkConfig.include_heading_path=True` prepends the heading trail
            # ("Paper Title > Appendix > A Benchmark Data") to the chunk text.
            # Those words come from a different region, often a different page,
            # so probing them tests nothing about this chunk's own bboxes.
            # Drop them, and drop the heading separator, before building a probe.
            body = ch.text
            heading_path = (ch.metadata or {}).get("heading_path") or []
            for heading in heading_path:
                body = body.replace(str(heading), " ")
            heading_words = norm(
                " ".join(str(h) for h in heading_path).split()
            )
            probe = [
                w
                for w in norm(body.split())
                if len(w) > 4 and w not in heading_words
            ][:15]
            if not probe:
                verified += 1
                continue
            hits = sum(1 for w in probe if w in have)
            if hits >= max(1, len(probe) // 2):
                verified += 1
            else:
                text_mismatch += 1
                examples.append(
                    {
                        "chunk": ch.chunk_id,
                        "problem": f"only {hits}/{len(probe)} probe words inside the "
                                   f"union of its {len(provs)} region(s)",
                        "pages": sorted({p.page for p in provs}),
                        "chunk_text_head": " ".join(ch.text.split())[:110],
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
    specials = 0
    for ch in chunks:
        # Real papers contain the literal string "<|endoftext|>", which tiktoken
        # refuses to encode by default. Found the hard way: it crashed this
        # harness mid-run. Treat it as ordinary text and count it.
        if "<|" in ch.text:
            specials += 1
        real = len(enc.encode(ch.text, disallowed_special=()))
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

    if specials:
        finding(
            "corpus-reality",
            "major",
            f"{specials} chunk(s) contain a tokenizer special-token literal such as "
            "'<|endoftext|>'; tiktoken refuses to encode these by default and raised "
            "ValueError mid-run. Any component that counts tokens with a real "
            "tokenizer must pass disallowed_special=()",
        )
    print(f"  chunks w/ '<|'    : {specials}")
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

    # F8, now fixed: EvalConfig carries the fusion weights, so the ablation runs
    # through EvalRunner itself. It used to require calling HybridRankerComponent
    # directly because there was no way to turn a retriever off between runs.
    print("  through EvalRunner, using EvalConfig fusion weights (F8)")
    runner = EvalRunner(kb)
    through_harness = {}
    for label, weights in (
        ("fused", {}),
        ("lexical_only", {"dense_weight": 0.0}),
        ("dense_only", {"lexical_weight": 0.0}),
    ):
        report = runner.execute(
            dataset,
            EvalConfig(k_values=(1, 5, 10), top_k=10, evaluate_answers=False, **weights),
        )
        through_harness[label] = {
            "hit_rate@10": report.metrics["hit_rate@10"],
            "ndcg@10": report.metrics["ndcg@10"],
            "scored_cases": report.metrics["scored_cases"],
            "dataset_errors": report.metrics["dataset_errors"],
        }
        print(
            f"    {label:13s} hit_rate@10 {report.metrics['hit_rate@10']:.4f}"
            f"  ndcg@10 {report.metrics['ndcg@10']:.4f}"
            f"  (scored {report.metrics['scored_cases']:.0f},"
            f" dataset_errors {report.metrics['dataset_errors']:.0f})"
        )

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
    return {
        "recall_at_10": out,
        "no_truth_cases": missing_truth,
        "through_harness": through_harness,
    }


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

    # Chunk counts alone are not evidence: compare document identity. A count
    # gap with the same doc set means chunks were lost; a smaller doc set means
    # whole documents were lost.
    docs = {c.doc_id for c in all_chunks(kb)}
    print(f"  distinct doc_ids: {len(docs)} (expected {len(nodes)})")
    if len(docs) != len(nodes):
        finding(
            "pipelines",
            "critical",
            f"concurrent ingest through dag indexed {len(docs)} distinct documents "
            f"out of {len(nodes)}, while reporting {len(result.completed)} nodes "
            "completed and 0 failed - documents were lost silently",
        )
    return {
        "nodes": len(nodes),
        "completed": len(result.completed),
        "failed": len(result.failed),
        "skipped": len(result.skipped),
        "chunks": kb.count(),
        "distinct_docs": len(docs),
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
    ap.add_argument(
        "--no-parse-cache",
        action="store_true",
        help="re-parse every PDF with pdfplumber (the honest extraction measurement)",
    )
    args = ap.parse_args()
    global USE_PARSE_CACHE
    USE_PARSE_CACHE = not args.no_parse_cache
    os.makedirs(OUT_DIR, exist_ok=True)

    report: dict = {"step": args.step, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    kb = None

    def need_kb():
        nonlocal kb
        if kb is None:
            kb = build_kb(os.path.join(OUT_DIR, "vectors"))
            report["ingest"] = step_ingest(kb)
        return kb

    # Each step is isolated: a crash in one must not cost the evidence from the
    # others. Learned the hard way - a tiktoken ValueError in `tokens` aborted
    # `eval` and `ablation` on the first full run.
    plan = [
        ("ingest", lambda: {"note": "ran as part of kb construction"} if need_kb() else {}),
        ("provenance", lambda: step_provenance(need_kb())),
        ("tokens", lambda: step_tokens(need_kb())),
        ("eval", lambda: step_eval(need_kb())),
        ("ablation", lambda: step_ablation(need_kb())),
        ("dagfanout", step_dagfanout),
    ]
    for name, fn in plan:
        if args.step not in ("all", name):
            continue
        try:
            got = fn()
            if name != "ingest":
                report[name] = got
        except Exception:
            report[name] = {"crashed": traceback.format_exc()}
            print(f"\n!!! step '{name}' crashed; continuing with the rest\n")
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

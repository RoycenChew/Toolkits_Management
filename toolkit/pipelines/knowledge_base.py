"""The two pipelines, and the facade that wires them with working defaults.

`KnowledgeBase` exists so the common case is short enough to type from memory
during a hackathon:

    kb = KnowledgeBase()
    kb.ingest_folder("./pdfs")
    answer = kb.ask("what is the voltage limit?")

Every default is offline and needs no API key: `HashingEmbedder` on CPU,
`InMemoryVectorStore`, SQLite FTS5 for keywords. Conference wifi is a known
adversary, and a demo that needs a hosted embedding endpoint is a demo that dies.
Swap any part by passing it in — the facade only assembles ports, it never
reaches past them.

Without an LLM, `ask` still works and returns the best retrieved passage with its
citations. That extractive mode is genuinely useful, and it means retrieval can be
debugged before a model is ever involved.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from typing import Any

from ..adapters.embedders import HashingEmbedder
from ..adapters.sources import default_sources
from ..adapters.stores import InMemoryVectorStore, SqliteFtsIndex
from ..chunking import ChunkConfig, ChunkerComponent, ChunkRequest
from ..concurrency import MapConfig, embed_all
from ..core.errors import AdapterError, ToolkitError
from ..core.models import Chunk, Document, Message, Provenance, SearchHit, Usage
from ..durable_steps import (
    DurableStepsComponent,
    SqliteCheckpointStore,
    Step,
    WorkflowRequest,
)
from ..guardrails import SYSTEM_PREAMBLE, GuardrailComponent
from ..guardrails.models import ShieldedSource
from ..hybrid_ranker import (
    FusionConfig,
    FusionMethod,
    FusionRequest,
    HybridRankerComponent,
    RankedItem,
    RankedList,
)
from .models import (
    Answer,
    AskConfig,
    Citation,
    DocumentOutcome,
    IngestConfig,
    IngestResult,
    RetrievalTrace,
)

_MARKER = re.compile(r"\[(\d{1,2})\]")

_UNSET: Any = object()
"""Distinguishes 'argument not supplied' from an explicit None.

Without it, `lexical_index=None` cannot mean "run without keyword search",
because None is indistinguishable from the default. Any optional dependency that
a caller might legitimately want to switch *off* needs this.
"""

_STOPWORDS = frozenset(
    """the and for are but not you all any can had her was one our out day get has him his how
    its may new now old see two who boy did man men put say she too use what when where which
    with this that from have been will would should could does doing about into than then them
    they there these those your yours very much many more most some such only own same than
    take takes taking long does do done how why""".split()  # noqa: SIM905 - a wrapped
    # block of words reads far better here than a sixty-element list literal
)

_SYSTEM_PROMPT = (
    "You answer strictly from the numbered sources provided. "
    "Cite every claim with the source number in square brackets, like [1] or [2]. "
    "If the sources do not contain the answer, say so plainly instead of guessing. "
    "Never cite a number that was not shown to you."
)


class KnowledgeBase:
    """Ingest documents, then ask grounded questions about them."""

    def __init__(
        self,
        embedder: Any | None = None,
        vector_store: Any = _UNSET,
        lexical_index: Any = _UNSET,
        llm: Any | None = None,
        reranker: Any | None = None,
        sources: Sequence[Any] | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        self.embedder = embedder or HashingEmbedder(dimension=256)
        # Pass None explicitly to run without that retriever; omit for the default.
        self.vector_store = (
            InMemoryVectorStore() if vector_store is _UNSET else vector_store
        )
        self.lexical_index = SqliteFtsIndex() if lexical_index is _UNSET else lexical_index
        if self.vector_store is None and self.lexical_index is None:
            raise ValueError(
                "at least one retriever is required; pass a vector_store, a"
                " lexical_index, or both"
            )
        self.llm = llm
        self.reranker = reranker
        self.sources = list(sources) if sources else default_sources()
        self.chunk_config = chunk_config or ChunkConfig()
        self._chunker = ChunkerComponent(embedder=self.embedder)
        self._ranker = HybridRankerComponent()
        self._guard = GuardrailComponent()
        self._chunks: dict[str, Chunk] = {}
        self._doc_paths: dict[str, str] = {}
        self._path_docs: dict[str, str] = {}
        """path -> current doc_id. The supersede index; see `_supersede`. In-memory
        only, so a persistent store outliving the process needs this rebuilt or
        persisted — a profile B/C concern, stated in the README."""
        self._index_model_version: str | None = None
        """Local mirror, so an answer can carry full chunk objects and provenance
        even when the vector store only round-trips a text field."""

    # --- ingest -----------------------------------------------------------

    def ingest_folder(
        self,
        folder: str,
        config: IngestConfig | None = None,
        recursive: bool = True,
    ) -> IngestResult:
        paths: list[str] = []
        if not os.path.isdir(folder):
            raise AdapterError("not a directory: " + folder)
        for root, _, names in os.walk(folder):
            for name in sorted(names):
                path = os.path.join(root, name)
                if any(source.supports(path) for source in self.sources):
                    paths.append(path)
            if not recursive:
                break
        return self.ingest(paths, config)

    def ingest(
        self, paths: Sequence[str], config: IngestConfig | None = None
    ) -> IngestResult:
        cfg = config or IngestConfig()
        outcomes: list[DocumentOutcome] = []
        total_chunks = 0
        embedding_calls = 0

        durable = None
        if cfg.durable_db:
            durable = DurableStepsComponent(SqliteCheckpointStore(cfg.durable_db))

        for path in paths:
            try:
                outcome, chunk_count, calls = self._ingest_one(path, cfg, durable)
            except ToolkitError as exc:
                if not cfg.skip_failed:
                    raise
                outcomes.append(
                    DocumentOutcome(path=path, status="failed", error=str(exc))
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a corpus
                if not cfg.skip_failed:
                    raise
                outcomes.append(
                    DocumentOutcome(
                        path=path, status="failed", error=type(exc).__name__ + ": " + str(exc)
                    )
                )
                continue
            outcomes.append(outcome)
            total_chunks += chunk_count
            embedding_calls += calls

        return IngestResult(
            documents=outcomes,
            chunks_indexed=total_chunks,
            embedding_calls=embedding_calls,
        )

    def _ingest_one(
        self, path: str, cfg: IngestConfig, durable: Any | None
    ) -> tuple[DocumentOutcome, int, int]:
        """Ingest one document, optionally as a single durable step.

        The durable unit is the whole document, not each stage inside it. That is
        deliberate: a `Document` is not JSON-serialisable, so checkpointing
        between parse and embed would mean inventing a serialisation for it. The
        promise that matters — a crash mid-corpus costs the document in flight and
        nothing else — holds at this granularity, and the checkpoint stays a small
        honest record of what finished.
        """
        if durable is None:
            return self._do_ingest(path, cfg)

        holder: dict[str, Any] = {}

        def work(_context: Any) -> dict[str, Any]:
            outcome, chunks, calls = self._do_ingest(path, cfg)
            holder["outcome"] = outcome
            return {"doc_id": outcome.doc_id, "chunks": chunks, "pages": outcome.pages,
                    "calls": calls}

        result = durable.execute(
            WorkflowRequest(run_id=self._run_id(path), steps=[Step("ingest", work)])
        )
        payload = result.context.get("ingest", {})
        if "outcome" in holder:
            return holder["outcome"], int(payload.get("chunks", 0)), int(payload.get("calls", 0))

        # Replayed from a previous run: the work did not happen again, and the
        # store was populated then. Report it as replayed rather than pretending
        # to have done it.
        return (
            DocumentOutcome(
                path=path,
                doc_id=str(payload.get("doc_id", "")),
                chunks=int(payload.get("chunks", 0)),
                pages=int(payload.get("pages", 0)),
                status="replayed",
            ),
            0,
            0,
        )

    def _run_id(self, path: str) -> str:
        """Fingerprint the file without reading it: path, size, mtime.

        Content hashing would be more precise but requires loading the file, which
        is most of the work the checkpoint exists to avoid. Size plus mtime catches
        every realistic edit.
        """
        stat = os.stat(path)
        seed = os.path.abspath(path) + "|" + str(stat.st_size) + "|" + str(int(stat.st_mtime))
        return "ingest:" + hashlib.blake2b(seed.encode("utf-8"), digest_size=10).hexdigest()

    def _do_ingest(self, path: str, cfg: IngestConfig) -> tuple[DocumentOutcome, int, int]:
        self._require_matching_embedder()
        document = self._load(path, cfg)
        self._supersede(path, document.doc_id)
        result = self._chunker.execute(ChunkRequest(document, self.chunk_config))
        if not result.chunks:
            # No chunks now, but the document may have had some before. Sweeping
            # with an empty keep-set is the only way an emptied document stops
            # being retrievable.
            self._sweep(document.doc_id, keep=[])
            return DocumentOutcome(path, document.doc_id, 0, document.page_count), 0, 0

        calls = 0
        if self.vector_store is not None:
            texts = [chunk.text for chunk in result.chunks]
            # A CachedEmbedder exposes `.calls` (texts that actually reached the
            # model). Reading it around the batch is how the caller learns what
            # the run really cost; a plain embedder has no counter, so fall back
            # to the pessimistic assumption that every text was computed.
            before = getattr(self.embedder, "calls", None)
            vectors = embed_all(
                self.embedder,
                texts,
                batch_size=cfg.batch_size,
                config=MapConfig(max_workers=cfg.max_workers, fail_fast=True),
            )
            after = getattr(self.embedder, "calls", None)
            calls = (
                (after - before)
                if (before is not None and after is not None)
                else len(texts)
            )
            self.vector_store.upsert(result.chunks, vectors)
        if cfg.lexical and self.lexical_index is not None:
            self.lexical_index.index(result.chunks)

        # Upsert first, sweep second. Chunk ids are `doc_id#index` slots, so a
        # document that now yields five chunks where it previously yielded eight
        # leaves #5..#7 behind — stale, still retrievable, still citable. Doing
        # it in this order means a failure mid-ingest leaves the old version
        # visible rather than nothing at all.
        kept = [chunk.chunk_id for chunk in result.chunks]
        self._sweep(document.doc_id, keep=kept)

        for chunk in result.chunks:
            self._chunks[chunk.chunk_id] = chunk
        for stale in [
            cid
            for cid, chunk in list(self._chunks.items())
            if chunk.doc_id == document.doc_id and cid not in set(kept)
        ]:
            self._chunks.pop(stale, None)
        self._index_model_version = self.embedder.model_version
        self._doc_paths[document.doc_id] = path

        return (
            DocumentOutcome(
                path=path,
                doc_id=document.doc_id,
                chunks=len(result.chunks),
                pages=document.page_count,
            ),
            len(result.chunks),
            calls,
        )

    def _load(self, path: str, cfg: IngestConfig | None = None) -> Document:
        limits = (cfg or IngestConfig()).limits
        for source in self.sources:
            if source.supports(path):
                return source.load(path, limits)
        raise AdapterError("no DocumentSource supports " + os.path.basename(path))

    # --- deletion and consistency ----------------------------------------

    def forget(self, doc_id: str) -> int:
        """Remove a document from every index. Returns rows deleted.

        The retention and right-to-erasure path. Without it a document can be
        ingested but never un-ingested, which blocks any data-retention policy
        and makes a wrongly-ingested or superseded document permanently
        retrievable.
        """
        removed = self._sweep(doc_id, keep=None)
        for chunk_id in [
            cid for cid, chunk in list(self._chunks.items()) if chunk.doc_id == doc_id
        ]:
            self._chunks.pop(chunk_id, None)
        path = self._doc_paths.pop(doc_id, None)
        if path is not None and self._path_docs.get(path) == doc_id:
            self._path_docs.pop(path, None)
        return removed

    def _supersede(self, path: str, new_doc_id: str) -> int:
        """Retire the previous version of whatever lives at this path.

        `doc_id` is a content hash, so an edited document arrives with a *new*
        doc_id and new chunk ids. Sweeping by doc_id therefore cannot find the
        previous version — it is filed under the old hash, and would stay
        retrievable and citable forever.

        The stable identity of "the document at this path" is the path. The
        doc_id identifies a *version* of it. Keeping the path→doc_id mapping is
        what lets a re-ingest supersede rather than accumulate.
        """
        previous = self._path_docs.get(path)
        if previous is None or previous == new_doc_id:
            self._path_docs[path] = new_doc_id
            return 0
        removed = self.forget(previous)
        self._path_docs[path] = new_doc_id
        return removed

    def _sweep(self, doc_id: str, keep: Sequence[str] | None) -> int:
        """Delete a document's rows from both indexes, optionally keeping some.

        Not transactional across two stores. A crash between them leaves the
        document partially visible, which is why the architecture treats
        deletion as a state with a reconciling sweep rather than an operation
        that either happened or did not.
        """
        removed = 0
        for store in (self.vector_store, self.lexical_index):
            if store is None:
                continue
            try:
                removed += store.delete_by_doc(doc_id, keep)
            except AttributeError as exc:
                raise AdapterError(
                    type(store).__name__
                    + " does not implement delete_by_doc; it predates the"
                    + " deletion contract and cannot be used where retention"
                    + " or re-ingest correctness matters"
                ) from exc
        return removed

    def _require_matching_embedder(self) -> None:
        """Refuse to mix vectors from two different embedders.

        Dimension agreement is not identity. Two models of the same size produce
        vectors a store accepts and a search returns, with quality silently
        gone. Refusing at ingest is the only point where the caller can still
        act on it.
        """
        current = self.embedder.model_version
        if self._index_model_version is None or self._index_model_version == current:
            return
        raise AdapterError(
            "this index was built with embedder '"
            + self._index_model_version
            + "' but the configured embedder is '"
            + current
            + "'. Vectors from different models are not comparable even at the"
            + " same dimension. Re-ingest into a fresh index, or restore the"
            + " original embedder."
        )

    @property
    def index_model_version(self) -> str | None:
        """Which embedder produced the vectors currently indexed."""
        return self._index_model_version

    # --- ask --------------------------------------------------------------

    def ask(self, query: str, config: AskConfig | None = None) -> Answer:
        cfg = config or AskConfig()
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        self._require_matching_embedder()

        dense, lexical = self._retrieve(query, cfg)
        if cfg.relevance_gate and not self._is_relevant(query, dense, lexical, cfg):
            dense, lexical = [], []
        fused = self._fuse(query, dense, lexical, cfg)
        candidates = [
            self._chunks[hit.id] for hit in fused.items if hit.id in self._chunks
        ]
        selected = self._distinct(candidates, cfg)[: cfg.top_k]
        scores = {hit.id: hit.score for hit in fused.items}

        trace = RetrievalTrace(
            query=query,
            dense_hits=dense,
            lexical_hits=lexical,
            fused_ids=[item.id for item in fused.items],
            used_ids=[chunk.chunk_id for chunk in selected],
            reranked=fused.reranked,
        )

        if not selected:
            return Answer(
                text=(
                    "The indexed documents do not contain anything relevant to that question."
                    if cfg.refuse_without_context
                    else ""
                ),
                citations=[],
                chunks=[],
                trace=trace,
                grounded=False,
            )

        if self.llm is None:
            return self._extractive(selected, scores, trace)
        return self._generated(query, selected, scores, trace, cfg)

    def _distinct(self, chunks: Sequence[Chunk], cfg: AskConfig) -> list[Chunk]:
        """Drop near-duplicates, preserving rank order.

        Applied before the top_k cut rather than after, so a suppressed
        duplicate is replaced by the next distinct result instead of shrinking
        the context. Comparing against every kept chunk is O(k^2) in the
        selected set, which is fine at k in the tens.
        """
        if cfg.dedupe_threshold >= 1.0:
            return list(chunks)
        kept: list[Chunk] = []
        signatures: list[set[str]] = []
        for chunk in chunks:
            words = set(chunk.text.lower().split())
            if not words:
                continue
            if any(
                len(words & seen) / min(len(words), len(seen)) > cfg.dedupe_threshold
                for seen in signatures
                if seen
            ):
                continue
            kept.append(chunk)
            signatures.append(words)
        return kept

    def _retrieve(
        self, query: str, cfg: AskConfig
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        dense: list[SearchHit] = []
        if self.vector_store is not None and self.vector_store.count():
            vector = self.embedder.embed([query])[0]
            dense = list(self.vector_store.search(vector, cfg.candidates_per_retriever))
        lexical: list[SearchHit] = []
        if self.lexical_index is not None and self.lexical_index.count():
            lexical = list(self.lexical_index.search(query, cfg.candidates_per_retriever))
        return dense, lexical

    def _is_relevant(
        self,
        query: str,
        dense: Sequence[SearchHit],
        lexical: Sequence[SearchHit],
        cfg: AskConfig,
    ) -> bool:
        """Is anything retrieved actually about the question?

        Term overlap is the primary signal because it is the only one that is
        portable: a shared content word is evidence under any backend. The dense
        arm is opt-in via `min_dense_similarity`, since cosine floors do not
        transfer between embedders — see that field's docstring for measurements
        showing an irrelevant query out-scoring a relevant one under the default
        embedder.

        Overlap is checked against both retrievers' hits, not just the lexical
        ones: when no lexical index is configured, the dense hits are all there is.
        """
        if (
            cfg.min_dense_similarity is not None
            and dense
            and dense[0].score >= cfg.min_dense_similarity
        ):
            return True
        terms = self._content_terms(query)
        if not terms:
            # A query of nothing but stopwords carries no testable signal; let it
            # through rather than refusing on a technicality.
            return True
        for hit in list(lexical[:5]) + list(dense[:5]):
            haystack = (hit.text or "").lower()
            if any(term in haystack for term in terms):
                return True
        return False

    def _content_terms(self, query: str) -> list[str]:
        """Query words that carry meaning. Stopwords match everything, so a gate
        built on them would never refuse anything."""
        words = re.findall(r"[\w-]+", query.lower())
        return [w for w in words if len(w) > 2 and w not in _STOPWORDS]

    def _fuse(
        self,
        query: str,
        dense: Sequence[SearchHit],
        lexical: Sequence[SearchHit],
        cfg: AskConfig,
    ) -> Any:
        lists: list[RankedList] = []
        if dense:
            lists.append(
                RankedList(
                    "dense",
                    [RankedItem(h.chunk_id, h.score, {"text": h.text}) for h in dense],
                    weight=cfg.dense_weight,
                )
            )
        if lexical:
            lists.append(
                RankedList(
                    "lexical",
                    [RankedItem(h.chunk_id, h.score, {"text": h.text}) for h in lexical],
                    weight=cfg.lexical_weight,
                )
            )
        return self._ranker.execute(
            FusionRequest(
                query=query,
                ranked_lists=lists,
                config=FusionConfig(
                    method=FusionMethod.RRF,
                    rrf_k=cfg.rrf_k,
                    rerank_budget=cfg.rerank_budget,
                    top_k=max(cfg.top_k, cfg.rerank_budget or cfg.top_k),
                ),
            ),
            reranker=self.reranker,
        )

    # --- answer assembly --------------------------------------------------

    def _citation(self, marker: int, chunk: Chunk, score: float) -> Citation:
        prov = chunk.provenances[0] if chunk.provenances else Provenance(page=1)
        return Citation(
            marker=marker,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            page=prov.page,
            bbox=prov.bbox,
            source_uri=str(chunk.metadata.get("source_uri", "")),
            heading_path=list(chunk.metadata.get("heading_path", [])),
            quote=chunk.text[:400],
            score=score,
        )

    def _extractive(
        self, selected: Sequence[Chunk], scores: dict[str, float], trace: RetrievalTrace
    ) -> Answer:
        """No LLM: return the passages themselves, fully cited.

        Useful on its own, and the right way to debug retrieval — if the correct
        passage is not here, no model was ever going to fix it.
        """
        citations = [
            self._citation(index, chunk, scores.get(chunk.chunk_id, 0.0))
            for index, chunk in enumerate(selected, start=1)
        ]
        body = "\n\n".join(
            "[" + str(c.marker) + "] (page " + str(c.page) + ") " + c.quote
            for c in citations
        )
        return Answer(
            text=body,
            citations=citations,
            chunks=list(selected),
            trace=trace,
            grounded=True,
        )

    def _generated(
        self,
        query: str,
        selected: Sequence[Chunk],
        scores: dict[str, float],
        trace: RetrievalTrace,
        cfg: AskConfig,
    ) -> Answer:
        llm = self.llm
        if llm is None:
            return self._extractive(selected, scores, trace)

        # Shield before the text ever reaches a prompt. Fencing each source and
        # stating that fenced content is data is the layer that still works
        # against a payload no pattern anticipated; neutralisation is the
        # heuristic on top.
        shield = self._guard.shield(
            {str(i): c.text for i, c in enumerate(selected, start=1)},
            cfg.guardrails,
        )
        injection_flags = [f.render() for f in shield.findings]

        if shield.refused:
            return Answer(
                text=(
                    "A retrieved document contains an embedded instruction, so this"
                    " question was not answered."
                ),
                citations=[],
                chunks=list(selected),
                trace=trace,
                grounded=False,
                injection_flags=injection_flags,
            )

        kept = shield.included
        if not kept:
            return Answer(
                text=(
                    "Every relevant passage was withheld because it contained an"
                    " embedded instruction."
                ),
                citations=[],
                chunks=list(selected),
                trace=trace,
                grounded=False,
                injection_flags=injection_flags,
            )

        # Citation markers must keep pointing at the chunks they were numbered
        # for, so excluding a source renumbers the prompt and this map carries
        # the correspondence back. Getting this wrong would attach a real page
        # number to the wrong passage, which is the exact failure the verifier
        # exists to prevent.
        visible: list[Chunk] = []
        for source in kept:
            visible.append(selected[int(source.source_id) - 1])
        renumbered = [
            ShieldedSource(
                source_id=str(position),
                text=self._label(visible[position - 1], source.text),
                original_text=source.original_text,
                findings=source.findings,
            )
            for position, source in enumerate(kept, start=1)
        ]

        context = self._guard.render_context(
            renumbered, cfg.guardrails, cfg.context_char_limit
        )
        system = (
            SYSTEM_PREAMBLE + "\n\n" + _SYSTEM_PROMPT
            if cfg.guardrails.delimit_sources
            else _SYSTEM_PROMPT
        )
        messages = [
            Message("system", system),
            Message("user", "Sources:\n\n" + context + "\n\nQuestion: " + query.strip()),
        ]
        completion = llm.complete(messages, cfg.temperature, cfg.max_tokens)

        policy = self._guard.check_output(
            completion.text, [s.text for s in renumbered], cfg.guardrails
        )
        policy_flags = [f.rule + ": " + f.detail for f in policy.findings]

        if policy.should_refuse:
            # The model echoed injected instructions back. That is strong
            # evidence the attack worked, and returning the answer is worse than
            # refusing it.
            return Answer(
                text=(
                    "The generated answer was withheld because it reproduced an"
                    " instruction embedded in a source document."
                ),
                citations=[],
                chunks=list(visible),
                usage=completion.usage or Usage(),
                trace=trace,
                grounded=False,
                injection_flags=injection_flags,
                policy_flags=policy_flags,
            )

        # Verify every marker against what the model was actually shown. A marker
        # outside that range means the model invented a source, which is the one
        # failure worth surfacing loudly rather than rendering as a citation.
        cited = [int(m) for m in _MARKER.findall(completion.text)]
        valid = [n for n in dict.fromkeys(cited) if 1 <= n <= len(visible)]
        unverified = sorted({n for n in cited if not 1 <= n <= len(visible)})

        citations = [
            self._citation(n, visible[n - 1], scores.get(visible[n - 1].chunk_id, 0.0))
            for n in valid
        ]
        return Answer(
            text=completion.text,
            citations=citations,
            chunks=list(visible),
            usage=completion.usage or Usage(),
            trace=trace,
            grounded=bool(citations),
            injection_flags=injection_flags,
            policy_flags=policy_flags,
            unverified_markers=unverified,
        )

    def _label(self, chunk: Chunk, text: str) -> str:
        prov = chunk.provenances[0] if chunk.provenances else None
        return ("(page " + str(prov.page) + ")\n" + text) if prov else text


    # --- introspection ----------------------------------------------------

    def count(self) -> int:
        return len(self._chunks)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


__all__ = ["KnowledgeBase"]

"""Command line interface: ingest a corpus once, then query it in a second.

Why a CLI at all, when the library is right there: it is the most portable
interface this toolkit can have. Every coding agent can run a shell command,
including the ones that support neither MCP nor the Agent Skills format, and so
can cron, a Makefile, CI, and a person who has not read the API.

It only became worth building once `KnowledgeBase.save`/`load` existed. Before
that, every invocation would have re-parsed the corpus - 38 minutes for 49
papers - which is no interface at all.

    toolkit ingest papers/*.pdf --save papers.kb
    toolkit ask papers.kb "what does the paper say about supersingular surfaces?"
    toolkit eval papers.kb golden.jsonl
    toolkit inspect papers.kb

Every vendor import here is inside a function, never at module level, so
`toolkit --help` works on the stdlib-only install and the CI job that asserts no
vendor SDK reaches `sys.modules` keeps passing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_FAILED = 1

_EMBEDDERS = ("hashing", "fastembed")
_LEXICAL = ("sqlite", "bm25", "none")


def _build_embedder(name: str) -> Any:
    """Construct an embedder by short name, importing the backend lazily."""
    if name == "hashing":
        from .adapters import HashingEmbedder

        return HashingEmbedder()
    if name == "fastembed":
        from .adapters import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise SystemExit("unknown embedder: " + name + " (choose from " + ", ".join(_EMBEDDERS) + ")")


def _embedder_for_saved_index(directory: str, override: str | None) -> Any:
    """Pick the embedder an index was built with, unless told otherwise.

    `save` records the class name. Guessing wrong is not silently wrong - `load`
    re-embeds from the saved text when the version does not match - but it costs
    the embedding time the index was saved to avoid, so default to the right one.
    """
    if override:
        return _build_embedder(override)
    manifest_path = os.path.join(directory, "manifest.json")
    recorded = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                recorded = json.load(handle).get("embedder_class")
        except (OSError, ValueError):
            recorded = None
    mapping = {"HashingEmbedder": "hashing", "FastEmbedEmbedder": "fastembed"}
    return _build_embedder(mapping.get(str(recorded), "hashing"))


def _build_lexical(name: str) -> Any:
    if name == "none":
        return None
    if name == "sqlite":
        from .adapters import SqliteFtsIndex

        return SqliteFtsIndex()
    if name == "bm25":
        from .adapters import Bm25sIndex

        return Bm25sIndex()
    raise SystemExit("unknown lexical index: " + name)


def _expand(paths: list[str]) -> list[str]:
    """Expand directories to the files inside them, keeping order stable.

    Shells on Windows do not glob, so `*.pdf` can arrive literally; a directory
    argument is the portable way to say "everything in here".
    """
    out: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                out.extend(os.path.join(root, name) for name in sorted(names))
        else:
            out.append(path)
    return out


# --------------------------------------------------------------------------- #
def cmd_ingest(args: argparse.Namespace) -> int:
    from .pipelines import IngestConfig, KnowledgeBase

    paths = _expand(args.paths)
    if not paths:
        print("nothing to ingest", file=sys.stderr)
        return EXIT_USAGE

    from .adapters import InMemoryVectorStore
    from .chunking import ChunkConfig

    kb = KnowledgeBase(
        embedder=_build_embedder(args.embedder),
        vector_store=InMemoryVectorStore(),
        lexical_index=_build_lexical(args.lexical),
        chunk_config=ChunkConfig(max_tokens=args.chunk_tokens),
    )

    width = len(str(len(paths)))

    def progress(index: int, total: int, outcome: Any) -> None:
        if args.quiet:
            return
        mark = "ok " if outcome.status in ("ingested", "replayed", "reingested") else "FAIL"
        detail = (
            str(outcome.chunks) + " chunks"
            if outcome.status != "failed"
            else str(outcome.error)[:70]
        )
        print(
            "[%*d/%d] %-4s %-40s %s"
            % (width, index, total, mark, os.path.basename(outcome.path)[:40], detail),
            flush=True,
        )

    result = kb.ingest(
        paths,
        IngestConfig(
            durable_db=args.durable,
            on_document=progress,
            max_workers=args.workers,
        ),
    )
    failed = [d for d in result.documents if d.status == "failed"]
    print()
    print("documents : %d (%d failed)" % (len(result.documents), len(failed)))
    print("chunks    : %d" % kb.count())

    if args.save:
        manifest = kb.save(args.save)
        print("saved     : %s (%d chunks, %d dims)" % (
            args.save, manifest["chunks"], manifest["dimension"]))
    else:
        print("not saved : pass --save DIR to keep this index", file=sys.stderr)

    # A corpus where every document failed is a failure, not a 0-chunk success.
    return EXIT_FAILED if failed and len(failed) == len(result.documents) else EXIT_OK


def cmd_ask(args: argparse.Namespace) -> int:
    from .adapters import InMemoryVectorStore
    from .pipelines import AskConfig, KnowledgeBase

    kb = KnowledgeBase.load(
        args.index,
        embedder=_embedder_for_saved_index(args.index, args.embedder),
        vector_store=InMemoryVectorStore(),
        lexical_index=_build_lexical(args.lexical),
    )
    answer = kb.ask(args.question, AskConfig(top_k=args.top_k))

    if args.json:
        print(json.dumps(
            {
                "question": args.question,
                "answer": answer.text,
                "grounded": answer.grounded,
                "citations": [
                    {
                        "chunk_id": citation.chunk_id,
                        "pages": list(getattr(citation, "pages", []) or []),
                    }
                    for citation in answer.citations
                ],
                "injection_flags": list(answer.injection_flags),
            },
            indent=2,
            ensure_ascii=False,
        ))
        return EXIT_OK if answer.citations else EXIT_FAILED

    print(answer.text)
    if answer.citations:
        print()
        print("citations:")
        for citation in answer.citations:
            chunk = kb.chunk(citation.chunk_id)
            pages = sorted({p.page for p in (chunk.provenances if chunk else [])})
            source = os.path.basename(
                (chunk.metadata or {}).get("source_uri", "") if chunk else ""
            )
            print("  %-28s %-26s pages %s" % (citation.chunk_id, source[:26], pages or "-"))
    if answer.injection_flags:
        print()
        print("injection flags: " + ", ".join(answer.injection_flags), file=sys.stderr)
    # An ungrounded answer is a refusal; say so in the exit code so a script can
    # tell "no answer in this corpus" from "here is an answer".
    return EXIT_OK if answer.citations else EXIT_FAILED


def cmd_eval(args: argparse.Namespace) -> int:
    from .adapters import InMemoryVectorStore
    from .evaluation import EvalConfig, EvalDataset, EvalRunner, diff_reports
    from .pipelines import KnowledgeBase

    kb = KnowledgeBase.load(
        args.index,
        embedder=_embedder_for_saved_index(args.index, args.embedder),
        vector_store=InMemoryVectorStore(),
        lexical_index=_build_lexical(args.lexical),
    )
    dataset = EvalDataset.from_jsonl(args.golden)
    report = EvalRunner(kb).execute(
        dataset,
        EvalConfig(
            k_values=tuple(int(k) for k in args.k.split(",")),
            top_k=args.top_k,
            evaluate_answers=args.answers,
            lexical_weight=args.lexical_weight,
            dense_weight=args.dense_weight,
        ),
    )

    for name in sorted(report.metrics):
        print("  %-28s %.4f" % (name, report.metrics[name]))

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {"metrics": dict(report.metrics), "config": dict(report.config)},
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            handle.write("\n")
        print()
        print("written: " + args.out)

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            previous = json.load(handle)
        diff = diff_reports(previous, report)
        print()
        print(diff.render() if hasattr(diff, "render") else str(diff))
    return EXIT_OK


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest_path = os.path.join(args.index, "manifest.json")
    if not os.path.exists(manifest_path):
        print("no saved index at " + args.index, file=sys.stderr)
        return EXIT_FAILED
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    for key in sorted(manifest):
        print("  %-24s %s" % (key, manifest[key]))

    documents = os.path.join(args.index, "documents.jsonl")
    if os.path.exists(documents):
        print()
        print("documents:")
        with open(documents, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    print("  %-22s %s" % (row["doc_id"], row["path"]))
    return EXIT_OK


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolkit",
        description="Ingest documents once, then query the saved index.",
    )
    sub = parser.add_subparsers(dest="command")

    common = {"embedder": None, "lexical": "sqlite"}

    p = sub.add_parser("ingest", help="parse, chunk, embed and index documents")
    p.add_argument("paths", nargs="+", help="files or directories")
    p.add_argument("--save", help="directory to write the index to")
    p.add_argument("--embedder", choices=_EMBEDDERS, default="hashing")
    p.add_argument("--lexical", choices=_LEXICAL, default=common["lexical"])
    p.add_argument("--chunk-tokens", type=int, default=512, dest="chunk_tokens")
    p.add_argument("--durable", help="checkpoint database, so a crash resumes")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quiet", action="store_true", help="suppress per-document progress")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="answer a question from a saved index")
    p.add_argument("index", help="directory written by `ingest --save`")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")
    p.add_argument("--embedder", choices=_EMBEDDERS, default=None,
                   help="defaults to the one recorded in the index")
    p.add_argument("--lexical", choices=_LEXICAL, default=common["lexical"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="score a saved index against a golden set")
    p.add_argument("index")
    p.add_argument("golden", help="JSONL golden set")
    p.add_argument("--k", default="1,5,10")
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--embedder", choices=_EMBEDDERS, default=None)
    p.add_argument("--lexical", choices=_LEXICAL, default=common["lexical"])
    p.add_argument("--answers", action="store_true", help="also score answer text")
    p.add_argument("--lexical-weight", type=float, default=1.0, dest="lexical_weight")
    p.add_argument("--dense-weight", type=float, default=1.0, dest="dense_weight")
    p.add_argument("--out", help="write metrics JSON here, for use as a later --baseline")
    p.add_argument("--baseline", help="metrics JSON from an earlier run, to diff against")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("inspect", help="describe a saved index")
    p.add_argument("index")
    p.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a CLI reports, it does not traceback
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())

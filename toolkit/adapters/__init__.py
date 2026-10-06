"""Adapters: the only place in the toolkit that may import a vendor SDK.

Every import of an optional backend happens lazily inside a method, so
`import toolkit` works with nothing installed and a missing package produces a
`MissingDependency` naming the extra to install rather than an ImportError at
module load.
"""
from .embedders import FastEmbedEmbedder, HashingEmbedder
from .llms import LiteLLMClient, ScriptedLLM
from .rerankers import CrossEncoderReranker, LexicalOverlapReranker, LLMReranker
from .sources import (
    DoclingSource,
    PdfPlumberSource,
    PlainTextSource,
    TesseractSource,
    default_sources,
    load_document,
)
from .stores import Bm25sIndex, InMemoryVectorStore, LanceDBStore, SqliteFtsIndex

__all__ = [
    "Bm25sIndex",
    "CrossEncoderReranker",
    "DoclingSource",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "LLMReranker",
    "LanceDBStore",
    "LexicalOverlapReranker",
    "LiteLLMClient",
    "PdfPlumberSource",
    "PlainTextSource",
    "ScriptedLLM",
    "SqliteFtsIndex",
    "TesseractSource",
    "default_sources",
    "load_document",
]

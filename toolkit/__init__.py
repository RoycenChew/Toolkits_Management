"""A reusable engineering toolkit: own the algorithms, borrow the backends.

Importing this package pulls in nothing outside the standard library. Optional
backends (Docling, LiteLLM, FastEmbed, LanceDB, bm25s) are imported lazily
inside the adapter that needs them, so a bare install still works end to end via
the stdlib implementations.

Layout:
    core/       shared contracts - Document, Block, Chunk, Provenance, Usage
    ports.py    Protocols the toolkit needs from the outside world
    adapters/   the only place a vendor SDK may be imported
    doc_layout/ reading order, heading inference, page furniture
    hybrid_ranker/    rank fusion, rerank cascade, MMR
    entity_resolution/  blocking, Fellegi-Sunter scoring, clustering
    durable_steps/      crash-resumable multi-step execution

Each component directory remains independently copyable: none of them import
this package, so any one can be lifted into an unrelated project on its own.
"""
from .core import (
    AdapterError,
    BBox,
    Block,
    BlockType,
    Chunk,
    Completion,
    Document,
    Message,
    MissingDependency,
    Provenance,
    RateLimited,
    SearchHit,
    ToolkitError,
    Usage,
    ValidationFailed,
)
from .ports import (
    LLM,
    Cache,
    DocumentSource,
    Embedder,
    LexicalIndex,
    Reranker,
    VectorStore,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # contracts
    "BBox",
    "Block",
    "BlockType",
    "Chunk",
    "Completion",
    "Document",
    "Message",
    "Provenance",
    "SearchHit",
    "Usage",
    # errors
    "AdapterError",
    "MissingDependency",
    "RateLimited",
    "ToolkitError",
    "ValidationFailed",
    # ports
    "Cache",
    "DocumentSource",
    "Embedder",
    "LLM",
    "LexicalIndex",
    "Reranker",
    "VectorStore",
]

"""A caching `DocumentSource`, written against the port rather than around it.

Parsing 212 MB of arXiv PDFs with pdfplumber takes ~38 minutes, and there is no
way to persist or reload a `KnowledgeBase`: chunks live in a private in-memory
dict and no public entry point re-indexes existing chunks. So every analysis
run would otherwise re-parse the whole corpus.

Rather than reach into the toolkit, this implements the `DocumentSource`
protocol and wraps a real source, pickling each parsed `Document` on first
load. That is what the port exists for, and it is the consumer adapting to the
toolkit instead of the toolkit bending to the consumer.

Caveat worth stating: this caches the parse, so it does not re-validate
pdfplumber's behaviour on later runs. The uncached numbers in
results/report_ingest.json are the real extraction evidence; this only makes
the downstream steps affordable.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any

from toolkit.core import Document, ScreeningLimits


class CachedDocumentSource:
    """Wraps a `DocumentSource`, persisting parsed `Document` objects.

    Satisfies the same protocol as `PdfPlumberSource`: `supports(path)` and
    `load(path, limits)`.
    """

    def __init__(self, inner: Any, cache_dir: str) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def supports(self, path: str) -> bool:
        return bool(self.inner.supports(path))

    def _key(self, path: str) -> str:
        stat = os.stat(path)
        raw = f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        return os.path.join(self.cache_dir, digest + ".pickle")

    def load(self, path: str, limits: ScreeningLimits | None = None) -> Document:
        key = self._key(path)
        if os.path.exists(key):
            try:
                with open(key, "rb") as fh:
                    doc = pickle.load(fh)
                self.hits += 1
                return doc
            except Exception:  # noqa: BLE001 - a bad cache entry must not be fatal
                os.remove(key)
        doc = self.inner.load(path, limits) if limits is not None else self.inner.load(path)
        self.misses += 1
        tmp = key + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(doc, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, key)
        return doc

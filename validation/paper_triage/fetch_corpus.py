"""Download a deliberately uncurated arXiv corpus for toolkit validation.

The point is NOT to assemble clean inputs. It is to get documents that nobody
in this repository authored, produced by hundreds of different LaTeX
toolchains, across decades. Old papers (1990s) are included on purpose: they
are the ones with scanned pages, missing text layers and odd encodings.

Politeness: arXiv's API terms ask for a delay between requests. We use 3s and
make one metadata request per category, then one PDF request per paper.

Resumable: already-downloaded PDFs are skipped, so an interrupted run can be
re-run without re-fetching.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
PDF_DIR = os.path.join(CORPUS, "pdf")
META_PATH = os.path.join(CORPUS, "metadata.json")

API = "https://export.arxiv.org/api/query"
DELAY = 3.0
TIMEOUT = 60

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

# (category, how many, sort order). "ascending" reaches the oldest papers on
# arXiv, which is where the badly-digitised documents live.
SLICES = [
    ("cs.IR", 8, "descending"),
    ("cs.CL", 7, "descending"),
    ("cs.CV", 6, "descending"),
    ("stat.ME", 5, "descending"),   # statistics: dense tables
    ("astro-ph", 6, "descending"),  # two-column, heavy tables and figures
    ("math.NT", 5, "descending"),   # notation-heavy
    ("hep-th", 5, "ascending"),     # oldest arXiv papers, often poorly digitised
    ("cs.DS", 4, "ascending"),      # old CS, odd toolchains
    ("eess.AS", 4, "descending"),
]


def fetch_metadata(category: str, count: int, order: str) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": count,
        "sortBy": "submittedDate",
        "sortOrder": order,
    }
    url = API + "?" + urllib.parse.urlencode(params)
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    out = []
    for entry in root.findall(ATOM + "entry"):
        raw_id = entry.findtext(ATOM + "id") or ""
        # https://arxiv.org/abs/2601.01234v1 -> 2601.01234v1
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        if not arxiv_id:
            continue
        pdf_url = ""
        for link in entry.findall(ATOM + "link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href") or ""
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        authors = [
            (a.findtext(ATOM + "name") or "").strip()
            for a in entry.findall(ATOM + "author")
        ]
        out.append(
            {
                "arxiv_id": arxiv_id,
                "slug": arxiv_id.replace("/", "_"),
                "title": " ".join((entry.findtext(ATOM + "title") or "").split()),
                "authors": [a for a in authors if a],
                "abstract": " ".join(
                    (entry.findtext(ATOM + "summary") or "").split()
                ),
                "published": entry.findtext(ATOM + "published") or "",
                "updated": entry.findtext(ATOM + "updated") or "",
                "primary_category": (
                    (entry.find(ARXIV + "primary_category") or ET.Element("x")).get(
                        "term"
                    )
                    or category
                ),
                "query_category": category,
                "pdf_url": pdf_url,
            }
        )
    return out


def download_pdf(record: dict) -> tuple[bool, str]:
    dest = os.path.join(PDF_DIR, record["slug"] + ".pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True, "cached"
    try:
        resp = requests.get(
            record["pdf_url"],
            timeout=TIMEOUT,
            headers={"User-Agent": "toolkit-validation/0.1 (research use)"},
        )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        body = resp.content
        if not body:
            return False, "empty body"
        # Deliberately NOT validating the %PDF header. A file that claims to be
        # a PDF and is not is exactly the kind of input the toolkit must survive,
        # so if arXiv hands us one we keep it.
        with open(dest, "wb") as fh:
            fh.write(body)
        return True, f"{len(body) // 1024} KiB"
    except Exception as exc:  # noqa: BLE001 - report, never hide
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    os.makedirs(PDF_DIR, exist_ok=True)

    records: list[dict] = []
    for category, count, order in SLICES:
        print(f"metadata: {category} ({count}, {order}) ... ", end="", flush=True)
        try:
            got = fetch_metadata(category, count, order)
            records.extend(got)
            print(f"{len(got)} entries")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {type(exc).__name__}: {exc}")
        time.sleep(DELAY)

    # De-duplicate by arxiv_id: categories overlap, which is itself realistic.
    seen: dict[str, dict] = {}
    for rec in records:
        seen.setdefault(rec["arxiv_id"], rec)
    records = sorted(seen.values(), key=lambda r: r["arxiv_id"])
    print(f"\n{len(records)} unique papers\n")

    ok, failed = 0, []
    for i, rec in enumerate(records, 1):
        print(f"[{i:3d}/{len(records)}] {rec['slug']:24s} ", end="", flush=True)
        success, note = download_pdf(rec)
        print(note)
        if success:
            ok += 1
        else:
            failed.append((rec["slug"], note))
        if note != "cached":
            time.sleep(DELAY)

    with open(META_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\ndownloaded/cached: {ok}/{len(records)}")
    if failed:
        print("failures (kept visible on purpose):")
        for slug, note in failed:
            print(f"  {slug}: {note}")
    print(f"metadata -> {META_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

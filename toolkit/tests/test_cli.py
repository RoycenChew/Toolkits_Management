"""The CLI: the most portable interface the toolkit has.

Every coding agent can run a shell command, including those that support
neither MCP nor the Agent Skills format, and so can cron, a Makefile and CI.
These tests drive `cli.main` directly rather than spawning a subprocess, so a
traceback points at the real line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from toolkit.cli import main  # noqa: E402


def _corpus(directory) -> str:
    """Two small Markdown documents. No PDF backend needed."""
    folder = os.path.join(str(directory), "corpus")
    os.makedirs(folder, exist_ok=True)
    for name, body in (
        ("relay.md", "# Relay\n\n## Ratings\n\nThe relay switches 240V at 10A.\n"),
        ("sensor.md", "# Sensor\n\n## Ratings\n\nThe sensor reports degrees Celsius.\n"),
    ):
        with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
    return folder


def test_no_arguments_prints_help_and_does_not_crash(capsys) -> None:
    """A bare invocation is a question, not an error to stack-trace at."""
    code = main([])
    out = capsys.readouterr().out
    assert code == 2
    assert "ingest" in out and "ask" in out


def test_importing_the_cli_pulls_in_no_vendor_sdk() -> None:
    """Every backend import must be inside a function.

    The stdlib-only CI job asserts no vendor SDK reaches `sys.modules`; a
    module-level `import fastembed` in the CLI would break the bare install for
    everyone, including `--help`.

    Runs in a fresh interpreter, because `sys.modules` is process-global: by the
    time the rest of this suite has run, pdfplumber and fastembed are loaded by
    other tests and an in-process assertion would report their imports as the
    CLI's. The first version of this test did exactly that and failed only when
    run with the whole suite.
    """
    probe = (
        "import sys; import toolkit.cli as cli; assert cli.build_parser() is not None; "
        "bad = [m for m in ('fastembed', 'pdfplumber', 'bm25s', 'lancedb', 'litellm',"
        " 'docling') if m in sys.modules]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    finished = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=120,
    )
    assert finished.returncode == 0, finished.stderr[-800:]
    assert "CLEAN" in finished.stdout, finished.stdout.strip()


def test_ingest_then_ask_round_trip(tmp_path, capsys) -> None:
    folder = _corpus(tmp_path)
    index = os.path.join(str(tmp_path), "docs.kb")

    assert main(["ingest", folder, "--save", index, "--quiet"]) == 0
    captured = capsys.readouterr().out
    assert "documents : 2 (0 failed)" in captured
    assert "saved" in captured

    assert main(["ask", index, "what voltage does the relay switch?"]) == 0
    answer = capsys.readouterr().out
    assert "240V" in answer
    assert "citations:" in answer


def test_ask_does_not_need_the_original_documents(tmp_path, capsys) -> None:
    """The whole point of `--save`: parsing happens once.

    Proven by deleting the corpus before asking. If `ask` touched the sources it
    would fail.
    """
    folder = _corpus(tmp_path)
    index = os.path.join(str(tmp_path), "docs.kb")
    main(["ingest", folder, "--save", index, "--quiet"])
    capsys.readouterr()

    for name in os.listdir(folder):
        os.remove(os.path.join(folder, name))

    assert main(["ask", index, "what voltage does the relay switch?"]) == 0
    assert "240V" in capsys.readouterr().out


def test_a_refusal_is_a_distinct_exit_code(tmp_path, capsys) -> None:
    """A script must be able to tell "no answer here" from "here is an answer".

    Both are successful runs of the program; only one found something, so the
    difference has to show up somewhere a shell can read.
    """
    folder = _corpus(tmp_path)
    index = os.path.join(str(tmp_path), "docs.kb")
    main(["ingest", folder, "--save", index, "--quiet"])
    capsys.readouterr()

    answered = main(["ask", index, "what voltage does the relay switch?"])
    capsys.readouterr()
    refused = main(["ask", index, "what is the warranty period in months?"])

    assert answered == 0
    assert refused == 1
    assert "do not contain" in capsys.readouterr().out


def test_ask_json_is_machine_readable(tmp_path, capsys) -> None:
    """`--json` exists so an agent or a script does not have to scrape prose."""
    folder = _corpus(tmp_path)
    index = os.path.join(str(tmp_path), "docs.kb")
    main(["ingest", folder, "--save", index, "--quiet"])
    capsys.readouterr()

    main(["ask", index, "what voltage does the relay switch?", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["question"]
    assert payload["grounded"] is True
    assert payload["citations"]
    assert "240V" in payload["answer"]


def test_inspect_describes_a_saved_index(tmp_path, capsys) -> None:
    folder = _corpus(tmp_path)
    index = os.path.join(str(tmp_path), "docs.kb")
    main(["ingest", folder, "--save", index, "--quiet"])
    capsys.readouterr()

    assert main(["inspect", index]) == 0
    out = capsys.readouterr().out
    assert "chunks" in out
    assert "embedder_model_version" in out
    assert "relay.md" in out


def test_inspect_of_a_missing_index_fails_cleanly(tmp_path, capsys) -> None:
    code = main(["inspect", os.path.join(str(tmp_path), "nope")])
    assert code == 1
    assert "no saved index" in capsys.readouterr().err


def test_eval_scores_a_golden_set_and_can_write_metrics(tmp_path, capsys) -> None:
    folder = _corpus(tmp_path)
    index = os.path.join(str(tmp_path), "docs.kb")
    main(["ingest", folder, "--save", index, "--quiet"])
    capsys.readouterr()

    golden = os.path.join(str(tmp_path), "golden.jsonl")
    with open(golden, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({
            "case_id": "relay",
            "query": "what voltage does the relay switch?",
            "expected_snippets": ["The relay switches 240V at 10A."],
        }) + "\n")
        fh.write(json.dumps({
            "case_id": "unanswerable",
            "query": "what is the warranty period in months?",
            "unanswerable": True,
        }) + "\n")

    metrics_path = os.path.join(str(tmp_path), "metrics.json")
    assert main(["eval", index, golden, "--out", metrics_path]) == 0
    out = capsys.readouterr().out
    assert "hit_rate@1" in out
    assert "scored_cases" in out

    with open(metrics_path, encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["metrics"]["cases"] == 2.0
    assert written["config"]["lexical_weight"] == 1.0


def test_ingest_reports_a_bad_file_without_abandoning_the_corpus(tmp_path, capsys) -> None:
    """One unreadable file must not cost the other documents."""
    folder = _corpus(tmp_path)
    with open(os.path.join(folder, "broken.pdf"), "wb") as fh:
        fh.write(b"not a pdf")

    index = os.path.join(str(tmp_path), "docs.kb")
    code = main(["ingest", folder, "--save", index])
    out = capsys.readouterr().out

    assert code == 0, "a single bad file must not fail the run"
    assert "FAIL" in out
    assert "broken.pdf" in out
    assert "(1 failed)" in out


def test_ingest_without_save_says_the_index_was_discarded(tmp_path, capsys) -> None:
    """Silently throwing away a long ingest would be the cruellest default."""
    folder = _corpus(tmp_path)
    assert main(["ingest", folder, "--quiet"]) == 0
    assert "not saved" in capsys.readouterr().err


def test_ingest_of_nothing_is_a_usage_error(tmp_path, capsys) -> None:
    empty = os.path.join(str(tmp_path), "empty")
    os.makedirs(empty, exist_ok=True)
    assert main(["ingest", empty]) == 2
    assert "nothing to ingest" in capsys.readouterr().err

"""Files run by path, not imported by the package: docs hooks, export and benchmark scripts, the speed module."""

import importlib.util
import itertools
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import speed
from benchmarks.compare import entries

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/results/compare-2026-11-02"


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- benchmarks/speed.py ---

class FakeCleaner:
    def __init__(self):
        self.calls = []

    def process_batch(self, texts, batch_size=None):
        self.calls.append((list(texts), batch_size))
        return [text.upper() for text in texts]


def test_run_batch_keeps_order_across_chunks():
    cleaner = FakeCleaner()
    results, seconds = speed.run_batch(cleaner, ["a", "b", "c"], 2, chunk=2)
    assert results == ["A", "B", "C"] and seconds >= 0
    assert cleaner.calls == [(["a", "b"], 2), (["c"], 2)]


def test_rates_have_no_rate_without_time():
    assert speed.rates(4, 40, 2.0) == {"documents": 4, "chars": 40, "seconds": 2.0,
                                       "docs_per_second": 2.0, "chars_per_second": 20.0}
    assert speed.rates(4, 40, 0.0)["docs_per_second"] is None


def test_measure_warms_up_then_reports_each_script_and_the_total(monkeypatch):
    clock = itertools.count()
    monkeypatch.setattr(speed.time, "perf_counter", lambda: next(clock))
    cleaner = FakeCleaner()
    records = [{"lang": "nl", "text": "hallo"}, {"lang": "ru", "text": "привет"}, {"lang": "nl", "text": "dag"}]
    result = speed.measure(cleaner, records, 3)
    assert cleaner.calls[0] == (["hallo", "привет", "dag"], 3)  # the warmup is not timed into a group
    assert set(result["by_script"]) == {"nl (LATIN)", "ru (CYRILLIC)"}
    assert result["by_script"]["nl (LATIN)"]["documents"] == 2 and result["by_script"]["nl (LATIN)"]["chars"] == 8
    assert result["total"]["documents"] == 3 and result["total"]["seconds"] == 2.0


def test_run_returns_none_when_a_command_is_missing_or_fails():
    assert speed._run([sys.executable, "-c", "print(' x ')"]) == "x"
    assert speed._run([sys.executable, "-c", "raise SystemExit(1)"]) is None
    assert speed._run(["no-such-command-here"]) is None


def test_environment_records_the_run(monkeypatch):
    monkeypatch.setattr(speed, "_run", lambda cmd: " M " if cmd[-1] == "--porcelain" else None)
    env = speed.environment()
    assert env["git_dirty"] is True and env["gpus"] == [] and env["git_commit"] is None
    assert env["python"] and env["packages"]["numpy"]
    monkeypatch.setattr(speed, "_run", lambda cmd: None)
    assert speed.environment()["git_dirty"] is None


# --- docs/hooks.py ---

@pytest.fixture
def hooks(monkeypatch):
    utils = SimpleNamespace(get_relative_url=lambda url, other: f"rel:{url}|{other}")
    monkeypatch.setitem(sys.modules, "mkdocs", SimpleNamespace(utils=utils))
    monkeypatch.setitem(sys.modules, "mkdocs.utils", utils)
    return load_script("docs/hooks.py")


def _content(hooks, path):
    files = SimpleNamespace(get_file_from_path=lambda src: SimpleNamespace(url=f"{src}/") if src != "missing.md" else None)
    html = f'<a href="https://github.com/rhnfzl/SqueakyCleanText/{path}">x</a>'
    return hooks.on_page_content(html, SimpleNamespace(url="page/"), files)


def test_hooks_keep_readers_on_the_site_for_known_pages(hooks):
    assert _content(hooks, "blob/main/CHANGELOG.md") == '<a href="rel:changelog.md/|page/">x</a>'
    assert _content(hooks, "tree/main/benchmarks#run") == '<a href="rel:benchmarks.md/|page/#run">x</a>'
    assert _content(hooks, "blob/main/docs/usage.md") == '<a href="rel:usage.md/|page/">x</a>'


def test_hooks_leave_other_links_alone(hooks):
    for path in ("blob/main/sct/sct.py", "blob/main/docs/missing.md", "issues/1"):
        original = f'<a href="https://github.com/rhnfzl/SqueakyCleanText/{path}">x</a>'
        assert _content(hooks, path) == original


def test_hooks_edit_url_comes_from_page_meta(hooks):
    page = SimpleNamespace(meta={"edit_url": "https://example.test/README.md"}, edit_url="old")
    assert hooks.on_page_markdown("text", page) == "text" and page.edit_url == "https://example.test/README.md"
    page = SimpleNamespace(meta={}, edit_url="old")
    assert hooks.on_page_markdown("text", page) == "text" and page.edit_url == "old"


# --- scripts/benchmark_models.py ---

def test_benchmark_models_writes_the_report_and_creates_its_directory(tmp_path, monkeypatch):
    script = load_script("scripts/benchmark_models.py")
    key = script.MODEL_CANDIDATES[0].key
    seen = {}

    def run(candidate, examples, cleaner):
        seen.update(key=candidate.key, examples=examples, cleaner=cleaner)
        return SimpleNamespace(to_dict=lambda: {"b": 1, "a": 2})

    monkeypatch.setattr(script, "candidate_config", lambda candidate: f"cfg:{candidate.key}")
    monkeypatch.setattr(script, "TextCleaner", lambda cfg, include_entities: (cfg, include_entities))
    monkeypatch.setattr(script, "load_evaluation_jsonl", lambda path: f"examples:{path}")
    monkeypatch.setattr(script, "run_model_benchmark", run)
    output = tmp_path / "new" / "out.json"
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "d.jsonl", "--candidate", key, "--output", str(output)])
    script.main()
    assert seen == {"key": key, "examples": "examples:d.jsonl", "cleaner": (f"cfg:{key}", True)}
    assert output.read_text() == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_benchmark_models_rejects_an_unknown_candidate(monkeypatch):
    script = load_script("scripts/benchmark_models.py")
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "d", "--candidate", "nope", "--output", "o"])
    with pytest.raises(SystemExit) as stop:
        script.main()
    assert stop.value.code == 2


# --- scripts/export_onnx_models.py ---

@pytest.fixture
def exporter():
    return load_script("scripts/export_onnx_models.py")


def test_export_passes_the_model_flags_to_optimum(exporter, tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(exporter.subprocess, "run", lambda cmd, check: commands.append(cmd))
    exporter.export_model("MatteoFasulo/ModernBERT-base-NER", tmp_path)
    exporter.export_model("dslim/bert-base-NER", tmp_path)
    assert commands[0][-3:] == ["--device", "cpu", str(tmp_path)]
    assert "--device" not in commands[1] and commands[1][-1] == str(tmp_path)


def test_upload_sends_only_files_that_exist(exporter, tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_text("m")
    uploads = []
    api = SimpleNamespace(create_repo=lambda *a, **k: None,
                          upload_file=lambda **k: uploads.append((k["path_in_repo"], k["repo_id"])))
    monkeypatch.setattr(exporter, "HfApi", lambda: api)
    exporter.upload_model(tmp_path, "me/repo")
    assert uploads == [("model.onnx", "me/repo")]


def test_dry_run_touches_no_hub(exporter, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exporter, "HfApi", lambda: pytest.fail("no hub call in a dry run"))
    exporter.upload_model(tmp_path, "me/repo", dry_run=True)
    assert "DRY RUN" in capsys.readouterr().out


def test_main_rejects_an_unknown_model(exporter, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--models", "no/such"])
    with pytest.raises(SystemExit) as stop:
        exporter.main()
    assert stop.value.code == 1


def test_main_keeps_going_after_a_failure_and_cleans_up(exporter, tmp_path, monkeypatch, capsys):
    exported, uploaded = [], []

    def export(source, output_dir):
        output_dir.mkdir()
        exported.append(output_dir)
        if source == "dslim/bert-base-NER":
            raise RuntimeError("no disk")

    monkeypatch.setattr(exporter, "export_model", export)
    monkeypatch.setattr(exporter, "upload_model", lambda path, repo, dry_run: uploaded.append((repo, dry_run)))
    monkeypatch.setattr(sys, "argv", ["x", "--dry-run", "--output-base", str(tmp_path), "--models",
                                      "dslim/bert-base-NER", "Babelscape/wikineural-multilingual-ner"])
    with pytest.raises(SystemExit, match="bert-base-NER"):
        exporter.main()
    assert uploaded == [("rhnfzl/wikineural-multilingual-ner-onnx", True)]
    assert "FAILED: no disk" in capsys.readouterr().out
    assert len(exported) == 2 and not any(path.exists() for path in exported)


# --- scripts/diagrams/draw_diagrams.py ---

@pytest.fixture
def diagrams():
    return load_script("scripts/diagrams/draw_diagrams.py")


@pytest.mark.parametrize("build", ["overview", "flow", "pii_modes"])
def test_diagrams_match_the_committed_files_and_pass_their_self_check(diagrams, tmp_path, build):
    diagram = getattr(diagrams, build)()
    diagram.write(tmp_path)
    assert (tmp_path / f"{diagram.name}.svg").read_text() == (ROOT / "resources" / f"{diagram.name}.svg").read_text()
    drawing = json.loads((tmp_path / f"{diagram.name}.excalidraw").read_text())
    ids = [element["id"] for element in drawing["elements"]]
    assert len(ids) == len(set(ids))


def test_diagram_check_rejects_overflow_tiny_text_and_off_canvas(diagrams):
    wide = diagrams.Diagram("d", 100, "t")
    wide.rect(0, 0, 100, 50, "grey")
    wide.text(5, 30, "this text is far wider than the box that holds it")
    tiny = diagrams.Diagram("d", 100, "t")
    tiny.text(5, 30, "x", size=10)
    off = diagrams.Diagram("d", 100, "t")
    off.rect(0, 0, 50, 500, "grey")
    for diagram, message in ((wide, "overflows"), (tiny, "too small"), (off, "off canvas")):
        with pytest.raises(ValueError, match=message):
            diagram.check()


def test_diagram_svg_escapes_text_and_marks_dashed_and_headless_lines(diagrams):
    diagram = diagrams.Diagram("d", 100, "a<b")
    diagram.text(5, 30, "<PERSON>")
    diagram.arrow((0, 0), (10, 0), dashed=True, head=False)
    svg = diagram.svg()
    assert "<title>a&lt;b</title>" in svg and "&lt;PERSON&gt;" in svg
    line = next(row for row in svg.splitlines() if row.startswith("<polyline"))
    assert 'stroke-dasharray="6 4"' in line and "marker-end" not in line


# --- benchmarks/results/compare-2026-11-02/make_comparison.py ---

def test_make_comparison_builds_the_report_from_the_committed_bundle(monkeypatch, capsys):
    tail = BUNDLE / "comparison-tail.md"
    monkeypatch.setattr(sys, "argv", ["make_comparison.py", str(BUNDLE), str(tail)])
    runpy.run_path(str(BUNDLE / "make_comparison.py"), run_name="__main__")
    report = capsys.readouterr().out
    committed = (BUNDLE / "COMPARISON.md").read_text(encoding="utf-8")
    assert report.splitlines()[0] == committed.splitlines()[0]
    assert "COMPLETION_DEFAULT" not in report  # the tail's placeholder is filled from the bundle
    assert tail.read_text(encoding="utf-8").strip().splitlines()[-1] in report
    assert all(name in report for name in entries.ENTRIES)

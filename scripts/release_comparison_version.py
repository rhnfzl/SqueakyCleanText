"""Run one installed SCT version with the frozen release benchmark scorer."""

import ast
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def literal_assignment(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal {name} in {path}")


def configure(harness):
    import sct
    import sct.config
    from sct.sct import TextCleaner
    from sct.utils import constants

    # Both runtimes use the exact same evaluator, not each package's historical copy.
    load_module("sct.evaluation", harness / "support/evaluation.py")
    load_module("sct.utils.atomic_file", harness / "support/atomic_file.py")
    legacy = sct.__version__ == "0.7.1"
    if legacy:
        constants.LEGACY_TYPE_NAMES = literal_assignment(harness / "support/constants.py", "LEGACY_TYPE_NAMES")
        preset = types.ModuleType("sct.presets")
        preset.PRESETS = {"training": literal_assignment(harness / "support/presets.py", "_TRAINING")}
        sys.modules["sct.presets"] = preset
        from sct.utils.process_result import ProcessResult

        original_result = ProcessResult.__init__

        def result_init(self, lm_text, stat_text, language, metadata=None):
            original_result(self, lm_text, stat_text, language, {} if metadata is None else metadata)

        ProcessResult.__init__ = result_init
        original_batch = TextCleaner.process_batch

        def batch(self, texts, max_workers=4, **kwargs):
            return original_batch(self, texts, batch_size=max_workers, **kwargs)

        TextCleaner.process_batch = batch
    sys.path.insert(0, str(harness))
    from benchmarks import run, score

    if legacy:
        import onnxruntime as ort

        config_class = sct.config.TextCleanerConfig
        fields = {f.name for f in dataclasses.fields(config_class)}
        original_options = ort.SessionOptions
        threads = {"n": 16}

        def session_options():
            opts = original_options()
            opts.intra_op_num_threads = threads["n"]
            return opts

        ort.SessionOptions = session_options

        def config(**kwargs):
            if kwargs.get("ner_device") not in (None, "cpu"):
                raise ValueError("The legacy comparison is CPU only")
            threads["n"] = kwargs.get("onnx_intra_op_threads", threads["n"])
            return config_class(**{k: v for k, v in kwargs.items() if k in fields})

        run.TextCleanerConfig = config

        def cleanup(view, offsets, cfg):
            # Mirror 0.7.1's actual cleanup; its quotes and unlimited brackets differ from the release.
            if cfg.check_remove_isolated_letters:
                view, offsets = score._drop(view, offsets, constants.ISOLATED_LETTERS_REGEX)
            if cfg.check_remove_isolated_special_symbols:
                for enabled, pattern in (
                    (cfg.check_remove_bracket_content, constants.BRACKET_CONTENT_REGEX),
                    (cfg.check_remove_brace_content, constants.BRACE_CONTENT_REGEX),
                    (True, constants.ISOLATED_SPECIAL_SYMBOLS_REGEX),
                    (True, constants.ISOLATED_QUOTES_REGEX),
                ):
                    if enabled:
                        view, offsets = score._drop(view, offsets, pattern)
            return view, offsets

        score._sct_cleanup = cleanup
        original_lid = run._run_lid_suite

        class LidProxy:
            def __init__(self, cleaner):
                self.cleaner = cleaner

            def _detect_language(self, text):
                return self.cleaner._detect_language(text), None, "detected", None

        def lid(cleaners, records, chars):
            original_get = cleaners.get
            cleaners.get = lambda *a, **kw: LidProxy(original_get(*a, **kw))
            try:
                return original_lid(cleaners, records, chars)
            finally:
                cleaners.get = original_get

        run._run_lid_suite = lid

    original_get_dataset = run.datasets.get
    hashes = {}

    def get_dataset(name):
        module = original_get_dataset(name)

        def load(*args, **kwargs):
            records = module.load(*args, **kwargs)
            hashes[name] = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            return records

        return types.SimpleNamespace(**{**vars(module), "load": load})

    run.datasets.get = get_dataset
    original_run_dataset = run.run_dataset

    def run_dataset(name, *args, **kwargs):
        result = original_run_dataset(name, *args, **kwargs)
        result["sample_sha256"] = hashes[name]
        result.pop("sample_ids", None)  # Public results contain aggregate scores and a sample hash only.
        return result

    run.run_dataset = run_dataset
    return run


if __name__ == "__main__":
    benchmark = configure(Path(sys.argv[1]).resolve())
    benchmark.main(sys.argv[2:])

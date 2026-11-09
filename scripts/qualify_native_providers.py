"""Exercise one installed provider offline against explicitly prepared model artifacts."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', required=True, choices=(
        'onnx', 'torch', 'gliner', 'ensemble_onnx', 'ensemble_torch', 'presidio_gliner', 'gliner2', 'classify'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new result path to preserve previous qualification evidence')
    artifacts = json.loads(args.artifacts.read_text())
    required = {
        'onnx': ('onnx',), 'torch': ('torch',), 'gliner': ('gliner', 'gliner_parent'),
        'ensemble_onnx': ('onnx', 'gliner', 'gliner_parent'),
        'ensemble_torch': ('torch', 'gliner', 'gliner_parent'),
        'presidio_gliner': ('gliner', 'gliner_parent', 'spacy'), 'gliner2': ('gliner2',),
        'classify': ('classify',),
    }[args.provider]
    for key in required:
        artifact = artifacts[key]
        for relative, expected in artifact['files'].items():
            assert digest(Path(artifact['snapshot']) / relative) == expected, (key, relative)

    import torch
    import onnxruntime
    from sct import DetectionContext, DetectorRegistry, DetectorSpec, TextCleaner, TextCleanerConfig
    from sct.config import LANG_KEYS
    from sct.detectors_builtin import BuiltinNERDetector
    from sct.utils.ner import GeneralNER

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    root = Path(__file__).resolve().parents[1]
    record = {
        'provider': args.provider, 'status': 'started', 'executable': sys.executable, 'python': sys.version,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'artifacts': {key: artifacts[key] for key in required},
        'source_sha256': {str(path.relative_to(root)): digest(path) for path in sorted((root / 'sct').rglob('*.py'))},
        'probe_sha256': digest(__file__), 'runtime': {},
        'scope': 'Synthetic functional inference, not accuracy or production qualification',
    }
    for distribution, module in (
        ('torch', 'torch'), ('transformers', 'transformers'), ('onnxruntime', 'onnxruntime'),
        ('gliner', 'gliner'), ('gliner2', 'gliner2'), ('gliclass', 'gliclass'),
        ('presidio-analyzer', 'presidio_analyzer'), ('spacy', 'spacy'),
    ):
        origin = importlib.util.find_spec(module).origin
        assert Path(origin).is_relative_to(Path(sys.prefix)), origin
        record['runtime'][distribution] = {'version': importlib.metadata.version(distribution), 'path': origin}

    network, forwards, onnx_calls = [], Counter(), []
    original_connect, original_run = socket.socket.connect, onnxruntime.InferenceSession.run
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            network.append(1)
            raise RuntimeError('Native qualification forbids network access')
        return original_connect(sock, address)
    def counted_run(session, *positional, **keywords):
        onnx_calls.append(1)
        return original_run(session, *positional, **keywords)
    def forwarded(module, inputs, output):
        forwards[type(module).__module__ + '.' + type(module).__qualname__] += 1
    socket.socket.connect = offline
    onnxruntime.InferenceSession.run = counted_run
    hook = torch.nn.modules.module.register_module_forward_hook(forwarded)
    try:
        options = dict(language='ENGLISH', check_detect_language=False, pii_processing='unified',
                       allow_network_detectors=True, ner_device='cpu')
        if args.provider == 'classify':
            artifact = artifacts['classify']
            options.update(check_ner_process=False, check_classify_document=True,
                gliclass_model=artifact['snapshot'], gliclass_revision=artifact['revision'],
                gliclass_labels=('technology', 'sports', 'science'), gliclass_threshold=0.0)
            cleaner = TextCleaner(TextCleanerConfig.from_preset('training', **options))
            result = cleaner.process(
                'The new computer chip runs artificial intelligence models. Contact john.smith@example.org.')
            classes = result.metadata['classes']
            assert classes and all(math.isfinite(row['score']) and 0 <= row['score'] <= 1 for row in classes)
            assert 'john.smith@example.org' not in result.lm_text
            assert forwards
            record.update(classes=classes, contact_protected=True)
        else:
            backend = 'gliner' if args.provider == 'gliner2' else args.provider
            options.update(ner_backend=backend, ner_confidence_threshold=.3)
            if backend in ('onnx', 'ensemble_onnx'):
                artifact = artifacts['onnx']
                options.update(ner_models={key: artifact['repo'] for key in LANG_KEYS},
                               ner_model_revisions={key: artifact['revision'] for key in LANG_KEYS})
            if backend in ('torch', 'ensemble_torch'):
                artifact = artifacts['torch']
                options.update(torch_ner_models={key: artifact['snapshot'] for key in LANG_KEYS},
                               torch_ner_model_revisions={key: artifact['revision'] for key in LANG_KEYS})
            if backend in ('gliner', 'ensemble_onnx', 'ensemble_torch', 'presidio_gliner'):
                artifact = artifacts['gliner2' if args.provider == 'gliner2' else 'gliner']
                model = artifact['repo'] if args.provider == 'gliner2' else artifact['snapshot']
                options.update(gliner_model=model, gliner_revision=artifact['revision'],
                               gliner_variant='gliner2' if args.provider == 'gliner2' else 'gliner',
                               gliner_onnx=False, gliner_threshold=.3)
            config = TextCleanerConfig.from_preset('training', **options)
            ner = GeneralNER(config)
            text = '😀 John Smith works for Microsoft in London. Email john.smith@example.org.'
            hits = ner.analyze(text, language='ENGLISH', context=DetectionContext())
            assert hits and any(hit.entity_type == 'PERSON' for hit in hits), 'No PERSON finding in the positive control'
            assert all(0 <= hit.start < hit.end <= len(text) and hit.coordinate_space == 'canonical' for hit in hits)
            assert all(hit.score is None or type(hit.score) is float and math.isfinite(hit.score)
                       and 0 <= hit.score <= 1 for hit in hits)
            registry = DetectorRegistry()
            registry.register('qualified_native', lambda *_: BuiltinNERDetector(ner))
            from dataclasses import replace
            specs = (DetectorSpec('builtin_regex'), DetectorSpec('qualified_native'))
            cleaner = TextCleaner(replace(config, detector_specs=specs), detector_registry=registry)
            result = cleaner.process(text)
            assert 'John Smith' not in result.lm_text and 'john.smith@example.org' not in result.lm_text
            assert 'John Smith' not in json.dumps(result.metadata['findings_v2'])
            if backend in ('onnx', 'ensemble_onnx'):
                assert onnx_calls
            if backend != 'onnx':
                assert forwards
            record.update(canonical_findings=[asdict(hit) for hit in hits],
                          unified_findings=result.metadata['findings_v2'], person_protected=True, contact_protected=True,
                          provenance=ner.provenance)
            if backend == 'presidio_gliner':
                import en_core_web_lg
                assert Path(en_core_web_lg.__file__).is_relative_to(Path(sys.prefix))
                record['nlp_runtime'] = {'version': en_core_web_lg.__version__, 'path': en_core_web_lg.__file__,
                    'recognizers': sorted(type(item).__name__ for item in ner._analyzer.registry.recognizers)}
        assert not network
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', exception_type=type(error).__name__, message=str(error)[:2000])
        raise
    finally:
        hook.remove()
        socket.socket.connect = original_connect
        onnxruntime.InferenceSession.run = original_run
        record.update(network_attempts=len(network), onnx_session_runs=len(onnx_calls),
                      torch_module_forward_counts=dict(forwards))
        args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

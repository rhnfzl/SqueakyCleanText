"""Application registration validates data and never imports request names."""
import dataclasses
import json
import pickle

import pytest

from sct.config import TextCleanerConfig


def test_spec_freezes_nested_options_and_roundtrips():
    from sct.detectors import DetectorSpec
    options = {'nested': [{'limit': 3}]}
    spec = DetectorSpec('example', options)
    options['nested'][0]['limit'] = 9
    assert spec.options['nested'][0]['limit'] == 3
    with pytest.raises(TypeError):
        spec.options['nested'][0]['limit'] = 8
    assert DetectorSpec(**json.loads(json.dumps(dataclasses.asdict(spec)))) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec  # noqa: S301


@pytest.mark.parametrize('value', [float('nan'), float('inf'), object(), {1: 'bad'}])
def test_spec_rejects_non_json_values(value):
    from sct.detectors import DetectorSpec
    with pytest.raises((TypeError, ValueError)):
        DetectorSpec('example', {'value': value})


def test_registry_duplicate_unknown_freeze_and_lazy_creation():
    from sct.detectors import DetectorRegistry, DetectorSpec
    registry = DetectorRegistry()
    registry.register('example', lambda cfg, options: ('created', options['value']))
    with pytest.raises(ValueError):
        registry.register('example', lambda cfg, options: None)
    with pytest.raises(ValueError):
        registry.create(DetectorSpec('os.system', {}), TextCleanerConfig(check_ner_process=False))
    registry.freeze()
    with pytest.raises(ValueError):
        registry.register('later', lambda cfg, options: None)


def test_cleaner_freezes_registry_and_rejects_unimplemented_execution():
    from sct import TextCleaner
    from sct.detectors import DetectorRegistry, DetectorSpec
    registry = DetectorRegistry()
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False), detector_registry=registry)
    assert cleaner.process('hello').lm_text
    with pytest.raises(ValueError):
        registry.register('later', lambda cfg, options: None)
    with pytest.raises(ValueError, match='detector_specs'):
        TextCleaner(TextCleanerConfig(check_ner_process=False, detector_specs=(DetectorSpec('builtin_regex', {}),)))


def test_config_roundtrip_and_explicit_cleaner_config():
    from sct import TextCleaner
    from sct.detectors import DetectorSpec
    spec = DetectorSpec('builtin_regex', {})
    cfg = TextCleanerConfig(check_ner_process=False, detector_specs=(spec,))
    serialized = json.dumps({"detector_specs": cfg._plain_state()["detector_specs"]})
    assert TextCleanerConfig(**json.loads(serialized)).detector_specs == (spec,)
    configured = dataclasses.replace(cfg, pii_processing="unified")
    assert TextCleaner(configured).cfg.detector_specs == (spec,)


def test_network_provider_is_rejected_before_factory_runs():
    from sct.detectors import DetectorRegistry, DetectorSpec, DetectorUnavailable
    registry = DetectorRegistry()
    def forbidden(cfg, options):
        pytest.fail('Network factory ran without opt-in')
    registry.register('remote', forbidden, network_access=True)
    with pytest.raises(DetectorUnavailable):
        registry.create(DetectorSpec('remote'), TextCleanerConfig(check_ner_process=False))


def test_builtin_factory_creates_fresh_working_detectors():
    from sct.detectors import DetectorRegistry, DetectorSpec
    from sct.findings import DetectionContext
    registry = DetectorRegistry()
    cfg = TextCleanerConfig(check_ner_process=False)
    first = registry.create(DetectorSpec('builtin_regex'), cfg)
    second = registry.create(DetectorSpec('builtin_regex'), cfg)
    assert first is not second
    assert [(hit.start, hit.end, hit.entity_type) for hit in first.detect(
        'a@example.com', language=None, context=DetectionContext())] == [(0, 13, 'EMAIL')]


def worker_registry(cfg):
    from sct.detectors import DetectorRegistry
    registry = DetectorRegistry()
    registry.register('worker', lambda cfg, options: None)
    return registry


def test_worker_handle_rebuilds_registry_after_pickle_and_pid_change(monkeypatch):
    from sct.integrations._runtime import CleanerHandle
    import sct.integrations._runtime as runtime
    handle = CleanerHandle(TextCleanerConfig(check_ner_process=False), registry_factory=worker_registry)
    first = handle.get()
    assert 'worker' in first._detector_registry.identity
    restored = pickle.loads(pickle.dumps(handle))  # noqa: S301
    assert restored.get() is not first
    monkeypatch.setattr(runtime.os, 'getpid', lambda: -1)
    assert handle.get() is not first


def test_config_pickle_preserves_specs():
    from sct.detectors import DetectorSpec
    cfg = TextCleanerConfig(detector_specs=(DetectorSpec('example', {'nested': [1, 2]}),))
    restored = pickle.loads(pickle.dumps(cfg))  # noqa: S301
    assert restored.detector_specs == cfg.detector_specs
    with pytest.raises(TypeError):
        restored.detector_specs[0].options['x'] = 1


@pytest.mark.parametrize('operation', [
    lambda options: options.update(x=1), lambda options: options.setdefault('x', 1),
    lambda options: options.pop('nested'), lambda options: options.popitem(),
    lambda options: options.clear(), lambda options: options.__delitem__('nested'),
    lambda options: options.__ior__({'x': 1}),
])
def test_options_reject_all_ordinary_mutations(operation):
    from sct.detectors import DetectorSpec
    with pytest.raises(TypeError):
        operation(DetectorSpec('example', {'nested': [1]}).options)


def test_network_capability_mismatch_rejected_and_explicit_opt_in_works():
    from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec, DetectorUnavailable
    class Remote:
        capabilities = DetectorCapabilities('remote', network_access=True)
        def detect(self, text, *, language, context):
            return ()
    registry = DetectorRegistry()
    registry.register('remote', lambda cfg, options: Remote())
    with pytest.raises(DetectorUnavailable):
        registry.create(DetectorSpec('remote'), TextCleanerConfig())
    assert registry.create(DetectorSpec('remote'), TextCleanerConfig(allow_network_detectors=True)).capabilities.id == 'remote'


def test_capabilities_copy_nested_identity_values():
    from sct.detectors import DetectorCapabilities
    identities = [['model', 'revision', None]]
    capabilities = DetectorCapabilities('example', model_identities=identities)
    identities[0][0] = 'changed'
    assert capabilities.model_identities == (('model', 'revision', None),)


def test_worker_rejects_import_strings_and_local_factories():
    from sct.integrations._runtime import CleanerHandle
    for factory in ('module.factory', lambda cfg: None):
        with pytest.raises(ValueError):
            CleanerHandle(TextCleanerConfig(check_ner_process=False), registry_factory=factory)


def test_missing_optional_dependency_has_explicit_failure():
    from sct.detectors import DetectorRegistry, DetectorSpec, DetectorUnavailable
    registry = DetectorRegistry()
    def unavailable(cfg, options):
        raise ImportError('optional provider is absent')
    registry.register('optional', unavailable)
    with pytest.raises(DetectorUnavailable):
        registry.create(DetectorSpec('optional'), TextCleanerConfig())


def test_config_type_hints_and_export_are_resolvable():
    from typing import get_type_hints
    from sct import DetectorSpec
    assert get_type_hints(TextCleanerConfig)['detector_specs'] == tuple[DetectorSpec, ...]


def test_supported_haystack_config_serializer_roundtrip():
    pytest.importorskip('haystack')
    from sct.detectors import DetectorSpec
    from sct.integrations.haystack import SCTDocumentCleaner
    cfg = TextCleanerConfig(check_ner_process=False, detector_specs=(DetectorSpec('example', {'items': [1]}),))
    component = SCTDocumentCleaner(config=cfg)
    restored = SCTDocumentCleaner.from_dict(json.loads(json.dumps(component.to_dict())))
    assert restored.handle.config.detector_specs == cfg.detector_specs


def test_cleaner_rejects_invalid_registry_without_processing():
    from sct import TextCleaner
    with pytest.raises(TypeError):
        TextCleaner(TextCleanerConfig(check_ner_process=False), detector_registry='module.registry')


def test_builtin_load_failure_is_detector_unavailable(monkeypatch):
    from sct.detectors import DetectorRegistry, DetectorSpec, DetectorUnavailable
    from sct.utils import ner
    def unavailable(cfg):
        raise ner.ModelLoadError('model artifacts unavailable')
    monkeypatch.setattr(ner, 'GeneralNER', unavailable)
    with pytest.raises(DetectorUnavailable):
        DetectorRegistry().create(DetectorSpec('builtin_ner'), TextCleanerConfig(allow_network_detectors=True))


def pid_registry(cfg):
    import os
    from sct.detectors import DetectorRegistry
    registry = DetectorRegistry()
    registry.register(f'worker_{os.getpid()}', lambda cfg, options: None)
    return registry


def child_registry_identity(handle, connection):
    import os
    connection.send((os.getpid(), handle.get()._detector_registry.identity))
    connection.close()


def test_spawned_worker_builds_registry_in_its_own_process():
    import multiprocessing
    import os
    from sct.integrations._runtime import CleanerHandle
    handle = CleanerHandle(TextCleanerConfig(check_ner_process=False), registry_factory=pid_registry)
    parent_identity = handle.get()._detector_registry.identity
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=child_registry_identity, args=(handle, send))
    process.start()
    send.close()
    try:
        assert receive.poll(20), 'worker did not return a registry identity'
        pid, identity = receive.recv()
        process.join(20)
        assert process.exitcode == 0
        assert pid != os.getpid()
        assert f'worker_{pid}' in identity
        assert f'worker_{os.getpid()}' in parent_identity
        assert f'worker_{os.getpid()}' not in identity
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
            process.join()


def test_registry_rejects_factory_without_detector_method():
    from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec
    class Invalid:
        capabilities = DetectorCapabilities('invalid')
    registry = DetectorRegistry()
    registry.register('invalid', lambda cfg, options: Invalid())
    with pytest.raises(TypeError):
        registry.create(DetectorSpec('invalid'), TextCleanerConfig())


class StatefulRegistryOwner:
    def __init__(self, *, unpicklable=False):
        self.synthetic_key = 'SYNTHETIC_KEY_MARKER'
        self.model = (lambda: None) if unpicklable else None

    def factory(self, cfg):
        return worker_registry(cfg)


@pytest.mark.parametrize('unpicklable', [False, True])
def test_handle_rejects_bound_factory_owner_state(unpicklable):
    from sct.integrations._runtime import CleanerHandle
    owner = StatefulRegistryOwner(unpicklable=unpicklable)
    with pytest.raises(ValueError):
        CleanerHandle(TextCleanerConfig(check_ner_process=False), registry_factory=owner.factory)


@pytest.mark.parametrize('path', [(), ('nested',), ('empty',)])
def test_options_reject_reinitialization(path):
    from sct.detectors import DetectorSpec
    spec = DetectorSpec('example', {'nested': {'limit': 3}, 'empty': {}})
    options = spec.options
    for key in path:
        options = options[key]
    with pytest.raises(TypeError):
        options.__init__({'limit': object()})
    assert spec.options == {'nested': {'limit': 3}, 'empty': {}}


def test_handle_serialization_contains_function_reference_only():
    from sct.integrations._runtime import CleanerHandle
    handle = CleanerHandle(TextCleanerConfig(check_ner_process=False), registry_factory=worker_registry)
    state = handle.__getstate__()
    assert state['registry_factory_reference'] == (__name__, 'worker_registry')
    assert all(not callable(value) for value in state.values())

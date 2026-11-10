"""The trained export must run offline and preserve dynamic-batch quality scores."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(not os.environ.get('SCT_QUALITY_EXPORT_ARTIFACTS'),
                    reason='requires explicitly prepared trained quality artifacts and torch/quantize extras')
def test_trained_quality_export_matches_torch_for_dynamic_shapes(tmp_path, monkeypatch):
    import torch
    from scripts import export_quality_head as exporter
    from sct.score import ONNXQualityScorer

    artifacts = json.loads(Path(os.environ['SCT_QUALITY_EXPORT_ARTIFACTS']).read_text())
    head_info, backbone_info = artifacts['quality_head'], artifacts['quality_backbone']
    for info in (head_info, backbone_info):
        assert len(info['revision']) == 40
        assert Path(info['snapshot']).name == info['revision']
        for name, expected in info['files'].items():
            with (Path(info['snapshot']) / name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    assert head_info['revision'] == 'bf13b1ac9448e874d0eb545891a58524d3f9afd1'
    assert backbone_info['revision'] == 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'

    original_connect = socket.socket.connect
    network_attempts = []
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            network_attempts.append(address)
            raise AssertionError('Quality export must use prepared local artifacts')
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', offline)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')

    # Resolve only prepared artifacts; the real exporter and model execute unchanged.
    def snapshot(repo, *, revision):
        assert repo == head_info['repo'] and revision == head_info['revision']
        return head_info['snapshot']
    monkeypatch.setattr(exporter.huggingface_hub, 'snapshot_download', snapshot)
    def model_info(self, repo, *, revision):
        info = head_info if repo == head_info['repo'] else backbone_info
        assert repo == info['repo'] and revision == info['revision']
        return SimpleNamespace(sha=revision)
    monkeypatch.setattr(exporter.huggingface_hub.HfApi, 'model_info', model_info)
    loaded = {}
    load_model = exporter.AutoModel.from_pretrained
    load_tokenizer = exporter.AutoTokenizer.from_pretrained
    def model(repo, *, revision, **kwargs):
        assert repo == backbone_info['repo'] and revision == backbone_info['revision']
        loaded['backbone'] = load_model(backbone_info['snapshot'], local_files_only=True, **kwargs)
        return loaded['backbone']
    def tokenizer(repo, *, revision, **kwargs):
        assert repo == backbone_info['repo'] and revision == backbone_info['revision']
        loaded['tokenizer'] = load_tokenizer(backbone_info['snapshot'], local_files_only=True, **kwargs)
        return loaded['tokenizer']
    monkeypatch.setattr(exporter.AutoModel, 'from_pretrained', model)
    monkeypatch.setattr(exporter.AutoTokenizer, 'from_pretrained', tokenizer)
    monkeypatch.setattr(sys, 'argv', ['export_quality_head', '--out', str(tmp_path),
        '--head-revision', head_info['revision'], '--backbone-revision', backbone_info['revision']])
    exporter.main()

    head = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Dropout(.2),
                               torch.nn.Linear(256, 1))
    state = torch.load(Path(head_info['snapshot']) / 'model.pt', weights_only=True, map_location='cpu')
    head.load_state_dict({key.removeprefix('classifier.'): value for key, value in state.items()})
    trained = exporter.QualityModel(loaded['backbone'], head).eval()
    scorer = ONNXQualityScorer(tmp_path, revision=head_info['revision'], batch_size=3)
    assert scorer.manifest['backbone_revision'] == backbone_info['revision']
    cohorts = [['A river carries water toward the sea.'],
        ['short', 'A careful experiment records its input and measured output. ' * 8, 'qzx %% 123 junk'],
        ['Contact john.smith@example.org for this synthetic example.', 'Another short text.'],
        # Frozen numerical controls expose saturation; they do not establish model quality.
        ['A river carries water from higher ground toward a lake or the sea. Rain and melting snow feed its flow. '
         'Plants along the banks slow erosion, while roots hold soil in place. Floodplains absorb water during '
         'heavy rain and provide habitat for many species. Scientists measure flow, temperature, and sediment '
         'to understand how the river changes over time.',
         'The project team records each experiment so another engineer can reproduce the result. Each record '
         'includes the input, the method, and the measured output. Clear notes distinguish an observation from '
         'a prediction. Contact john.smith@example.org for this synthetic example; the address is not a real service.']]
    parity = []
    for texts in cohorts:
        tokens = loaded['tokenizer'](texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            expected = trained(tokens['input_ids'], tokens['attention_mask']).tolist()
        actual = scorer(texts)
        assert actual == pytest.approx(expected, rel=1e-4, abs=1e-5)
        assert len(actual) == len(texts) and all(0 <= score <= 1 for score in actual)
        parity.append({'input_shape': list(tokens['input_ids'].shape), 'torch': expected,
                       'onnx': actual, 'max_absolute_error': max(abs(a - b) for a, b in zip(actual, expected))})
    reference_scores = [score for cohort in parity for score in cohort['torch']]
    assert min(reference_scores) < .99, 'Parity controls must include a safely unsaturated trained-model response'
    assert [1.0] * len(reference_scores) != pytest.approx(reference_scores, rel=1e-4, abs=1e-5)
    assert network_attempts == []
    (tmp_path / 'parity.json').write_text(json.dumps({
        'cohorts': parity, 'network_attempts': len(network_attempts),
        'relative_tolerance': 1e-4, 'absolute_tolerance': 1e-5,
        'minimum_reference_score': min(reference_scores), 'constant_one_rejected': True,
        'exporter_sha256': hashlib.sha256(Path(exporter.__file__).read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_sha256': {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                          for name in ('model.onnx', 'tokenizer.json', 'sct_quality.json')},
    }, indent=2) + '\n')

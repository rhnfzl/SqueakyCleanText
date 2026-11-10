"""Private, versioned corpus identity and atomic manifest publication."""
from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import os
from types import MappingProxyType


def _require_bundled_gliner(model_id: str, revision: str) -> None:
    from huggingface_hub import hf_hub_download

    message = ('Prepare a pinned self-contained GLiNER snapshot with embedded encoder_config, '
               'tokenizer_config.json and tokenizer.json; external or auxiliary parent models '
               'are unsupported in native corpus recipes. Direct detector use remains available.')
    try:
        metadata = {
            name: json.loads(Path(hf_hub_download(model_id, name, revision=revision,
                                                  local_files_only=True)).read_text())
            for name in ('gliner_config.json', 'tokenizer_config.json', 'tokenizer.json')
        }
        config = metadata['gliner_config.json']
        encoder = config.get('encoder_config')
        if (not isinstance(encoder, dict) or not encoder.get('model_type')
                or config.get('labels_encoder') or config.get('labels_decoder')
                or not metadata['tokenizer_config.json'].get('tokenizer_class')
                or not isinstance(metadata['tokenizer.json'].get('model'), dict)):
            raise ValueError(message)
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(message) from error


def _json_values(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Configuration keys must be strings')
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_values(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('Unsupported configuration value; use a versioned data-only configuration')


def config_digest(value) -> str:
    """Fingerprint JSON configuration without publishing its sensitive values."""
    try:
        encoded = json.dumps(_json_values(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError):
        raise ValueError('Configuration must contain finite JSON values') from None
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, repr=False)
class RunIdentity:
    run_id: str
    baseline_version: str
    package_version: str
    input_partitions: tuple[tuple[str, str, tuple[str, ...]], ...]
    config_digest: str
    component_identities: tuple[tuple[str, str], ...]
    random_seed: int

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', self.run_id):
            raise ValueError('Invalid run identity')
        if not all(isinstance(x, str) and x for x in (self.baseline_version, self.package_version)):
            raise ValueError('Invalid version identity')
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ValueError('Invalid random seed')
        if not isinstance(self.config_digest, str) or not re.fullmatch('[a-f0-9]{64}', self.config_digest):
            raise ValueError('Invalid configuration digest')
        if not isinstance(self.input_partitions, tuple) or not isinstance(self.component_identities, tuple):
            raise ValueError('Identity collections must be immutable tuples')
        ids = set()
        for partition in self.input_partitions:
            if (not isinstance(partition, tuple) or len(partition) != 3 or not isinstance(partition[0], str)
                    or not isinstance(partition[1], str) or not re.fullmatch('[a-f0-9]{64}', partition[1])
                    or not isinstance(partition[2], tuple)):
                raise ValueError('Invalid input partition identity')
            for record_id in partition[2]:
                if not isinstance(record_id, str) or not record_id or record_id in ids:
                    raise ValueError('Invalid or duplicate stable identity')
                ids.add(record_id)
        if any(not isinstance(x, tuple) or len(x) != 2 or any(not isinstance(v, str) or not v for v in x)
               for x in self.component_identities):
            raise ValueError('Invalid component identity')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, repr=False)
class RunManifest:
    identity: RunIdentity
    output_shards: tuple[tuple[str, str], ...]
    counts: Mapping[str, int] = field(repr=False)
    retention: str = 'caller_managed'
    completion_status: str = 'complete'
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.completion_status != 'complete' or self.retention != 'caller_managed':
            raise ValueError('Invalid manifest state')
        if (not isinstance(self.counts, Mapping) or set(self.counts) != {'input', 'kept', 'rejected'}
                or any(type(v) is not int or v < 0 for v in self.counts.values())
                or self.counts['kept'] + self.counts['rejected'] != self.counts['input']):
            raise ValueError('Manifest counts do not reconcile')
        if not isinstance(self.output_shards, tuple):
            raise ValueError('Output shard identities must be an immutable tuple')
        seen = set()
        for shard in self.output_shards:
            if not isinstance(shard, tuple) or len(shard) != 2:
                raise ValueError('Invalid output shard identity')
            name, digest = shard
            if (not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name
                    or name in seen or not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest)):
                raise ValueError('Invalid output shard identity')
            seen.add(name)
        object.__setattr__(self, 'counts', MappingProxyType(dict(self.counts)))

    def to_dict(self):
        return {'schema_version': self.schema_version, **self.identity.to_dict(),
                'output_shards': dict(self.output_shards), 'counts': dict(self.counts),
                'retention': self.retention, 'completion_status': self.completion_status}

    def publish(self, path: Path):
        if path.exists():
            raise ValueError('Manifest destination already exists')
        for name, digest in self.output_shards:
            if file_digest(path.parent / name) != digest:
                raise ValueError('Output checksum mismatch')
        temporary = path.with_suffix('.pending')
        with temporary.open('x', encoding='utf-8') as stream:
            temporary.chmod(0o600)
            json.dump(self.to_dict(), stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

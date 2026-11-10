"""Explicit real-provider smoke checks, never auto-install or fetch models."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import pytest

PROFILE = os.environ.get("SCT_OPTIONAL_PROFILE")
pytestmark = pytest.mark.skipif(not PROFILE, reason="select an installed SCT_OPTIONAL_PROFILE")
PROVIDERS = {
    "secrets": ("datafog-core", "datafog_core"),
    "documents": ("docling", "docling"),
    "token-store": ("cryptography", "cryptography"),
    "corpus": ("datatrove", "datatrove"),
}


def test_installed_profile_runtime(tmp_path):
    distribution, module_name = PROVIDERS[PROFILE]
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(Path(sys.prefix).resolve())
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "python": sys.version,
                "version": importlib.metadata.version(distribution),
                "module_path": str(module_path),
            }
        )
    )
    if PROFILE == "token-store":
        from cryptography.fernet import Fernet

        cipher = Fernet(Fernet.generate_key())
        assert cipher.decrypt(cipher.encrypt(b"profile smoke")) == b"profile smoke"
    elif PROFILE == "corpus":
        import spacy
        import orjson

        assert Path(spacy.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert importlib.metadata.version("spacy") == "3.8.16"
        assert [token.text for token in spacy.blank("en")("Native fixture")] == ["Native", "fixture"]
        assert orjson.loads(orjson.dumps({"native": True})) == {"native": True}
        from datasketch import MinHash

        first, second = MinHash(num_perm=32), MinHash(num_perm=32)
        first.update(b"same")
        second.update(b"same")
        assert first.jaccard(second) == 1.0
        from datatrove.pipeline.readers import JsonlReader

        source = tmp_path / "source"
        source.mkdir()
        (source / "input.jsonl").write_text('{"id":"smoke","text":"Local fixture"}\n')
        assert [doc.text for doc in JsonlReader(str(source))()] == ["Local fixture"]
    elif PROFILE == "documents":
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat

        converter = DocumentConverter(allowed_formats=[InputFormat.MD])
        source = tmp_path / "fixture.md"
        source.write_text("# Local heading\n\nLocal fixture.")
        assert "Local fixture" in converter.convert(source).document.export_to_markdown()
    else:
        text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"
        findings = module.scan(text)
        assert any(
            f.entity_type == "BEARER_TOKEN" and (f.codepoint_range.start, f.codepoint_range.end) == (22, 54) for f in findings
        )

"""Haystack component (``pip install squeakycleantext[haystack]``)."""

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, Sequence, Union

from haystack import Document, component, default_from_dict, default_to_dict

from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle, as_columns, clean_records


def _plain(value: Any) -> Any:
    """Mappings and sets, at any depth, become types Haystack can serialize."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


@component
class SCTDocumentCleaner:
    """Clean ``content`` (and listed ``meta`` keys); ``meta["sct"]`` gets the signals.

    Returns new documents with ids recomputed from the cleaned content.
    """

    def __init__(
        self,
        columns: Sequence[str] = ("content",),
        output: str = "overwrite",
        config: Union[TextCleanerConfig, str, None] = None,
    ):
        self.columns = as_columns(columns)
        self.output = output
        self._config = config
        self.handle = CleanerHandle(config)

    def to_dict(self) -> dict[str, Any]:
        """Preset names stay names; a config keeps only its non-default fields.

        Haystack raises SerializationError for values it cannot store (custom steps).
        """
        config = self._config
        if isinstance(config, TextCleanerConfig):
            default = TextCleanerConfig()
            config = {
                f.name: _plain(value)
                for f in fields(config)
                # supported_languages is recomputed by __post_init__.
                if f.name != "supported_languages"
                and (value := getattr(config, f.name)) != getattr(default, f.name)
            }
        return default_to_dict(self, columns=self.columns, output=self.output, config=config)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCTDocumentCleaner":
        params = dict(data.get("init_parameters", {}))
        if isinstance(params.get("config"), dict):
            params["config"] = TextCleanerConfig(**params["config"])
        return default_from_dict(cls, {**data, "init_parameters": params})

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        pairs = clean_records(
            self.handle, [(d.content, d.meta) for d in documents], "content", self.columns, self.output
        )
        # id="" makes Document recompute it, so no hash of the raw text survives.
        return {"documents": [replace(d, id="", content=t, meta=m) for d, (t, m) in zip(documents, pairs)]}

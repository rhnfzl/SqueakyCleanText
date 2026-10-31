"""Haystack component (``pip install squeakycleantext[haystack]``)."""

from dataclasses import replace
from typing import Sequence, Union

from haystack import Document, component

from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle, clean_records


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
        self.columns = tuple(columns)
        self.output = output
        self.handle = CleanerHandle(config)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        pairs = clean_records(
            self.handle, [(d.content, d.meta) for d in documents], "content", self.columns, self.output
        )
        # id="" makes Document recompute it, so no hash of the raw text survives.
        return {"documents": [replace(d, id="", content=t, meta=m) for d, (t, m) in zip(documents, pairs)]}

"""LangChain document transformer (``pip install squeakycleantext[langchain]``)."""

from typing import Any, Sequence, Union

from langchain_core.documents import BaseDocumentTransformer, Document

from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle, as_columns, clean_records


class SCTDocumentTransformer(BaseDocumentTransformer):
    """Clean ``page_content`` (and listed metadata keys); ``metadata["sct"]`` gets the signals.

    Returns new documents; the inputs are not modified.
    """

    def __init__(
        self,
        columns: Sequence[str] = ("page_content",),
        output: str = "overwrite",
        config: Union[TextCleanerConfig, str, None] = None,
    ):
        self.columns = as_columns(columns)
        self.output = output
        self.handle = CleanerHandle(config)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        docs = list(documents)
        pairs = clean_records(
            self.handle, [(d.page_content, d.metadata) for d in docs], "page_content", self.columns, self.output
        )
        return [d.model_copy(update={"page_content": t, "metadata": m}) for d, (t, m) in zip(docs, pairs)]

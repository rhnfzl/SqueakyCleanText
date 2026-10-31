"""DataTrove pipeline step (``pip install squeakycleantext[datatrove]``)."""

from itertools import batched
from typing import Sequence, Union

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints

from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle, as_columns, clean_records


class SCTCleaner(PipelineStep):
    """Clean ``doc.text`` (and listed ``doc.metadata`` keys); ``doc.metadata["sct"]`` gets the signals.

    Place it after language and quality filters. The cleaner is built inside
    each worker on first use.
    """

    name = "SCT"
    type = "🧹 - CLEAN"

    def __init__(
        self,
        columns: Sequence[str] = ("text",),
        output: str = "overwrite",
        config: Union[TextCleanerConfig, str, None] = None,
        batch_size: int = 64,
    ):
        super().__init__()
        self.columns = as_columns(columns)
        self.output = output
        self.batch_size = batch_size
        self.handle = CleanerHandle(config)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for chunk in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                pairs = clean_records(
                    self.handle, [(d.text, d.metadata) for d in chunk], "text", self.columns, self.output
                )
            for doc, (text, meta) in zip(chunk, pairs):
                self.stat_update(StatHints.total)
                doc.text, doc.metadata = text, meta
                signals = meta.get("sct")
                if signals:
                    self.stat_update("pii", value=signals["pii_count"])
                    if signals["sensitive"]:
                        self.stat_update("sensitive")
                    for pii_type, count in signals["pii_types"].items():
                        self.stat_update(f"pii_{pii_type}", value=count)
                yield doc

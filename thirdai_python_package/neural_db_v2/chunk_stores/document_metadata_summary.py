import random
import sys

from pandas import Series

from ..core.types import (
    ChunkMetaDataSummary,
    MetadataType,
    NumericChunkMetadataSummary,
    StringChunkMetadataSummary,
)


class DocumentMetadataSummary:
    def __init__(self):
        self.summarized_metadata = {}

    def summarize_metadata(
        self,
        key: str,
        metadata_series: Series,
        metadata_type: MetadataType,
        doc_id: int,
        doc_version: int,
        overwrite_type: bool = False,
    ):
        metadata_series = metadata_series.dropna()

        if metadata_series.isna().all():
            # all the values of the series is None
            return

        if (doc_id, doc_version) not in self.summarized_metadata:
            self.summarized_metadata[(doc_id, doc_version)] = {}

        if overwrite_type or key not in self.summarized_metadata[(doc_id, doc_version)]:
            summary = (
                NumericChunkMetadataSummary(max=-sys.maxsize + 1, min=sys.maxsize)
                if metadata_type in [MetadataType.FLOAT, MetadataType.INTEGER]
                else StringChunkMetadataSummary(unique_values=set())
            )
            self.summarized_metadata[(doc_id, doc_version)] = {
                key: ChunkMetaDataSummary(metadata_type=metadata_type, summary=summary)
            }

        if metadata_type in [MetadataType.FLOAT, MetadataType.INTEGER]:
            # Summarizing the Numeric metadata series
            self.summarized_metadata[(doc_id, doc_version)].summary.min = min(
                self.summarized_metadata[(doc_id, doc_version)].summary.min,
                metadata_series.min(),
            )
            self.summarized_metadata[(doc_id, doc_version)].summary.max = max(
                self.summarized_metadata[(doc_id, doc_version)].summary.max,
                metadata_series.max(),
            )
        else:
            # Summarizing the string/bool metadata series
            unique_values = self.summarized_metadata[
                (doc_id, doc_version)
            ].summary.unique_values
            unique_values.add(set(metadata_series.unique()))
            unique_values.discard(None)  # Remove None value
            self.summarized_metadata[(doc_id, doc_version)].summary.unique_values = (
                random.sample(unique_values, k=100)
            )

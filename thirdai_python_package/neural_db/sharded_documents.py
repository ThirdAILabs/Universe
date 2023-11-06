import copy
import os
import random
import tempfile
from io import StringIO
from typing import List, Tuple

import pandas as pd

from .documents import CSV, DocumentDataSource


class ShardedDataSource:
    def __init__(self, document_data_source: DocumentDataSource, number_shards: int):
        self.data_source = ShardedDataSource.copy_datasource(document_data_source)
        self.number_shards = number_shards
        self.seed = 0
        self.segment_prefix = f"{random.randint(100000, 999999)}"
        self.label_index = {}

    @staticmethod
    def copy_datasource(document_data_source: DocumentDataSource):
        """
        This function makes a deep copy of the Document Data Source. Ideally, we should not have to make a copy of the Document Data Source (we can restart it?)
        """
        new_data_source = DocumentDataSource(
            document_data_source.id_column,
            document_data_source.strong_column,
            document_data_source.weak_column,
        )
        new_data_source.documents = copy.deepcopy(document_data_source.documents)
        new_data_source._size = document_data_source._size
        return new_data_source

    def generate_temp_csvs(self, segments: List[pd.DataFrame]):
        """
        Stores a list of dataframes in temporary files so that they can be read as CSV files later.
        """
        segment_filenames = []
        # We need to store the segment objects so that we can delete the files once we are done with sharding and creating a new dataframe
        segment_objects = []
        for index, segment in enumerate(segments):
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                delete=True,
                suffix=".csv",
                prefix=f"{self.segment_prefix}_{index}_",
            )

            segment_name = temp_file.name
            segment.to_csv(segment_name, index=False)
            segment_filenames.append(segment_name)
            segment_objects.append(temp_file)
        return segment_filenames, segment_objects

    def get_csv_document(
        self, shard_name: str, shard_object: tempfile.NamedTemporaryFile
    ):
        csv_object = CSV(
            path=shard_name,
            id_column=self.data_source.id_column,
            strong_columns=[self.data_source.strong_column],
            weak_columns=[self.data_source.weak_column],
        )
        shard_object.close()
        return csv_object

    @staticmethod
    def get_dataframe(data_source: DocumentDataSource):
        """
        Iterates through the document data source and generates a dataframe
        """
        string_io = StringIO("\n".join(data_source._get_line_iterator()))
        df = pd.read_csv(string_io)
        return df

    def shard_data_source(self):
        df = ShardedDataSource.get_dataframe(self.data_source)

        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        segment_size = len(df) // self.number_shards
        segments = [
            df.iloc[i * segment_size : (i + 1) * segment_size]
            for i in range(self.number_shards)
        ]

        if len(df) % self.number_shards != 0:
            segments[-1] = pd.concat(
                [segments[-1], df.iloc[self.number_shards * segment_size :]]
            )

        for index, segment in enumerate(segments):
            unique_labels = (
                segment[self.data_source.id_column].unique().astype(int).tolist()
            )
            for label in unique_labels:
                if label not in self.label_index:
                    self.label_index[label] = []
                self.label_index[label].append(index)

        return self.generate_temp_csvs(segments)

    def get_shards(
        self, shard_names=None, shard_objects=None
    ) -> List[DocumentDataSource]:
        if not shard_names or not shard_objects:
            shard_names, shard_objects = self.shard_data_source()

        shard_data_sources = []
        for name, temp_object in zip(shard_names, shard_objects):
            shard_data_source = DocumentDataSource(
                id_column=self.data_source.id_column,
                strong_column=self.data_source.strong_column,
                weak_column=self.data_source.weak_column,
            )

            shard_data_source.add(self.get_csv_document(name, temp_object), start_id=0)
            shard_data_sources.append(shard_data_source)
        return shard_data_sources

    def shard_using_index(self, data_source: DocumentDataSource):
        """
        This function is used to shard another data source using the label to shard mapping generated for the data source that this object was initialized with.
        """
        if len(self.label_index) == 0:
            raise Exception(
                "Cannot shard a data source without an uninitialized label index."
            )

        df = ShardedDataSource.get_dataframe(data_source)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        segments = [[] for _ in range(self.number_shards)]
        for index, row in df.iterrows():
            labels = [
                int(x) for x in str(row[self.data_source.id_column]).split(":") if x
            ]

            insertion_index_segments = set()
            for label in labels:
                if label in self.label_index:
                    target_segments = set(self.label_index[label])
                    for target in target_segments:
                        insertion_index_segments.add(target)
            for x in insertion_index_segments:
                segments[x].append(row)

        segments = [pd.DataFrame(segment) for segment in segments]

        shard_names, shard_objects = self.generate_temp_csvs(segments)

        return self.get_shards(shard_names=shard_names, shard_objects=shard_objects)

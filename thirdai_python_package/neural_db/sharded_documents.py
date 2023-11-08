import copy
import random
import tempfile
from io import StringIO
from typing import List, Tuple

import pandas as pd

from .documents import CSV, DocumentDataSource


class ShardedDataSource:
    """
    Initialization Variables:
        * document_data_source -> The data source we are supposed to shard.
        * number_shards -> The number of shards to create for the data source.
        * label_index -> A dictionary that tracks what label goes to what shard. This label index is supposed to be a dictionary reference from Mach Mixture class and this class will modify the label_index.
        * seed -> Seed for sharding the dataset (since we randomly shard the data source)

    External APIs :
        shard_data_source :
            Args:
                self : ShardedDataSource
            Returns:
                sharded_data_sources : List[DocumentDataSource]
                    Each element in the list corresponds to a shard of the original data source
            Note:
                Updates the label index with label_id -> shard index map

        shard_using_index:
            Args:
                data_source : DocumentDataSource
                    Data source to shard
                label_index : dict
                    Label index used to shard the data source
                number_shards : int
                    number of shards to create for the data source.
            Returns:
                sharded_data_sources : List[DocumentDataSource]
                    Each element in the list corresponds to a shard of the original data source.
            Note:
                Does not modify the label index.
    """

    def __init__(
        self,
        document_data_source: DocumentDataSource,
        number_shards: int,
        label_index: dict = {},
        seed: int = 0,
    ):
        self.data_source = document_data_source
        self.number_shards = number_shards
        self.seed = seed
        self.label_index = label_index

    @staticmethod
    def _generate_temp_csvs(segments: List[pd.DataFrame]):
        """
        Stores a list of dataframes in temporary files so that they can be read as CSV files later.
        """
        segment_prefix = f"{random.randint(100000, 999999)}"
        segment_filenames = []
        # We need to store the segment objects so that we can delete the files once we are done with sharding and creating a new dataframe
        segment_objects = []
        for index, segment in enumerate(segments):
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                delete=True,
                suffix=".csv",
                prefix=f"{segment_prefix}_{index}_",
            )

            segment_name = temp_file.name
            segment.to_csv(segment_name, index=False)
            segment_filenames.append(segment_name)
            segment_objects.append(temp_file)
        return segment_filenames, segment_objects

    @staticmethod
    def _get_csv_document(
        id_column: str,
        strong_column: str,
        weak_column: str,
        shard_name: str,
        shard_object: tempfile.NamedTemporaryFile,
    ):
        """
        This function takes as input the name of the tempfile and the tempfile object. We load the tempfile into a CSV Document and then closes the tempfile (which effectively means deleting it)
        """
        csv_object = CSV(
            path=shard_name,
            id_column=id_column,
            strong_columns=[strong_column],
            weak_columns=[weak_column],
            has_offset=True,
        )
        shard_object.close()
        return csv_object

    @staticmethod
    def _get_dataframe(data_source: DocumentDataSource):
        """
        Iterates through the document data source and generates a dataframe
        """
        string_io = StringIO("\n".join(data_source._get_line_iterator()))
        df = pd.read_csv(string_io)
        data_source.restart()
        return df

    def shard_data_source(self):
        df = ShardedDataSource._get_dataframe(self.data_source)

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

        shard_names, shard_objects = ShardedDataSource._generate_temp_csvs(segments)

        shards = ShardedDataSource._get_shards(
            self.data_source, shard_names=shard_names, shard_objects=shard_objects
        )
        return shards

    @staticmethod
    def _get_shards(
        data_source: DocumentDataSource, shard_names=None, shard_objects=None
    ) -> List[DocumentDataSource]:
        shard_data_sources = []
        for name, temp_object in zip(shard_names, shard_objects):
            shard_data_source = DocumentDataSource(
                id_column=data_source.id_column,
                strong_column=data_source.strong_column,
                weak_column=data_source.weak_column,
            )

            shard_data_source.add(
                ShardedDataSource._get_csv_document(
                    data_source.id_column,
                    data_source.strong_column,
                    data_source.weak_column,
                    name,
                    temp_object,
                ),
                start_id=0,
            )
            shard_data_sources.append(shard_data_source)
        return shard_data_sources

    @staticmethod
    def shard_using_index(
        data_source: DocumentDataSource,
        label_index: dict,
        number_shards: int,
    ):
        """
        This function is used to shard another data source using the label to shard mapping generated for the data source that this object was initialized with.
        """
        if len(label_index) == 0:
            raise Exception(
                "Cannot shard a data source without an uninitialized label index."
            )

        df = ShardedDataSource._get_dataframe(data_source)

        segments = [[] for _ in range(number_shards)]
        for _, row in df.iterrows():
            labels = [int(x) for x in str(row[data_source.id_column]).split(":") if x]

            insertion_index_segments = set()
            for label in labels:
                if label in label_index:
                    target_segments = set(label_index[label])
                    for target in target_segments:
                        insertion_index_segments.add(target)
            for x in insertion_index_segments:
                segments[x].append(row)

        segments = [pd.DataFrame(segment) for segment in segments]

        shard_names, shard_objects = ShardedDataSource._generate_temp_csvs(segments)

        return ShardedDataSource._get_shards(
            data_source=data_source,
            shard_names=shard_names,
            shard_objects=shard_objects,
        )

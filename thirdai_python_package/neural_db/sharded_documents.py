import copy
import random
import tempfile
from collections import defaultdict
from io import StringIO
from typing import List

import pandas as pd

from .documents import CSV, DocumentDataSource


class DataLoadMultiplexer:
    def __init__(self, num_segments, flush_frequency=1_000_000):
        self.num_segments = num_segments
        self.flush_frequency = flush_frequency

    def _generate_temp_csvs(self):
        """
        Stores a list of dataframes in temporary files so that they can be read as CSV files later.
        """
        segment_prefix = f"{random.randint(100000, 999999)}"
        segment_filenames = []
        # We need to store the segment objects so that we can delete the files once we are done with sharding and creating a new dataframe
        segment_objects = []
        for index in range(self.num_segments):
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                delete=True,
                suffix=".csv",
                prefix=f"{segment_prefix}_{index}_",
            )

            segment_filenames.append(temp_file.name)
            segment_objects.append(temp_file)
        return segment_filenames, segment_objects

    def create_segments_with_segment_map(self, data_source, label_to_segment_map):
        segment_filenames, segment_objects = self._generate_temp_csvs()

        current_index = 0
        for data in data_source._get_line_iterator():
            # header
            if current_index == 0:
                for segments in segment_objects:
                    segments.write(data)
            else:
                current_label = int(data.split(",", 1)[0])
                # TODO(pratik/shubh): Having list as map values is for experiments,
                # we would be just having one elment in list for each index. We should
                # remove this going forward.
                current_segment = label_to_segment_map[current_label][-1]
                segment_objects[current_segment].write("\n" + data)

            current_index += 1
            if current_index % self.flush_frequency == 0:
                for segment in segment_objects:
                    segment.flush()

        for segment in segment_objects:
            segment.flush()

        data_source.restart()
        return (
            segment_filenames,
            segment_objects,
            label_to_segment_map,
        )

    def create_segments_with_data_source(
        self, data_source, label_to_segment_map, shard_using_index=False
    ):
        if shard_using_index:
            return self.create_segments_with_segment_map(
                data_source, label_to_segment_map
            )
        else:
            indices = [i for i, _ in enumerate(data_source._get_line_iterator())]
            random.shuffle(indices)
            for index, randomised_index in enumerate(indices):
                label_to_segment_map[index].append(randomised_index % self.num_segments)

            data_source.restart()
            return self.create_segments_with_segment_map(
                data_source, label_to_segment_map
            )


class ShardedDataSource:
    """
    Initialization Variables:
        * document_data_source -> The data source we are supposed to shard.
        * number_shards -> The number of shards to create for the data source.
        * label_to_segment_map -> A dictionary that tracks what label goes to what shard. This label index is supposed to be a dictionary reference from Mach Mixture class and this class will modify the label_to_segment_map.
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
                label_to_segment_map : dictionary
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
        label_to_segment_map: defaultdict = None,
        seed: int = 0,
    ):
        self.data_source = document_data_source
        self.number_shards = number_shards
        self.seed = seed
        if label_to_segment_map == None:
            self.label_to_segment_map = defaultdict(list)
        else:
            self.label_to_segment_map = label_to_segment_map
        self.data_load_multiplexer = DataLoadMultiplexer(number_shards)

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

    def shard_data_source(self):
        (
            shard_names,
            shard_objects,
            self.label_to_segment_map,
        ) = self.data_load_multiplexer.create_segments_with_data_source(
            self.data_source, self.label_to_segment_map, shard_using_index=False
        )

        shards = ShardedDataSource._get_shards(
            self.data_source, shard_names=shard_names, shard_objects=shard_objects
        )
        return shards

    @staticmethod
    def shard_using_index(
        data_source: DocumentDataSource,
        label_to_segment_map: defaultdict,
        number_shards: int,
        flush_frequency: int = 1_000_000,
    ):
        """
        This function is used to shard another data source using the label to shard mapping generated for the data source that this object was initialized with.
        """
        if len(label_to_segment_map) == 0:
            raise Exception(
                "Cannot shard a data source without an uninitialized label index."
            )

        (shard_names, shard_objects, _) = DataLoadMultiplexer(
            num_segments=number_shards, flush_frequency=flush_frequency
        ).create_segments_with_data_source(
            data_source, label_to_segment_map, shard_using_index=True
        )

        return ShardedDataSource._get_shards(
            data_source=data_source,
            shard_names=shard_names,
            shard_objects=shard_objects,
        )

import random
import tempfile
from collections import defaultdict
from typing import List, Union

from .documents import CSV, DocumentDataSource
from .supervised_datasource import Sup, SupDataSource


class DataLoadMultiplexer:
    """
    A data loader that efficiently handles large datasets by segmenting them into smaller,
    manageable temporary CSV files. This approach is particularly useful for processing
    large datasets that do not fit entirely into memory.

    The class optimizes memory usage by reading and writing data line by line,
    rather than loading entire datasets into DataFrames. This method reduces the
    memory footprint, especially when dealing with large datasets, and avoids
    the overhead of storing extra DataFrame objects in memory.

    Attributes:
        num_segments (int): Number of segments to divide the data into.
        flush_frequency (int): Frequency at which to flush data to the temporary files,
                               reducing memory usage during processing.

    Methods:
        create_segments_with_data_source(data_source, label_to_segment_map, update_index):
            Creates data segments based on the provided data source and label mapping,
            optionally sharding the data using an index.

    The class utilizes temporary files to store individual data segments, which are then
    processed independently. This design allows for efficient data handling and scalability,
    making it suitable for large-scale data processing tasks where memory optimization is crucial.

    NOTE: This multiplexer assumes that the id column is the first element of the csv lines yielded by the line iterator of the data source. And hence, this multiplexer is only supposed to be used with DataSources with the above behaviour.
    """

    def __init__(self, num_segments, flush_frequency=1_000_000):
        self.num_segments = num_segments
        self.flush_frequency = flush_frequency
        self.seed = 42

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

    def _create_segments_with_segment_map(
        self,
        data_source: Union[DocumentDataSource, SupDataSource],
        label_to_segment_map,
    ):
        segment_filenames, segment_objects = self._generate_temp_csvs()

        current_index = 0

        if isinstance(data_source, DocumentDataSource):
            line_iterator = data_source._get_line_iterator()
        if isinstance(data_source, SupDataSource):
            # SupDataSource's line iterator can return multiple labels in the same line concatened by its id_delimiter and hence, we pass concat_labels = False so that each line has just one label.
            line_iterator = data_source._get_line_iterator(concat_labels=False)  # type: ignore

        for data in line_iterator:  # type: ignore
            # header
            if current_index == 0:
                for segments in segment_objects:
                    segments.write(data)
            else:
                current_label = int(data.split(",", 1)[0])
                # TODO(pratik/shubh): Having list as map values is for experiments,
                # we would be just having one elment in list for each index. We should
                # remove this going forward.
                if current_label not in label_to_segment_map:
                    raise ValueError(
                        "Label '{}' is not in 'label_to_segment_map'. Ensure it is"
                        " included if sharding by index.".format(current_label)
                    )
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

    def update_label_segment_map(self, data_source, label_to_segment_map):
        indices = data_source.indices()
        random.seed(self.seed)
        random.shuffle(indices)
        for randomised_index in indices:
            label_to_segment_map[randomised_index].append(
                randomised_index % self.num_segments
            )

    def create_segments_with_data_source(
        self, data_source, label_to_segment_map, update_index
    ):
        if update_index:
            self.update_label_segment_map(data_source, label_to_segment_map)
        else:
            if len(label_to_segment_map) == 0:
                raise Exception("label_to_segment_map is empty")

        return self._create_segments_with_segment_map(data_source, label_to_segment_map)


def verify_data_source_type(data_source, valid_types: List, operation_name: str):
    if not any(isinstance(data_source, data_type) for data_type in valid_types):
        valid_types_str = ", ".join([data_type.__name__ for data_type in valid_types])
        raise TypeError(
            f"{operation_name} not supported for data source of the type"
            f" {type(data_source)}. Expected types are: {valid_types_str}"
        )


def transform_shard_to_datasource(
    original_data_source: Union[DocumentDataSource, SupDataSource],
    shard_path,
    shard_object,
):
    """
    We assume that the shard is stored as a CSV on the machine. Depending on the underlying original data source, we create a new data source from the shard with the same attributes except that the documents are loaded from the shard.
    """
    verify_data_source_type(
        data_source=original_data_source,
        valid_types=[DocumentDataSource, SupDataSource],
        operation_name="transform_shard_to_datasource",
    )
    if isinstance(original_data_source, DocumentDataSource):
        data_source = DocumentDataSource(
            id_column=original_data_source.id_column,
            strong_column=original_data_source.strong_column,
            weak_column=original_data_source.weak_column,
        )
        csv_doc = CSV(
            path=shard_path,
            id_column=original_data_source.id_column,
            strong_columns=[original_data_source.strong_column],
            weak_columns=[original_data_source.weak_column],
            has_offset=True,
        )

        shard_object.close()
        data_source.add(document=csv_doc, start_id=0)
        return data_source

    if isinstance(original_data_source, SupDataSource):
        sup = Sup(
            csv=shard_path,
            query_column=original_data_source.query_col,
            id_column=original_data_source.id_column,
            id_delimiter=original_data_source.id_delimiter,
            uses_db_id=True,
        )
        shard_object.close()
        return SupDataSource(
            doc_manager=original_data_source.doc_manager,
            query_col=original_data_source.query_col,
            data=[sup],
            id_delimiter=original_data_source.id_delimiter,
            id_column=original_data_source.id_column,
        )


def shard_data_source(
    data_source: Union[DocumentDataSource, SupDataSource],
    label_to_segment_map: defaultdict,
    number_shards: int,
    update_segment_map: bool,
    flush_frequency: int = 1_000_000,
):
    """
    NOTE:
    1. Sharding only works for data sources that have id column as the first element of the lines yielded by their line iterator and that each line contains only one id.
    2. Sharding is supported for DocumentDataSource and SupDataSource currently.
    Args:
        data_source : DocumentDataSource or SupDataSource
            The data source to be sharded
        label_to_segment_map : defaultdict
            map of label id to shard id
        number_shards : int
            number of shards to shard the dataset into
            If number_shards is 1, then returns the datasource as it is.
        update_segment_map : bool
            If set to True, then we first randomly shard the data_source, and update the label_to_segment_map.
            If set to False, then the data_source is sharded using the label_to_segment_map provided by the user.
    Returns:
        sharded_data_sources : List[DocumentDataSource] or List[SupDataSource]
            Each element in the list corresponds to a shard of the original data source
    Note:
        Updates the label_to_segment_map with label_id -> shard index map if update_segment_map is True.
    """
    verify_data_source_type(
        data_source=data_source,
        valid_types=[DocumentDataSource, SupDataSource],
        operation_name="shard_data_source",
    )

    data_load_multiplexer = DataLoadMultiplexer(
        number_shards, flush_frequency=flush_frequency
    )

    if number_shards == 1:
        if update_segment_map:
            data_load_multiplexer.update_label_segment_map(
                data_source, label_to_segment_map
            )
        return [data_source]

    (
        shard_names,
        shard_objects,
        _,
    ) = data_load_multiplexer.create_segments_with_data_source(
        data_source, label_to_segment_map, update_index=update_segment_map
    )

    return [
        transform_shard_to_datasource(data_source, shard_path, shard_object)
        for shard_path, shard_object in zip(shard_names, shard_objects)
    ]

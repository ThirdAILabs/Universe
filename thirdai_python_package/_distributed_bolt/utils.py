import importlib
import logging

from thirdai import data


def init_logging(logger_file: str):
    """
    Returns logger from a logger file
    """
    logger = logging.getLogger(logger_file)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logger_file)
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_num_cpus():
    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except ImportError:
        print("Could not find num_cpus, setting num_cpus to DEFAULT=1")
        return 1


def check_torch_installed():
    try:
        importlib.import_module("torch")
    except ImportError as e:
        raise ImportError(
            "Distributed Bolt requires Torch Distributed as its communication backend. Please ensure that Torch is installed to enable distributed training with Bolt.t"
        ) from e


def _pandas_iterator(path, chunksize, node_index, num_nodes, sep):
    import pandas as pd

    with pd.read_csv(path, chunksize=chunksize, sep=sep) as reader:
        for chunk_id, chunk in enumerate(reader):
            if chunk_id % num_nodes == node_index:
                yield chunk
    while True:
        yield None


class PandasColumnMapGenerator(data.ColumnMapGenerator):
    def __init__(
        self,
        path,
        num_nodes,
        node_index,
        lines_per_load,
        dense_int_cols=set(),
        int_col_dims={},
        col_dtype_overrides={},
        load_whole_file_per_node=False,
        sep=",",
    ):
        self.path = path
        self.num_nodes = num_nodes
        self.node_index = node_index
        self.lines_per_load = lines_per_load
        self.dense_int_cols = dense_int_cols
        self.int_col_dims = int_col_dims
        self.current_iterator = None
        self.col_dtype_overrides = col_dtype_overrides
        self.load_whole_file_per_node = load_whole_file_per_node
        self.sep = sep

    def next(self):
        # We do this here instead of the constructor so we don't need to
        # pickle the generator
        if self.current_iterator == None:
            self.restart()

        load = next(self.current_iterator)
        if load is None:
            return None

        for col_name, col_type in self.col_dtype_overrides.items():
            load[col_name] = load[col_name].astype(col_type)

        return data.pandas_to_columnmap(
            load,
            dense_int_cols=self.dense_int_cols,
            int_col_dims=self.int_col_dims,
        )

    def restart(self):
        if not self.load_whole_file_per_node:
            self.current_iterator = _pandas_iterator(
                self.path,
                self.lines_per_load,
                self.node_index,
                self.num_nodes,
                self.sep,
            )
        else:
            # Passing in 0 as node index and 1 as num_nodes, make sure that
            # we iterate over all the data source
            self.current_iterator = _pandas_iterator(
                self.path,
                self.lines_per_load,
                node_index=0,
                num_nodes=1,
                sep=self.sep,
            )

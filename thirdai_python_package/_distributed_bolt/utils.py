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
    except (ImportError):
        print("Could not find num_cpus, setting num_cpus to DEFAULT=1")
        return 1


def get_gradients(wrapped_model):
    """
    :return: list of gradients, in order of node traversal. The order is
    guarenteed to be the same for all nodes because the model is compiled before
    being distributed.
    """
    nodes = wrapped_model.model.nodes()
    gradients = []
    for node in nodes:
        if hasattr(node, "weight_gradients"):
            gradients.append(node.weight_gradients.copy())
        if hasattr(node, "bias_gradients"):
            gradients.append(node.bias_gradients.copy())

    return gradients


def set_gradients(wrapped_model, gradients):
    """
    This function sets the gradients in the current network to the
    gradients provided, in the same order as get_gradients
    """
    nodes = wrapped_model.model.nodes()
    gradient_position = 0
    for node in nodes:
        if hasattr(node, "weight_gradients"):
            node.weight_gradients.set(gradients[gradient_position])
            gradient_position += 1
        if hasattr(node, "bias_gradients"):
            node.bias_gradients.set(gradients[gradient_position])
            gradient_position += 1

    return gradients


def _pandas_iterator(path, chunksize, node_index, num_nodes):
    import pandas as pd

    with pd.read_csv(path, chunksize=chunksize) as reader:
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
    ):
        self.path = path
        self.num_nodes = num_nodes
        self.node_index = node_index
        self.lines_per_load = lines_per_load
        self.dense_int_cols = dense_int_cols
        self.int_col_dims = int_col_dims
        self.current_iterator = None
        self.col_dtype_overrides = col_dtype_overrides

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
        self.current_iterator = _pandas_iterator(
            self.path, self.lines_per_load, self.node_index, self.num_nodes
        )

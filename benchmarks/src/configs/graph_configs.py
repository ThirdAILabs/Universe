from thirdai import bolt

from .udt_configs import UDTBenchmarkConfig
from .utils import AdditionalMetricCallback, get_gnn_roc_auc_metric_fn


class YelpChiUDTBenchmark(UDTBenchmarkConfig):
    config_name = "yelp_chi_udt"
    dataset_name = "yelp_chi"

    train_file = "yelp_chi/yelp_train.csv"
    test_file = "yelp_chi/yelp_test.csv"

    target = "target"
    delimiter = ","

    num_epochs = 15
    learning_rate = 0.001
    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=get_gnn_roc_auc_metric_fn("target"),
        )
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "node_id": bolt.types.node_id(),
            **{f"col_{i}": bolt.types.numerical([0, 1]) for i in range(32)},
            "target": bolt.types.categorical(n_classes=2, type="int"),
            "neighbors": bolt.types.neighbors(),
        }


class PokecUDTBenchmark(UDTBenchmarkConfig):
    config_name = "pokec_udt"
    dataset_name = "pokec"

    train_file = "pokec/train.csv"
    test_file = "pokec/test.csv"

    target = "target"
    delimiter = ","

    num_epochs = 15
    learning_rate = 0.001
    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=get_gnn_roc_auc_metric_fn("target"),
        )
    ]

    @staticmethod
    def get_data_types(path_prefix):
        pokec_col_ranges = [[0, 1] for _ in range(65)]
        pokec_col_ranges[1] = [0, 100]
        pokec_col_ranges[13] = [13, 112]
        return {
            "node_id": bolt.types.node_id(),
            **{
                f"col_{col_name}": bolt.types.numerical(col_range)
                for col_name, col_range in enumerate(pokec_col_ranges)
            },
            "target": bolt.types.categorical(n_classes=2, type="int"),
            "neighbors": bolt.types.neighbors(),
        }

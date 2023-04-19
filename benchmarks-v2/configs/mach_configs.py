from thirdai import bolt

from .udt_configs import UDTBenchmarkConfig
from .utils import (
    AdditionalMetricCallback,
    get_mach_precision_at_k_metric_fn,
    get_mach_recall_at_k_metric_fn,
)


class ScifactMachUDTBenchmark(UDTBenchmarkConfig):
    config_name = "scifact_mach_udt"
    dataset_name = "scifact"

    train_file = "scifact/trn_supervised.csv"
    test_file = "scifact/tst_supervised.csv"

    target = "DOC_ID"
    integer_target = True
    n_target_classes = 5183
    delimiter = ","

    num_epochs = 10
    learning_rate = 0.001
    options = {"extreme_classification": True, "embedding_dimension": 1024}
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="precision@1",
            metric_fn=get_mach_precision_at_k_metric_fn(
                "DOC_ID", k=1, target_delimeter=":"
            ),
        ),
        AdditionalMetricCallback(
            metric_name="recall@5",
            metric_fn=get_mach_recall_at_k_metric_fn(
                "DOC_ID", k=5, target_delimeter=":"
            ),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        }

from abc import ABC, abstractmethod

from .utils import (
    AdditionalMetricCallback,
    get_qr_precision_at_k_metric_fn,
    get_qr_recall_at_k_metric_fn,
)


class QueryReformulationBenchmarkConfig(ABC):
    config_name = None
    dataset_name = None

    train_file = None
    test_file = None

    source_column = None
    target_column = None
    dataset_size = None

    additional_metric_fns = {}
    model_config = None
    options = {}


class CQICQUDTBenchmark(QueryReformulationBenchmarkConfig):
    config_name = "cq_icq_query_reformulation"
    dataset_name = "cq_icq_0.6"

    train_file = "query_reformulation/train_udt.csv"
    test_file = "query_reformulation/test_udt.csv"

    source_column = "incorrect_queries"
    target_column = "correct_queries"
    dataset_size = "large"

    additional_metric_fns = {
        "recall@5": get_qr_recall_at_k_metric_fn("correct_queries", k=5),
        "precision@1": get_qr_precision_at_k_metric_fn("correct_queries", k=1),
    }

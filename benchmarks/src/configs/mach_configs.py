from thirdai import bolt

from .udt_configs import UDTBenchmarkConfig


class ScifactMachUDTBenchmark(UDTBenchmarkConfig):
    config_name = "scifact_mach_udt"
    dataset_name = "scifact"

    train_file = "scifact/trn_supervised.csv"
    test_file = "scifact/tst_supervised.csv"

    target = "DOC_ID"
    delimiter = ","

    num_epochs = 10
    learning_rate = 0.001
    options = {
        "extreme_classification": True,
        "embedding_dimension": 1024,
        "extreme_output_dim": 1000,
    }
    metrics = ["precision@1", "recall@5"]
    callbacks = []

    cold_start_learning_rate = 0.001
    cold_start_num_epochs = 5
    cold_start_train_file = "scifact/unsupervised.csv"
    strong_column_names = ["TITLE"]
    weak_column_names = ["TEXT"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":", n_classes=5183, type="int"),
        }


class TrecCovidMachUDTBenchmark(UDTBenchmarkConfig):
    config_name = "trec_covid_mach_udt"
    dataset_name = "trec-covid"

    test_file = "trec-covid/tst_supervised.csv"

    target = "DOC_ID"
    delimiter = ","

    options = {
        "extreme_classification": True,
        "embedding_dimension": 3000,
        "extreme_output_dim": 20000,
    }
    metrics = ["precision@1", "precision@10"]
    callbacks = []

    cold_start_learning_rate = 0.001
    cold_start_num_epochs = 5
    cold_start_train_file = "trec-covid/unsupervised.csv"
    strong_column_names = ["TITLE"]
    weak_column_names = ["TEXT"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "QUERY": bolt.types.text(
                tokenizer="words-punct", contextual_encoding="local"
            ),
            "DOC_ID": bolt.types.categorical(
                delimiter=":", n_classes=171332, type="int"
            ),
        }

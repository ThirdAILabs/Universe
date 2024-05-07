from thirdai import bolt

from .udt_configs import UDTBenchmarkConfig


class ScifactColdStartUDTBenchmark(UDTBenchmarkConfig):
    config_name = "scifact_cold_start_udt"
    dataset_name = "scifact"

    cold_start_train_file = "scifact/unsupervised.csv"
    test_file = "scifact/tst_supervised.csv"

    target = "DOC_ID"
    integer_target = True
    n_target_classes = 5183
    delimiter = ","

    metrics = ["precision@1", "recall@5"]
    cold_start_num_epochs = 5
    cold_start_learning_rate = 0.001
    strong_column_names = ["TITLE"]
    weak_column_names = ["TEXT"]
    options = {"embedding_dimension": 1024}

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        }


class CookingColdStartUDTBenchmark(UDTBenchmarkConfig):
    config_name = "cooking_cold_start_udt"
    dataset_name = "cooking"

    cold_start_train_file = (
        "catalog_recommender/cooking/reformatted_trn_unsupervised.csv"
    )
    test_file = "catalog_recommender/cooking/reformatted_tst_supervised.csv"

    target = "LABEL_IDS"
    integer_target = True
    n_target_classes = 26109
    delimiter = ","
    model_config = {
        "inputs": ["input"],
        "nodes": [
            {
                "name": "emb",
                "type": "embedding",
                "dim": 256,
                "activation": "relu",
                "predecessor": "input",
            },
            {
                "name": "fc",
                "type": "fully_connected",
                "dim": 26109,
                "sparsity": 0.02,
                "activation": "sigmoid",
                "predecessor": "emb",
            },
        ],
        "output": "fc",
        "loss": "BinaryCrossEntropyLoss",
    }

    metrics = ["precision@1", "recall@100"]
    cold_start_num_epochs = 15
    cold_start_learning_rate = 0.001
    strong_column_names = []
    weak_column_names = ["DESCRIPTION", "BRAND"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "LABEL_IDS": bolt.types.categorical(delimiter=";"),
            "QUERY": bolt.types.text(contextual_encoding="local"),
        }

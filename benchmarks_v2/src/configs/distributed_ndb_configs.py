from abc import ABC


class DistributedNDBConfig(ABC):
    config_name = None
    dataset_name = None
    ray_checkpoint_storage = "ray_checkpoint/"


class Amazon200kConfig(DistributedNDBConfig):
    config_name = "amazon_200K_ndb"
    dataset_name = "amazon_200K"
    ray_config_path = "testing_distributed_NDB_with_amazon-200K"

    learning_rate = 0.005
    epochs = 20
    metrics = ["loss", "hash_precision@1"]
    batch_size = 50000

    doc_path = "dist_ndb/amazon_200K_title_id.csv"
    doc_id_column = "id"
    doc_strong_columns = ["TITLE"]
    doc_weak_columns = ["TITLE"]
    doc_reference_columns = ["TITLE", "id"]

from abc import ABC


class DistributedNDBConfig(ABC):
    config_name = None
    dataset_name = None
    ray_checkpoint_storage = "ray_checkpoint/"


class Amazon_200K_Config(DistributedNDBConfig):
    config_name = "amazon_200K_ndb"
    dataset_name = "amazon_200K"

    learning_rate = 0.005
    epochs = 5
    metrics = ["precision@1", "loss", "hash_precision@1"]
    batch_size = 20000
    max_in_memory_batches = 10

    doc_path = "dist_ndb/amazon_200K_title_id.csv"
    doc_id_column = "id"
    doc_strong_columns = ["TITLE"]
    doc_weak_columns = ["TITLE"]
    doc_reference_columns = ["TITLE", "id"]

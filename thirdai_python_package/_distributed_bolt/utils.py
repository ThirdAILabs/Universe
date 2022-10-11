import logging

from thirdai._thirdai import bolt, dataset
from thirdai.dataset import FixedVocabulary

def load_training_data(data_loader_config, batch_size, id):
    data_loader_type = data_loader_config["datasets"]["data_loader"]
    if data_loader_type == "svm":
        train_data, train_labels = dataset.load_bolt_svm_dataset(
            data_loader_config["datasets"]["train_file_name"][id],
            batch_size,
        )
        if data_loader_config["datasets"]["to_validate"] == 1:
            valid_data, valid_labels = dataset.load_bolt_svm_dataset(data_loader_config["validation"]["file_name"], batch_size)
            return train_data, train_labels, valid_data, valid_labels
        return train_data, train_labels

    
    elif data_loader_type == "mlm":
        vocab = FixedVocabulary.make(data_loader_config["datasets"]["vocab_path"])
        mlm_loader = dataset.MLMDatasetLoader(vocab, data_loader_config["datasets"]["pairgram_range"])

        train_data , _ , train_labels  = mlm_loader.load(data_loader_config["datasets"]["train_file_name"][id], batch_size)
        if data_loader_config["datasets"]["to_validate"] == 1:
            valid_data , _ , valid_labels = mlm_loader.load(data_loader_config["validation"]["file_name"], batch_size)
            return train_data, train_labels, valid_data, valid_labels
        
        return train_data, train_labels

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

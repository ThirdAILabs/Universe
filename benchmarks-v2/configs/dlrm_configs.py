from abc import ABC, abstractmethod

from thirdai import bolt, dataset


class DLRMConfig(ABC):
    config_name = None
    dataset_name = None

    learning_rate = None
    num_epochs = None
    delimiter = None
    metrics = ["categorical_accuracy"]
    compute_roc_auc = True

    train_dataset_path = None
    test_dataset_path = None

    @staticmethod
    @abstractmethod
    def get_model():
        pass

    @staticmethod
    @abstractmethod
    def load_datasets(path: str):
        pass


class CriteoDLRMConfig(DLRMConfig):
    config_name = "bolt_criteo_46m"
    dataset_name = "criteo_46m"

    def get_model():
        int_input = bolt.nn.Input(dim=13)
        hidden1 = bolt.nn.FullyConnected(dim=32, activation="relu")(int_input)

        cat_input = bolt.nn.TokenInput(dim=4294967295, num_tokens_range=(26, 26))

        embedding = bolt.nn.Embedding(
            num_embedding_lookups=4,
            lookup_size=8,
            log_embedding_block_size=20,
            reduction="concat",
            num_tokens_per_input=26,
        )(cat_input)

        feature_interaction = bolt.nn.DlrmAttention()(
            fc_layer=hidden1, embedding_layer=embedding
        )

        concat = bolt.nn.Concatenate()([hidden1, feature_interaction])

        hidden_output = concat
        for _ in range(3):
            hidden_output = bolt.nn.FullyConnected(
                dim=512,
                sparsity=0.4,
                activation="relu",
                sampling_config=bolt.nn.RandomSamplingConfig(),
            )(hidden_output)

        output = bolt.nn.FullyConnected(dim=2, activation="softmax")(hidden_output)

        model = bolt.nn.Model(inputs=[int_input, cat_input], output=output)
        model.compile(bolt.nn.losses.CategoricalCrossEntropy())

        return model

    learning_rate = 1e-4
    num_epochs = 1
    delimiter = " "

    train_dataset_path = "criteo/train_shuf.txt"
    test_dataset_path = "criteo/test_shuf.txt"

    def _load_click_through_dataset(
        filename,
        batch_size,
        max_num_numerical_features,
        max_categorical_features,
        delimiter=" ",
    ):
        bolt_dataset, bolt_token_dataset, labels = dataset.load_click_through_dataset(
            filename=filename,
            batch_size=batch_size,
            max_num_numerical_features=max_num_numerical_features,
            max_categorical_features=max_categorical_features,
            delimiter=delimiter,
        )
        return bolt_dataset, bolt_token_dataset, labels

    def load_datasets(path: str):
        max_num_categorical_features = 26
        num_numerical_features = 13
        batch_size = 512
        (
            train_bolt_dataset,
            train_bolt_token_dataset,
            train_labels,
        ) = CriteoDLRMConfig._load_click_through_dataset(
            filename=path + CriteoDLRMConfig.train_dataset_path,
            batch_size=batch_size,
            max_categorical_features=max_num_categorical_features,
            max_num_numerical_features=num_numerical_features,
        )
        (
            test_bolt_dataset,
            test_bolt_token_dataset,
            test_labels,
        ) = CriteoDLRMConfig._load_click_through_dataset(
            filename=path + CriteoDLRMConfig.test_dataset_path,
            batch_size=batch_size,
            max_categorical_features=max_num_categorical_features,
            max_num_numerical_features=num_numerical_features,
        )
        train_data = [train_bolt_dataset, train_bolt_token_dataset]
        test_data = [test_bolt_dataset, test_bolt_token_dataset]

        return train_data, train_labels, test_data, test_labels

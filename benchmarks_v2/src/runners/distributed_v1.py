from thirdai import bolt, bolt_v2

from ..configs.distributed_configs import DistributedBenchmarkConfig
from .runner import Runner


def create_udt_model(n_target_classes, output_dim, num_hashes):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
        options={
            "extreme_classification": True,
            "train_without_bias": True,
            "embedding_dimension": 2048,
            "freeze_hash_tables": False,
            "extreme_output_dim": output_dim,
            "extreme_num_hashes": num_hashes,
        },
    )
    model._get_model().summary()

    return model


class DistributedRunner(Runner):
    config_type = DistributedBenchmarkConfig

    def run_benchmark(config: DistributedBenchmarkConfig, path_prefix, mlflow_logger):
        model = create_udt_model(
            n_target_classes=config.n_target_classes,
            output_dim=config.output_dim,
            num_hashes=config.num_hashes,
        )

        validation = bolt.Validation(
            config.supervised_tst,
            interval=5000,
            metrics=config.val_metrics,
        )

        if hasattr(config, "supervised_trn"):
            model.train(
                filename=config.supervised_trn,
                learning_rate=config.learning_rate,
                epochs=config.num_epochs,
                metrics=config.train_metrics,
                validation=validation,
                # callbacks=[LoggingCallback(model, supervised_tst)],
            )

        if hasattr(config, "unsupervised_file"):
            model.cold_start(
                filename=config.unsupervised_file,
                strong_column_names=["TITLE"],
                weak_column_names=["TEXT"],
                learning_rate=config.learning_rate,
                epochs=20,
                metrics=[
                    "precision@1",
                    "recall@10",
                ],
                # callbacks=[LoggingCallback(model, supervised_tst)],
                validation=validation,
            )
        pass

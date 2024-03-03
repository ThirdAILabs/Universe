from thirdai import bolt
import mlflow
import os
import thirdai

# thirdai.licensing.activate("8A5483-502DCE-DEC601-084409-BCA046-V3")

model = bolt.UniversalDeepTransformer(
    data_types={
        "QUERY": bolt.types.text(tokenizer="char-4"),
        "DOC_ID": bolt.types.categorical(delimiter=":"),
    },
    target="DOC_ID",
    n_target_classes=105000,
    integer_target=True,
    options={
        "input_dim": 100000,
        "embedding_dimension": 2000,
        "extreme_classification": True,
        "extreme_num_hashes": 1,
        "extreme_output_dim": 50000,
        "hidden_bias": False,
        "output_bias": False,
        "softmax": True,
        "n_models": 5,
    },
)


class Logger(bolt.train.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.step = 0

    def before_update(self):
        norms = self.model.norms()
        mlflow.log_metric(
            key="emb.weight.grad",
            value=norms["emb_1_embeddings_grad_l2_norm"],
            step=self.step,
        )
        mlflow.log_metric(
            key="output.weight.grad",
            value=norms["fc_1_weight_grad_l2_norm"],
            step=self.step,
        )
        self.step += 1


logger = Logger()

mlflow.set_tracking_uri(
    "http://thirdai:rwRp8GBNVplnLCIveMIz@ec2-184-73-150-35.compute-1.amazonaws.com"
)
mlflow.set_experiment("multi-mach")
mlflow.start_run(run_name="1-hash-5-models-grad-tracking-converted")

for _ in range(20):
    metrics = model.cold_start(
        "/share/data/semantic_benchmarks/wiki/unsupervised_large.csv",
        strong_column_names=[],
        weak_column_names=["TEXT"],
        learning_rate=1e-3,
        epochs=1,
        metrics=["loss"],
        batch_size=10000,
        callbacks=[logger],
    )
    mlflow.log_metric("train_loss", metrics["train_loss"][-1], step=logger.step)

    for file in ["len_5", "len_10", "len_15", "len_20"]:
        metrics = model.evaluate(
            f"/share/data/semantic_benchmarks/wiki/{file}.csv", metrics=["precision@1"]
        )

        mlflow.log_metric(
            f"{file}_prec_at_1", metrics["val_precision@1"][-1], step=logger.step
        )

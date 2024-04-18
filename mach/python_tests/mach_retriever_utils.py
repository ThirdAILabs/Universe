import os

from thirdai import bolt, data

QUERY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../auto_ml/python_tests/texts.csv"
)


def train_simple_mach_retriever():
    model = (
        bolt.MachConfig()
        .text_col("text")
        .id_col("id")
        .tokenizer("words")
        .contextual_encoding("none")
        .emb_dim(512)
        .n_buckets(10000)
        .emb_bias()
        .output_bias()
        .output_activation("sigmoid")
        .build()
    )

    model.train(QUERY_FILE, learning_rate=1e-3, epochs=5, metrics=["precision@1"])

    model.evaluate(QUERY_FILE, metrics=["precision@1"])

    return model

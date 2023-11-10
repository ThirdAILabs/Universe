import torch
import torch.nn as nn
from thirdai import bolt, data, dataset
import time


class Precision:
    def __init__(self, k):
        self.k = k
        self.total_score = 0
        self.num_samples = 0

    def record(self, predictions, labels):
        score = 0
        for pred, _ in predictions[: self.k]:
            if pred in labels:
                score += 1
        # score /= self.k
        # self.total_score += score
        # self.num_samples += 1
        self.total_score += score
        self.num_samples += self.k

    def name(self):
        return f"precision_at_{self.k}"

    def value(self):
        return self.total_score / self.num_samples


class Recall:
    def __init__(self, k):
        self.k = k
        self.total_score = 0
        self.num_samples = 0

    def record(self, predictions, labels):
        score = 0
        for pred, _ in predictions[: self.k]:
            if pred in labels:
                score += 1
        # if len(labels) == 0:
        #     score = 0
        # else:
        #     score /= min(self.k, len(labels))
        # self.total_score += score
        # self.num_samples += 1
        self.total_score += score
        self.num_samples += min(self.k, len(labels))

    def name(self):
        return f"recall_at_{self.k}"

    def value(self):
        return self.total_score / self.num_samples


def bolt_model(input_dim, emb_dim, output_dim):
    input_ = bolt.nn.Input(input_dim)
    emb = bolt.nn.Embedding(input_dim=input_dim, dim=emb_dim, activation="relu")(input_)
    out = bolt.nn.FullyConnected(
        input_dim=emb_dim, dim=output_dim, sparsity=0.1, activation="sigmoid"
    )(emb)

    loss = bolt.nn.losses.BinaryCrossEntropy(out, bolt.nn.Input(output_dim))

    return bolt.nn.Model(inputs=[input_], outputs=[out], losses=[loss])


class Mach:
    entity_col = "__entities__"
    token_col = "__tokens__"
    mach_label_col = "__mach_labels__"

    def __init__(
        self,
        input_dim,
        emb_dim,
        n_buckets,
        n_entities,
        lr=1e-3,
        n_hashes=7,
        text_col="QUERY",
        label_col="DOC_ID",
        char_4_grams=False,
        csv_delimiter=",",
        label_delimiter=":",
    ):
        self.model = bolt_model(input_dim, emb_dim, n_buckets)

        self.index = dataset.MachIndex(
            output_range=n_buckets, num_hashes=n_hashes, num_elements=n_entities
        )

        self.input_dim = input_dim

        self.lr = lr
        self.text_col = text_col
        self.label_col = label_col
        self.char_4_grams = char_4_grams
        self.csv_delimiter = csv_delimiter
        self.label_delimiter = label_delimiter

    def _input_dim(self):
        return self.input_dim

    def _output_dim(self):
        return self.index.output_range()

    def _text_transform(self):
        extra_args = (
            {"tokenizer": dataset.CharKGramTokenizer(k=4)} if self.char_4_grams else {}
        )
        return data.transformations.Text(
            input_column=self.text_col,
            output_indices=self.token_col,
            encoder=dataset.NGramEncoder(n=2),
            **extra_args,
            dim=self._input_dim(),
            lowercase=True,
        )

    def _entity_parse_transform(self):
        return data.transformations.ToTokenArrays(
            input_column=self.label_col,
            output_column=self.entity_col,
            delimiter=self.label_delimiter,
        )

    def _build_data_pipeline(self, strong_cols=None, weak_cols=None):
        pipeline = data.transformations.Pipeline()

        if weak_cols or strong_cols:
            pipeline = pipeline.then(
                data.transformations.ColdStartText(
                    strong_columns=strong_cols,
                    weak_columns=weak_cols,
                    label_column=self.label_col,
                    output_column=self.text_col,
                )
            )

        pipeline = (
            pipeline.then(self._text_transform())
            .then(self._entity_parse_transform())
            .then(
                data.transformations.MachLabel(
                    input_column=self.entity_col,
                    output_column=self.mach_label_col,
                )
            )
        )

        return pipeline

    def _load_data(self, filename, pipeline, batch_size):
        data_iter = data.CsvIterator(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )

        state = data.transformations.State(self.index)

        loader = data.Loader(
            data_iter,
            pipeline,
            state,
            input_columns=[data.OutputColumns(self.token_col)],
            output_columns=[data.OutputColumns(self.mach_label_col)],
            batch_size=batch_size,
        )

        return loader.all()

    def train(self, filename, batch_size=2048, strong_cols=None, weak_cols=None):
        pipeline = self._build_data_pipeline(
            strong_cols=strong_cols, weak_cols=weak_cols
        )

        # batches = zip(*self._load_data(filename, pipeline, batch_size))
        batches = self._load_data(filename, pipeline, batch_size)

        trainer = bolt.train.Trainer(self.model)

        trainer.train(
            train_data=batches,
            learning_rate=self.lr,
            epochs=1,
            autotune_rehash_rebuild=True,
        )
        # start = time.perf_counter()
        # for inputs, labels in batches:
        #     self.model.train_on_batch(inputs, labels)
        #     self.model.update_parameters(self.lr)

        # end = time.perf_counter()
        # print(f"epoch complete - time={round(end -start, 4)}")

    def validate(self, filename, recall_at, precision_at, num_buckets_to_eval=25):
        columns = data.CsvIterator.all(
            dataset.FileDataSource(filename),
            delimiter=self.csv_delimiter,
        )
        columns = self._text_transform()(columns)
        columns = self._entity_parse_transform()(columns)

        batch_size = 10_000
        inputs = data.to_tensors(
            columns, [data.OutputColumns(self.token_col)], batch_size=batch_size
        )
        labels = []
        for i in range(0, len(columns), batch_size):
            labels.append(columns[self.entity_col].data()[i : i + batch_size])

        top_k = max(max(recall_at), max(precision_at))

        metrics = [Recall(k) for k in recall_at] + [Precision(k) for k in precision_at]

        for inputs, batch_labels in zip(inputs, labels):
            out = self.model.forward(inputs)[0].activations

            predictions = self.index.decode_batch(
                out, top_k=top_k, num_buckets_to_eval=num_buckets_to_eval
            )

            for preds, lbls in zip(predictions, batch_labels):
                for metric in metrics:
                    metric.record(preds, lbls)

        metric_vals = {}
        for metric in metrics:
            print(f"{metric.name()} = {metric.value()}")
            metric_vals[metric.name()] = metric.value()

        return metric_vals


def scifact():
    model = Mach(
        input_dim=100_000,
        emb_dim=1024,
        n_buckets=1_000,
        n_entities=5183,
        char_4_grams=False,
        lr=0.001,
    )

    for e in range(5):
        print("\nCold Start")
        model.train(
            "/Users/nmeisburger/ThirdAI/data/scifact/unsupervised.csv",
            strong_cols=["TITLE"],
            weak_cols=["TEXT"],
        )

        if e == 0:
            model.model.freeze_hash_tables()

        model.validate(
            "/Users/nmeisburger/ThirdAI/data/scifact/tst_supervised.csv",
            recall_at=[5],
            precision_at=[1],
        )

    for _ in range(10):
        print("\nSupervised")
        model.train(
            "/Users/nmeisburger/ThirdAI/data/scifact/trn_supervised.csv",
        )
        model.validate(
            "/Users/nmeisburger/ThirdAI/data/scifact/tst_supervised.csv",
            recall_at=[5],
            precision_at=[1],
        )


if __name__ == "__main__":
    scifact()

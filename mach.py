import time

import pandas as pd
from thirdai import bolt, data, dataset, smx


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
        # If each label is weighted equally use the following:
        self.total_score += score
        self.num_samples += self.k
        # If each sample is weighted equally use the following:
        # score /= self.k
        # self.total_score += score
        # self.num_samples += 1

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
        # If each label is weighted equally use the following:
        self.total_score += score
        self.num_samples += min(self.k, len(labels))
        # If each sample is weighted equally use the following:
        # if len(labels) == 0:
        #     score = 0
        # else:
        #     score /= min(self.k, len(labels))
        # self.total_score += score
        # self.num_samples += 1

    def name(self):
        return f"recall_at_{self.k}"

    def value(self):
        return self.total_score / self.num_samples


class MachModel(smx.Module):
    def __init__(self, input_dim, emb_dim, output_dim, sparsity):
        super().__init__()

        self.emb = smx.Embedding(emb_dim=emb_dim, n_embs=input_dim, reduce_mean=False)

        self.output = smx.SparseLinear(
            input_dim=emb_dim,
            dim=output_dim,
            sparsity=sparsity,
            updates_per_rebuild=2,
            updates_per_new_hash_fn=11,
        )

    def forward(self, x, y=None):
        out = smx.relu(self.emb(x))
        out = self.output(out, y)
        return out


class SmxMach:
    entity_col = "__entities__"
    token_col = "__tokens__"
    mach_label_col = "__mach_labels__"

    def __init__(
        self,
        input_dim,
        emb_dim,
        n_buckets,
        output_sparsity,
        n_entities,
        lr=1e-3,
        n_hashes=7,
        text_col="QUERY",
        label_col="DOC_ID",
        char_4_grams=False,
        csv_delimiter=",",
        label_delimiter=":",
    ):
        self.model = MachModel(
            input_dim=input_dim,
            emb_dim=emb_dim,
            output_dim=n_buckets,
            sparsity=output_sparsity,
        )
        self.optimizer = smx.optimizers.Adam(self.model.parameters(), lr=lr)
        self.optimizer.register_on_update_callback(
            self.model.output.on_update_callback()
        )

        self.index = dataset.MachIndex(
            output_range=n_buckets, num_hashes=n_hashes, num_elements=n_entities
        )

        self.n_embs = input_dim
        self.n_buckets = n_buckets
        self.text_col = text_col
        self.label_col = label_col
        self.char_4_grams = char_4_grams
        self.csv_delimiter = csv_delimiter
        self.label_delimiter = label_delimiter

    def input_dim(self):
        return self.n_embs

    def output_dim(self):
        return self.n_buckets

    def text_transform(self):
        extra_args = (
            {"tokenizer": dataset.CharKGramTokenizer(k=4)} if self.char_4_grams else {}
        )
        return data.transformations.Text(
            input_column=self.text_col,
            output_indices=self.token_col,
            encoder=dataset.NGramEncoder(n=2),
            **extra_args,
            dim=self.input_dim(),
            lowercase=True,
        )

    def entity_parse_transform(self):
        return data.transformations.ToTokenArrays(
            input_column=self.label_col,
            output_column=self.entity_col,
            delimiter=self.label_delimiter,
        )

    def mach_label_transform(self):
        return data.transformations.MachLabel(
            input_column=self.entity_col,
            output_column=self.mach_label_col,
        )

    def build_data_pipeline(self, strong_cols=[], weak_cols=[]):
        pipeline = data.transformations.Pipeline()

        if weak_cols or strong_cols:
            pipeline = pipeline.then(
                data.transformations.VariableLengthColdStart(
                    strong_columns=strong_cols,
                    weak_columns=weak_cols,
                    output_column=self.text_col,
                )
            )

        pipeline = (
            pipeline.then(self.text_transform())
            .then(self.entity_parse_transform())
            .then(self.mach_label_transform())
        )

        return pipeline

    def load_data(
        self, filename, batch_size, strong_cols=[], weak_cols=[], shuffle=True
    ):
        data_iter = data.CsvIterator(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )

        loader = data.Loader(
            data_iterator=data_iter,
            transformation=self.build_data_pipeline(
                strong_cols=strong_cols, weak_cols=weak_cols
            ),
            state=data.transformations.State(self.index),
            input_columns=[data.OutputColumns(self.token_col)],
            output_columns=[data.OutputColumns(self.mach_label_col)],
            batch_size=batch_size,
            shuffle=shuffle,
        )

        inputs, labels = loader.all_smx()

        return list(
            zip(
                [smx.Variable(x[0], requires_grad=False) for x in inputs],
                [smx.Variable(y[0], requires_grad=False) for y in labels],
            )
        )

    def train(self, filename, batch_size=2048, strong_cols=[], weak_cols=[]):
        self.model.train()

        batches = self.load_data(
            filename, batch_size, strong_cols=strong_cols, weak_cols=weak_cols
        )

        start = time.perf_counter()
        for tokens, labels in batches:
            self.optimizer.zero_grad()

            out = self.model(tokens, labels)
            loss = smx.binary_cross_entropy(out, labels.tensor)
            loss.backward()

            self.optimizer.step()

        end = time.perf_counter()
        print(
            f"epoch complete - train_loss={loss.tensor.scalar():.4f} - time={end-start:.4f}"
        )

    def validate(self, filename, recall_at=[], precision_at=[], num_buckets_to_eval=25):
        self.model.eval()

        batch_size = 10000
        batches = self.load_data(filename, batch_size=batch_size, shuffle=False)

        df = pd.read_csv(filename)
        labels = (
            df[self.label_col]
            .map(lambda x: list(map(int, x.split(self.label_delimiter))))
            .to_list()
        )

        label_batches = []
        for i in range(0, len(labels), batch_size):
            label_batches.append(labels[i : i + batch_size])

        top_k = max(recall_at + precision_at)

        metrics = [Recall(k) for k in recall_at] + [Precision(k) for k in precision_at]

        for (tokens, _), labels in zip(batches, label_batches):
            out = smx.sigmoid(self.model(tokens))

            predictions = self.index.decode_batch(
                out.tensor.numpy(),
                top_k=top_k,
                num_buckets_to_eval=num_buckets_to_eval,
            )

            for sample_preds, sample_labels in zip(predictions, labels):
                for metric in metrics:
                    metric.record(sample_preds, sample_labels)

        metric_vals = {}
        for metric in metrics:
            print(f"{metric.name()} = {metric.value():.4f}")
            metric_vals[metric.name()] = metric.value()

        return metric_vals

    def freeze_hash_tables(self):
        self.model.output.neuron_index.freeze()


class BoltMach:
    entity_col = "__entities__"
    token_col = "__tokens__"
    mach_label_col = "__mach_labels__"

    def __init__(
        self,
        input_dim,
        emb_dim,
        n_buckets,
        output_sparsity,
        n_entities,
        lr=1e-3,
        n_hashes=7,
        text_col="QUERY",
        label_col="DOC_ID",
        char_4_grams=False,
        csv_delimiter=",",
        label_delimiter=":",
    ):
        self.model = self.build_model(input_dim, emb_dim, n_buckets, output_sparsity)
        self.model.disable_sparse_parameter_updates()

        self.index = dataset.MachIndex(
            output_range=n_buckets, num_hashes=n_hashes, num_elements=n_entities
        )

        self.n_embs = input_dim

        self.lr = lr
        self.text_col = text_col
        self.label_col = label_col
        self.char_4_grams = char_4_grams
        self.csv_delimiter = csv_delimiter
        self.label_delimiter = label_delimiter

    def build_model(self, input_dim, emb_dim, output_dim, output_sparsity):
        input_ = bolt.nn.Input(input_dim)
        emb = bolt.nn.Embedding(
            input_dim=input_dim, dim=emb_dim, activation="relu", bias=False
        )(input_)
        out = bolt.nn.FullyConnected(
            input_dim=emb_dim,
            dim=output_dim,
            sparsity=output_sparsity,
            rebuild_hash_tables=2,
            reconstruct_hash_functions=11,
            activation="sigmoid",
        )(emb)

        loss = bolt.nn.losses.BinaryCrossEntropy(out, bolt.nn.Input(output_dim))

        return bolt.nn.Model(inputs=[input_], outputs=[out], losses=[loss])

    def input_dim(self):
        return self.n_embs

    def output_dim(self):
        return self.index.output_range()

    def text_transform(self):
        extra_args = (
            {"tokenizer": dataset.CharKGramTokenizer(k=4)} if self.char_4_grams else {}
        )
        return data.transformations.Text(
            input_column=self.text_col,
            output_indices=self.token_col,
            encoder=dataset.NGramEncoder(n=2),
            **extra_args,
            dim=self.input_dim(),
            lowercase=True,
        )

    def entity_parse_transform(self):
        return data.transformations.ToTokenArrays(
            input_column=self.label_col,
            output_column=self.entity_col,
            delimiter=self.label_delimiter,
        )

    def mach_label_transform(self):
        return data.transformations.MachLabel(
            input_column=self.entity_col,
            output_column=self.mach_label_col,
        )

    def build_data_pipeline(self, strong_cols=None, weak_cols=None):
        pipeline = data.transformations.Pipeline()

        if weak_cols or strong_cols:
            pipeline = pipeline.then(
                data.transformations.VariableLengthColdStart(
                    strong_columns=strong_cols,
                    weak_columns=weak_cols,
                    output_column=self.text_col,
                )
            )

        pipeline = (
            pipeline.then(self.text_transform())
            .then(self.entity_parse_transform())
            .then(self.mach_label_transform())
        )

        return pipeline

    def load_data(
        self, filename, batch_size, strong_cols=[], weak_cols=[], shuffle=True
    ):
        data_iter = data.CsvIterator(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )

        loader = data.Loader(
            data_iterator=data_iter,
            transformation=self.build_data_pipeline(
                strong_cols=strong_cols, weak_cols=weak_cols
            ),
            state=data.transformations.State(self.index),
            input_columns=[data.OutputColumns(self.token_col)],
            output_columns=[data.OutputColumns(self.mach_label_col)],
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return loader.all()

    def train(self, filename, batch_size=2048, strong_cols=None, weak_cols=None):

        batches = self.load_data(
            filename,
            batch_size,
            strong_cols=strong_cols,
            weak_cols=weak_cols,
            shuffle=True,
        )

        trainer = bolt.train.Trainer(self.model)

        trainer.train(
            train_data=batches,
            learning_rate=self.lr,
            epochs=1,
            autotune_rehash_rebuild=False,
        )

    def validate(self, filename, recall_at, precision_at, num_buckets_to_eval=25):
        batch_size = 10000
        batches, _ = self.load_data(filename, batch_size=batch_size, shuffle=False)

        df = pd.read_csv(filename)
        labels = (
            df[self.label_col]
            .map(lambda x: list(map(int, x.split(self.label_delimiter))))
            .to_list()
        )

        label_batches = []
        for i in range(0, len(labels), batch_size):
            label_batches.append(labels[i : i + batch_size])

        top_k = max(recall_at + precision_at)

        metrics = [Recall(k) for k in recall_at] + [Precision(k) for k in precision_at]

        for inputs, labels in zip(batches, label_batches):
            out = self.model.forward(inputs)[0].activations

            predictions = self.index.decode_batch(
                out,
                top_k=top_k,
                num_buckets_to_eval=num_buckets_to_eval,
            )

            for sample_preds, sample_labels in zip(predictions, labels):
                for metric in metrics:
                    metric.record(sample_preds, sample_labels)

        metric_vals = {}
        for metric in metrics:
            print(f"{metric.name()} = {metric.value():.4f}")
            metric_vals[metric.name()] = metric.value()

        return metric_vals

    def freeze_hash_tables(self):
        self.model.freeze_hash_tables()


def scifact():
    model = SmxMach(
        input_dim=100_000,
        emb_dim=1024,
        n_buckets=1_000,
        output_sparsity=0.1,
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

        # if e == 0:
        #     # Need to add insert when not found option to smx.
        #     model.freeze_hash_tables()

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

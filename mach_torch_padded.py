import os
import time

import torch
import torch.nn as nn
import tqdm
from thirdai import data, dataset


def to_padded_batches(rows, batch_size):
    batches = []
    for i in range(0, len(rows), batch_size):
        tokens = torch.nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(row) for row in rows[i : i + batch_size]],
            batch_first=True,
            padding_value=0,
        )
        batches.append(tokens)
    return batches


def to_csr(rows, batch_size, dense_dim):
    batches = []
    for i in range(0, len(rows), batch_size):
        indices = []
        values = []
        offsets = [0]

        for row in rows[i : i + batch_size]:
            indices.extend(row)
            values.extend([1.0] * len(row))
            offsets.append(len(indices))

        batches.append(
            torch.sparse_csr_tensor(
                crow_indices=offsets,
                col_indices=indices,
                values=values,
                size=(len(offsets) - 1, dense_dim),
                dtype=torch.float32,
            )
        )

    return batches


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


class QuantileMachModel(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim):
        super().__init__()

        self.emb = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim,
            padding_idx=0,
        )
        self.emb_bias = nn.Parameter(torch.empty(emb_dim))
        nn.init.normal_(self.emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.emb_bias, mean=0, std=0.01)

        self.output = nn.Linear(in_features=emb_dim, out_features=output_dim)
        nn.init.normal_(self.output.weight, mean=0, std=0.01)
        nn.init.normal_(self.output.bias, mean=0, std=0.01)

    def forward(self, tokens):
        out = self.emb(input=tokens)
        qs = torch.quantile(out, 0.9, dim=1, keepdims=True)
        out = torch.where(out >= qs, out, 0)
        out = torch.sum(out, dim=1)
        out += self.emb_bias
        # out = torch.nn.functional.relu(out)
        out = self.output(out)
        return out


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
        softmax=True,
        lr=1e-3,
        n_hashes=7,
        text_col="QUERY",
        label_col="DOC_ID",
        char_4_grams=False,
        csv_delimiter=",",
        label_delimiter=":",
    ):
        self.model = QuantileMachModel(
            input_dim=input_dim, emb_dim=emb_dim, output_dim=n_buckets
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-7)

        self.index = dataset.MachIndex(
            output_range=n_buckets, num_hashes=n_hashes, num_elements=n_entities
        )

        self.softmax = softmax

        self.text_col = text_col
        self.label_col = label_col
        self.char_4_grams = char_4_grams
        self.csv_delimiter = csv_delimiter
        self.label_delimiter = label_delimiter

    def _input_dim(self):
        return self.model.emb.num_embeddings

    def _output_dim(self):
        return self.model.output.out_features

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
        columns = data.CsvIterator.all(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )

        state = data.transformations.State(self.index)
        columns = pipeline(columns, state=state)
        columns.shuffle()

        inputs = to_padded_batches(
            columns[self.token_col].data(),
            batch_size=batch_size,
        )
        labels = to_csr(
            columns[self.mach_label_col].data(),
            batch_size=batch_size,
            dense_dim=self._output_dim(),
        )

        return list(zip(inputs, labels))

    def train(self, filename, batch_size=2048, strong_cols=None, weak_cols=None):
        self.model.train()

        pipeline = self._build_data_pipeline(
            strong_cols=strong_cols, weak_cols=weak_cols
        )

        batches = self._load_data(filename, pipeline, batch_size)

        if self.softmax:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        start = time.perf_counter()
        for tokens, labels in tqdm.tqdm(batches):
            self.optimizer.zero_grad()

            out = self.model(tokens)
            loss = loss_fn(out, labels.to_dense())
            loss.backward()

            # nn.utils.clip_grad.clip_grad_norm_(
            #     self.model.parameters(), max_norm=0.1, norm_type=2
            # )

            self.optimizer.step()

        end = time.perf_counter()
        print(
            f"epoch complete - train_loss={round(loss.item(), 4)} - time={round(end -start, 4)}"
        )

        with torch.no_grad():
            all_embs = self.model.emb(batches[0][0])
            qs = torch.quantile(all_embs, 0.9, dim=1, keepdims=True)
            masked = torch.where(all_embs >= qs, all_embs, 0)
            q_emb = torch.sum(masked, dim=1)
            reg_emb = torch.sum(all_embs, dim=1)

            dots = torch.sum(torch.mul(q_emb, reg_emb), dim=1)
            qmag = torch.norm(q_emb, p=2, dim=1)
            rmag = torch.norm(reg_emb, p=2, dim=1)

            cos_sims = dots / (qmag * rmag)

            print("cos sim: ", torch.mean(cos_sims))
            print("cos sim: ", torch.min(cos_sims))
            print("cos sim: ", torch.max(cos_sims))
            print("cos sim: ", torch.var(cos_sims))

    def validate(self, filename, recall_at=[], precision_at=[], num_buckets_to_eval=25):
        self.model.eval()

        columns = data.CsvIterator.all(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )
        columns = self._text_transform()(columns)
        columns = self._entity_parse_transform()(columns)

        batch_size = 10_000
        inputs = to_padded_batches(columns[self.token_col].data(), batch_size)
        label_batches = []
        for i in range(0, len(columns), batch_size):
            label_batches.append(columns[self.entity_col].data()[i : i + batch_size])

        top_k = max(recall_at + precision_at)

        metrics = [Recall(k) for k in recall_at] + [Precision(k) for k in precision_at]

        for tokens, labels in zip(inputs, label_batches):
            if self.softmax:
                out = nn.functional.softmax(self.model(tokens), dim=1)
            else:
                out = nn.functional.sigmoid(self.model(tokens))

            predictions = self.index.decode_batch(
                out.detach().numpy(),
                top_k=top_k,
                num_buckets_to_eval=num_buckets_to_eval,
            )

            for sample_preds, sample_labels in zip(predictions, labels):
                for metric in metrics:
                    metric.record(sample_preds, sample_labels)

        metric_vals = {}
        for metric in metrics:
            print(f"{metric.name()} = {metric.value()}")
            metric_vals[metric.name()] = metric.value()

        return metric_vals


DATA_DIR = "/Users/nmeisburger/ThirdAI/data"


def scifact(softmax=True):
    model = Mach(
        input_dim=100_000,
        emb_dim=1024,
        n_buckets=1_000,
        n_entities=5183,
        char_4_grams=False,
        lr=0.005 if softmax else 0.01,
        softmax=softmax,
    )

    for _ in range(5):
        print("\nCold Start")
        model.train(
            os.path.join(DATA_DIR, "scifact/unsupervised.csv"),
            strong_cols=["TITLE"],
            weak_cols=["TEXT"],
        )

        model.validate(
            os.path.join(DATA_DIR, "scifact/tst_supervised.csv"),
            recall_at=[5],
            precision_at=[1],
        )

    for _ in range(10):
        print("\nSupervised")
        model.train(
            os.path.join(DATA_DIR, "scifact/trn_supervised.csv"),
        )
        model.validate(
            os.path.join(DATA_DIR, "scifact/tst_supervised.csv"),
            recall_at=[5],
            precision_at=[1],
        )


def wiki_5k():
    model = Mach(
        input_dim=100_000,
        emb_dim=1000,
        n_buckets=10_000,
        n_entities=5000,
        char_4_grams=True,
        lr=0.005,
        softmax=True,
    )

    for _ in range(5):
        print("\nCold Start")
        model.train(
            os.path.join(DATA_DIR, "neuraldb_wiki_benchmark/unsupervised.csv"),
            strong_cols=[],
            weak_cols=["TEXT"],
        )

        model.validate(
            os.path.join(
                DATA_DIR, "neuraldb_wiki_benchmark/tst_supervised_cleaned.csv"
            ),
            precision_at=[1, 10],
        )


if __name__ == "__main__":
    # scifact(True)
    wiki_5k()

import torch
import torch.nn as nn
from thirdai import data, dataset
import time


def to_tokens_and_offsets(rows, batch_size):
    batches = []
    for i in range(0, len(rows), batch_size):
        tokens = []
        offsets = []
        for row in rows[i : i + batch_size]:
            offsets.append(len(tokens))
            tokens.extend(row)
        batches.append((torch.tensor(tokens), torch.tensor(offsets)))

    return batches


def to_padded_batch(rows, batch_size):
    batches = []
    for i in range(0, len(rows), batch_size):
        max_len = max(len(r) for r in rows[i : i + batch_size])
        tokens = [row + [0] * (max_len - len(row)) for row in rows[i : i + batch_size]]
        tokens = torch.tensor(tokens)
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
        for pred in predictions[: self.k]:
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
        for pred in predictions[: self.k]:
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


class Model(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim):
        super().__init__()

        # self.emb = nn.EmbeddingBag(num_embeddings=input_dim, embedding_dim=emb_dim)
        # self.emb_bias = nn.Parameter(torch.empty(emb_dim))
        # nn.init.normal_(self.emb_bias, mean=0, std=0.01)
        self.emb = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=emb_dim, padding_idx=0
        )

        self.dropout = nn.Dropout(p=0.1)

        self.output = nn.Linear(in_features=emb_dim, out_features=output_dim)

    def forward(self, tokens):
        out = self.emb(input=tokens)
        qs = torch.quantile(out, 0.9, dim=1, keepdims=True)
        out = torch.where(out >= qs, out, 0)
        out = torch.mean(out, dim=1)
        out = self.output(out)
        return out


class Classifier:
    def __init__(
        self,
        input_dim,
        emb_dim,
        n_classes,
        lr=1e-3,
        text_col="QUERY",
        label_col="DOC_ID",
        char_4_grams=False,
        csv_delimiter=",",
        label_delimiter=":",
    ):
        self.model = Model(input_dim=input_dim, emb_dim=emb_dim, output_dim=n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        return data.transformations.Text(
            input_column=self.text_col,
            output_indices=self.text_col,
            encoder=dataset.NGramEncoder(n=2),
            dim=self._input_dim(),
            lowercase=True,
        )

    def _entity_parse_transform(self):
        return data.transformations.ToTokenArrays(
            input_column=self.label_col,
            output_column=self.label_col,
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

        pipeline = pipeline.then(self._text_transform()).then(
            self._entity_parse_transform()
        )

        return pipeline

    def _load_data(self, filename, pipeline, batch_size):
        columns = data.CsvIterator.all(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )

        columns = pipeline(columns)
        columns.shuffle()

        inputs = to_padded_batch(
            columns[self.text_col].data(),
            batch_size=batch_size,
        )
        labels = to_csr(
            columns[self.label_col].data(),
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

        start = time.perf_counter()
        for tokens, labels in batches:
            self.optimizer.zero_grad()

            out = self.model(tokens)
            loss = nn.functional.cross_entropy(out, labels.to_dense())
            loss.backward()

            nn.utils.clip_grad.clip_grad_norm_(
                self.model.parameters(), max_norm=0.1, norm_type=2
            )

            self.optimizer.step()

        end = time.perf_counter()
        print(
            f"epoch complete - train_loss={round(loss.item(), 4)} - time={round(end -start, 4)}"
        )

    def validate(self, filename, recall_at=[], precision_at=[], num_buckets_to_eval=25):
        self.model.eval()

        columns = data.CsvIterator.all(
            dataset.FileDataSource(filename), delimiter=self.csv_delimiter
        )
        columns = self._text_transform()(columns)
        columns = self._entity_parse_transform()(columns)

        batch_size = 10_000
        inputs = to_padded_batch(columns[self.text_col].data(), batch_size)
        label_batches = []
        for i in range(0, len(columns), batch_size):
            label_batches.append(columns[self.label_col].data()[i : i + batch_size])

        top_k = max(recall_at + precision_at)

        metrics = [Recall(k) for k in recall_at] + [Precision(k) for k in precision_at]

        for tokens, labels in zip(inputs, label_batches):
            _, predictions = torch.topk(self.model(tokens), k=top_k, dim=1)

            for sample_preds, sample_labels in zip(predictions, labels):
                for metric in metrics:
                    metric.record(sample_preds, sample_labels)

        metric_vals = {}
        for metric in metrics:
            print(f"{metric.name()} = {metric.value()}")
            metric_vals[metric.name()] = metric.value()

        return metric_vals


def scifact():
    model = Classifier(
        input_dim=100_000,
        emb_dim=1024,
        n_classes=5183,
        lr=0.05,
    )

    for _ in range(5):
        print("\nCold Start")
        model.train(
            "/Users/nmeisburger/ThirdAI/data/scifact/unsupervised.csv",
            strong_cols=["TITLE"],
            weak_cols=["TEXT"],
        )

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

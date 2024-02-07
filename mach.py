from thirdai import smx, data


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

        self.emb = smx.Embedding(dim=emb_dim, input_dim=input_dim)

        self.output = smx.SparseLinear(
            input_dim=emb_dim, output_dim=output_dim, sparsity=sparsity
        )

    def forward(self, x, y=None):
        out = smx.relu(self.emb(x))
        out = self.output(out, y)
        return out

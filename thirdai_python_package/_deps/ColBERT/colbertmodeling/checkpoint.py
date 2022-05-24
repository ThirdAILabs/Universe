import torch


from thirdai._deps.ColBERT.colbertmodeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
)
from thirdai._deps.ColBERT.colbertmodeling.colbert import ColBERT


class Checkpoint(ColBERT):
    """
    Easy inference with ColBERT.

    TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name):
        super().__init__(name)

        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

    def query(self, *args, to_cpu=True, **kw_args):
        with torch.no_grad():
            Q = super().query(*args, **kw_args)
            return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=True, **kw_args):
        with torch.no_grad():
            D = super().doc(*args, **kw_args)

            if to_cpu:
                return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

            return D

    def queryFromText(self, queries, context=None):
        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context
        )
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs):
        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=True)

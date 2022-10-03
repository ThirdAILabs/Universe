import thirdai._thirdai.search
from thirdai._thirdai.search import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.search))


class EasyQA:
    """
    A simple QA system for if you are indexing less than ~100,000 documents.
    """

    def __init__(self):
        pass

    def index(self, id_passage_pairs):
        from thirdai.embeddings import DocSearchModel
        from thirdai.search import DocRetrieval
        from tqdm import tqdm

        self.embedding_model = DocSearchModel()
        reduced_centroids = self.embedding_model.getCentroids()[
            : max(len(id_passage_pairs) // 1000, 1)
        ]
        self.index = DocRetrieval(
            dense_input_dimension=128,
            num_tables=16,
            hashes_per_table=6,
            centroids=reduced_centroids,
        )
        for doc_id, doc_text in tqdm(id_passage_pairs):
            embedding = self.embedding_model.encodeDocs([doc_text])[0]
            self.index.add_doc(
                doc_id=doc_id, doc_text=doc_text, doc_embeddings=embedding
            )

        return self

    def query(self, question, top_k=1):
        embedding = self.embedding_model.encodeQuery(question)
        return self.index.query(embedding, top_k=top_k)

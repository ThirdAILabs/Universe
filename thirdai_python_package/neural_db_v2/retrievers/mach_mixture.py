from typing import List, Tuple, Iterable
from pathlib import Path

from core.types import DocId, Document
from core.retriever import Retriever, Score
from retrievers.mach import Mach
from utils.checkpointing import prepare_checkpoint_location


class MachMixture(Retriever):
    def __init__(self, num_models: int, **kwargs):
        self.models = [Mach(**kwargs) for _ in range(num_models)]

    def _doc_id_to_shard_id(self, doc_id: DocId) -> int:
        return doc_id % len(self.models)

    def _shard(self, docs: Iterable[Document]) -> List[Iterable[Document]]:
        def shard_generator(shard_id: int):
            for doc in docs:
                if self._doc_id_to_shard_id(doc.doc_id) == shard_id:
                    yield doc

        return [shard_generator(shard_id) for shard_id in range(len(self.models))]

    def find(self, query: str, top_k: int, **kwargs) -> List[Tuple[DocId, Score]]:
        return sorted(
            [
                result
                for model in self.models
                for result in model.find(query, top_k, **kwargs)
            ],
            key=lambda result: result[1],
            reverse=True,
        )

    def insert_batch(self, docs: Iterable[Document], checkpoint: Path, **kwargs):
        prepare_checkpoint_location(checkpoint)
        doc_shards = self._shard(docs)
        for shard_id, (model, doc_shard) in enumerate(zip(self.models, doc_shards)):
            model.insert_batch(doc_shard, checkpoint / str(shard_id), **kwargs)

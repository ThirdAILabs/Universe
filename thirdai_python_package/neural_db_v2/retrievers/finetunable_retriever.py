import json
import os
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
import torch
from thirdai import search
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..core.retriever import Retriever
from ..core.types import ChunkBatch, ChunkId, Score, SupervisedBatch


class Splade:
    def __init__(self):
        self.model = AutoModelForMaskedLM.from_pretrained(
            "naver/splade-cocondenser-selfdistil"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "naver/splade-cocondenser-selfdistil"
        )

    def augment(self, text: str) -> str:
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        output = self.model(**tokens)["logits"]
        scores, _ = torch.max(
            torch.log(1 + torch.relu(output)) * tokens["attention_mask"].unsqueeze(-1),
            dim=1,
        )

        tokens = scores.squeeze().nonzero().squeeze()
        tokens = " ".join(
            t
            for t in map(self.tokenizer._convert_id_to_token, tokens)
            if not t.startswith("##")
        )
        return text + " " + tokens


class FinetunableRetriever(Retriever):
    def __init__(
        self,
        save_path: Optional[str] = None,
        config: Optional[search.IndexConfig] = search.IndexConfig(),
        splade: bool = False,
        **kwargs
    ):
        super().__init__()
        self.retriever = search.FinetunableRetriever(save_path=save_path, config=config)
        if splade:
            self.splade = Splade()
        else:
            self.splade = None

    def search(
        self, queries: List[str], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        if self.splade:
            queries = [self.splade.augment(q) for q in queries]
        return self.retriever.query(queries, k=top_k)

    def rank(
        self, queries: List[str], choices: List[Set[ChunkId]], top_k: int, **kwargs
    ) -> List[List[Tuple[ChunkId, Score]]]:
        if self.splade:
            queries = [self.splade.augment(q) for q in queries]
        return self.retriever.rank(queries, candidates=choices, k=top_k)

    def upvote(self, queries: List[str], chunk_ids: List[ChunkId], **kwargs):
        self.retriever.finetune(
            doc_ids=list(map(lambda id: [id], chunk_ids)), queries=queries
        )

    def associate(
        self, sources: List[str], targets: List[str], associate_strength=4, **kwargs
    ):
        self.retriever.associate(
            sources=sources, targets=targets, strength=associate_strength
        )

    def insert(self, chunks: Iterable[ChunkBatch], index_batch_size=100000, **kwargs):
        for chunk in chunks:
            # Indexing in batches within a chunk reduces the RAM usage significantly
            # for large chunks
            for i in range(0, len(chunk), index_batch_size):
                ids = chunk.chunk_id[i : i + index_batch_size]

                keywords = chunk.keywords[i : i + index_batch_size].reset_index(
                    drop=True
                )
                text = chunk.text[i : i + index_batch_size].reset_index(drop=True)
                if self.splade:
                    text = pd.Series([self.splade.augment(t) for t in text])

                texts = keywords + " " + text
                self.retriever.index(ids=ids.to_list(), docs=texts.to_list())

    def supervised_train(self, samples: Iterable[SupervisedBatch], **kwargs):
        for batch in samples:
            self.retriever.finetune(
                doc_ids=batch.chunk_id.to_list(), queries=batch.query.to_list()
            )

    def delete(self, chunk_ids: List[ChunkId], **kwargs):
        self.retriever.remove(ids=chunk_ids)

    @staticmethod
    def options_path(path: str) -> str:
        return os.path.join(path, "options.json")

    def save(self, path: str):
        self.retriever.save(path)
        options = {"splade": bool(self.splade is not None)}
        with open(FinetunableRetriever.options_path(path), "w") as f:
            json.dump(options, f)

    @classmethod
    def load(cls, path: str, read_only: bool = False, **kwargs):
        instance = cls()
        instance.retriever = search.FinetunableRetriever.load(path, read_only=read_only)
        if os.path.exists(FinetunableRetriever.options_path(path)):
            with open(FinetunableRetriever.options_path(path), "r") as f:
                options = json.load(f)
        else:
            options = {}
        if "splade" in options and options["splade"]:
            instance.splade = Splade()
        else:
            instance.splade = None
        return instance

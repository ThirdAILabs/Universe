from typing import List, Tuple, Iterable, Callable
from pathlib import Path

from thirdai import bolt
from core.types import DocId, Document
from core.retriever import Retriever, Score
from retrievers.data_source import DocumentDataSource
from utils.checkpointing import prepare_checkpoint_location


DOC_ID = "doc_id"
STRONG = "strong"
WEAK = "weak"
QUERY = "query"


class TrainingProgressCallback(bolt.train.callbacks.Callback):
    def __init__(self, epochs_left: int, make_checkpoint: Callable[[int], None]):
        super().__init__()
        self.epochs_left = 0
        self.make_checkpoint = make_checkpoint

    def on_epoch_end(self):
        self.epochs_left -= 1
        self.make_checkpoint(self.epochs_left)


class Mach(Retriever):
    def __init__(self):
        self.model = bolt.UniversalDeepTransformer(
            data_types={
                QUERY: bolt.types.text(tokenizer=self.tokenizer),
                DOC_ID: bolt.types.categorical(delimiter=self.id_delimiter),
            },
            target=DOC_ID,
            n_target_classes=1000,  # This is going to be irrelevant
            integer_target=True,
            options={
                "extreme_classification": True,
                "extreme_output_dim": 50_000,
                "fhr": 100_000,
                "embedding_dimension": 2048,
                "extreme_num_hashes": 16,
                "hidden_bias": False,
                "rlhf": True,
            },
        )
        self.model.clear_index()

    def find(self, query: str, top_k: int, **kwargs) -> List[Tuple[DocId, Score]]:
        if not self.model:
            return []
        n_ids = self.model.get_index().num_entities()
        self.model.set_decode_params(min(self.n_ids, top_k), min(self.n_ids, 100))
        return self.model.predict({QUERY: query})

    def insert_batch(self, docs: Iterable[Document], checkpoint: Path, **kwargs):
        prepare_checkpoint_location(checkpoint)

        check_checkpoint = True
        doc_data_source = None
        if not (check_checkpoint and self.load_introduction_checkpoint(checkpoint)):
            check_checkpoint = False
            doc_data_source = DocumentDataSource(
                list(docs), DOC_ID, STRONG, WEAK, QUERY
            )
            self.model.introduce_documents_on_data_source(
                data_source=doc_data_source,
                strong_column_names=[STRONG],
                weak_column_names=[WEAK],
                fast_approximation=True,
                num_buckets_to_sample=24,
            )
            self.make_introduction_checkpoint(checkpoint)

        epochs_left = (
            self.load_training_checkpoint(checkpoint) if check_checkpoint else 10
        )
        if epochs_left:
            doc_data_source = doc_data_source or DocumentDataSource(
                list(docs), DOC_ID, STRONG, WEAK, QUERY
            )
            self.model.cold_start_on_data_source(
                data_source=doc_data_source,
                strong_column_names=[STRONG],
                weak_column_names=[WEAK],
                batch_size=2048,
                learning_rate=0.001,
                epochs=epochs_left,
                metrics=["hash_precision@5"],
                callbacks=[
                    TrainingProgressCallback(
                        epochs_left,
                        lambda epochs: self.make_training_checkpoint(
                            epochs, checkpoint
                        ),
                    )
                ],
            )

    def make_training_checkpoint(self, epochs_left: int, checkpoint: Path):
        pass

    def load_training_checkpoint(self, checkpoint: Path) -> int:
        pass

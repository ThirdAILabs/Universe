import copy
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import unidecode
from thirdai._thirdai import bolt
from thirdai.dataset.data_source import PyDataSource

from . import loggers, teachers
from .documents import CSV, Document, DocumentManager, Reference
from .models import CancelState, Mach
from .savable_state import State

Strength = Enum("Strength", ["Weak", "Medium", "Strong"])


def no_op(*args, **kwargs):
    pass


class Sup:
    def __init__(
        self,
        csv: str = None,
        query_column: str = None,
        id_column: str = None,
        queries: Sequence[str] = None,
        labels: Sequence[int] = None,
        source_id: str = "",
    ):
        if csv is not None and query_column is not None and id_column is not None:
            df = pd.read_csv(csv)
            self.queries = df[query_column]
            self.labels = df[id_column]
        elif queries is not None and labels is not None:
            if len(queries) != len(labels):
                raise ValueError(
                    "Queries and labels sequences must be the same length."
                )
            self.queries = queries
            self.labels = labels
        # elif csv is None and
        else:
            raise ValueError(
                "Sup must be initialized with csv, query_column and id_column, or queries and labels."
            )
        self.source_id = source_id


class SupDataSource(PyDataSource):
    def __init__(self, doc_manager: DocumentManager, query_col: str, data: List[Sup]):
        PyDataSource.__init__(self)
        self.doc_manager = doc_manager
        self.query_col = query_col
        self.data = data
        self.restart()

    def _csv_line(self, query: str, label: str):
        df = pd.DataFrame(
            {
                self.query_col: [query],
                self.doc_manager.id_column: [label],
            }
        )
        return df.to_csv(header=None, index=None).strip("\n")

    def _get_line_iterator(self):
        # First yield the header
        yield self._csv_line(self.query_col, self.doc_manager.id_column)
        # Then yield rows
        for sup in self.data:
            source_ids = self.doc_manager.match_source_id_by_prefix(sup.source_id)
            if len(source_ids) == 0:
                raise ValueError(f"Cannot find source with id {sup.source_id}")
            if len(source_ids) > 1:
                raise ValueError(f"Multiple sources match the prefix {sup.source_id}")
            _, start_id = self.doc_manager.source_by_id(source_ids[0])
            for query, label in zip(sup.queries, sup.labels):
                yield self._csv_line(query, str(label + start_id))

    def resource_name(self) -> str:
        return "Supervised training samples"


class NeuralDB:
    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._savable_state: Optional[State] = None

    def from_scratch(self, **kwargs) -> None:
        self._savable_state = State(
            model=Mach(id_col="id", query_col="query", **kwargs),
            logger=loggers.LoggerList([loggers.InMemoryLogger()]),
        )

    def from_checkpoint(
        self,
        checkpoint_path: str,
        on_progress: Callable = no_op,
        on_error: Callable = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        try:
            self._savable_state = State.load(checkpoint_path, on_progress)
            if self._savable_state.model and self._savable_state.model.get_model():
                self._savable_state.model.get_model().set_mach_sampling_threshold(0.01)
            if not isinstance(self._savable_state.logger, loggers.LoggerList):
                # TODO(Geordie / Yash): Add DBLogger to LoggerList once ready.
                self._savable_state.logger = loggers.LoggerList(
                    [self._savable_state.logger]
                )
        except Exception as e:
            self._savable_state = None
            if on_error is not None:
                on_error(error_msg=e.__str__())
            else:
                raise e

    def from_udt(
        self,
        udt: bolt.UniversalDeepTransformer,
        csv: Optional[str] = None,
        csv_id_column: Optional[str] = None,
        csv_strong_columns: Optional[List[str]] = None,
        csv_weak_columns: Optional[List[str]] = None,
        csv_reference_columns: Optional[List[str]] = None,
    ):
        if csv is None:
            udt.clear_index()

        udt.enable_rlhf()
        udt.set_mach_sampling_threshold(0.01)
        input_dim, emb_dim, out_dim = udt.model_dims()
        data_types = udt.data_types()

        if len(data_types) != 2:
            raise ValueError(
                f"Incompatible UDT model. Expected two data types but found {len(data_types)}."
            )
        query_col = None
        id_col = None
        id_delimiter = None
        for column, dtype in data_types.items():
            if isinstance(dtype, bolt.types.text):
                query_col = column
            if isinstance(dtype, bolt.types.categorical):
                id_col = column
                id_delimiter = dtype.delimiter
        if query_col is None:
            raise ValueError(f"Incompatible UDT model. Cannot find a query column.")
        if id_col is None:
            raise ValueError(f"Incompatible UDT model. Cannot find an id column.")

        model = Mach(
            id_col=id_col,
            id_delimiter=id_delimiter,
            query_col=query_col,
            input_dim=input_dim,
            hidden_dim=emb_dim,
            extreme_output_dim=out_dim,
        )
        model.model = udt
        logger = loggers.LoggerList([loggers.InMemoryLogger()])
        self._savable_state = State(model=model, logger=logger)

        if csv is not None:
            if (
                csv_id_column is None
                or csv_strong_columns is None
                or csv_weak_columns is None
                or csv_reference_columns is None
            ):
                error_msg = "If the `csv` arg is provided, then the following args must also be provided:\n"
                error_msg += " - `csv_id_column`\n"
                error_msg += " - `csv_strong_columns`\n"
                error_msg += " - `csv_weak_columns`\n"
                error_msg += " - `csv_reference_columns`\n"
                raise ValueError(error_msg)
            csv_doc = CSV(
                path=csv,
                id_column=csv_id_column,
                strong_columns=csv_strong_columns,
                weak_columns=csv_weak_columns,
                reference_columns=csv_reference_columns,
            )
            self._savable_state.documents.add([csv_doc])
            self._savable_state.model.set_n_ids(csv_doc.size)

    def in_session(self) -> bool:
        return self._savable_state is not None

    def ready_to_search(self) -> bool:
        return self.in_session() and self._savable_state.ready()

    def clear_session(self) -> None:
        self._savable_state = None

    def sources(self) -> Dict[str, str]:
        return self._savable_state.documents.sources()

    def save(self, save_to: str, on_progress: Callable = no_op) -> None:
        return self._savable_state.save(Path(save_to), on_progress)

    def insert(
        self,
        sources: List[Document],
        train: bool = True,
        use_weak_columns: bool = False,
        num_buckets_to_sample: int = 16,
        on_progress: Callable = no_op,
        on_success: Callable = no_op,
        on_error: Callable = None,
        on_irrecoverable_error: Callable = None,
        cancel_state: CancelState = None,
    ) -> List[str]:
        documents_copy = copy.deepcopy(self._savable_state.documents)
        try:
            intro_and_train, ids = self._savable_state.documents.add(sources)
        except Exception as e:
            self._savable_state.documents = documents_copy
            if on_error is not None:
                on_error(error_msg=f"Failed to add files. {e.__str__()}")
                return []
            raise e

        try:
            self._savable_state.model.index_documents(
                intro_documents=intro_and_train.intro,
                train_documents=intro_and_train.train,
                num_buckets_to_sample=num_buckets_to_sample,
                should_train=train,
                use_weak_columns=use_weak_columns,
                on_progress=on_progress,
                cancel_state=cancel_state,
            )

            self._savable_state.logger.log(
                session_id=self._user_id,
                action="Train",
                args={"files": intro_and_train.intro.resource_name()},
            )

            on_success()
            return ids

        except Exception as e:
            # If we fail during training here it's hard to guarantee that we
            # recover to a resumable state. E.g. if we're in the middle of
            # introducing new documents, we may be in a weird state where half
            # the documents are introduced while others aren't.
            # At the same time, if we fail here, then there must be something
            # wrong with the model, not how we used it, so it should be very
            # rare and probably not worth saving.
            self.clear_session()
            if on_irrecoverable_error is not None:
                on_irrecoverable_error(
                    error_msg=f"Failed to train model on added files. {e.__str__()}"
                )
                return []
            raise e

    def clear_sources(self) -> None:
        self._savable_state.documents.clear()
        self._savable_state.model.forget_documents()

    def search(
        self, query: str, top_k: int, on_error: Callable = None
    ) -> List[Reference]:
        try:
            result_ids = self._savable_state.model.infer_labels(
                samples=[query], n_results=top_k
            )[0]
            return [self._savable_state.documents.reference(rid) for rid in result_ids]
        except Exception as e:
            if on_error is not None:
                on_error(e.__str__())
                return []
            raise e

    def _get_text(self, result_id) -> str:
        return self._savable_state.documents.reference(result_id).text

    def text_to_result(self, text: str, result_id: int) -> None:
        teachers.upvote(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            query_id_para=[
                (text, upvote_id, self._get_text(result_id))
                for upvote_id in self._savable_state.documents.reference(
                    result_id
                ).upvote_ids
            ],
        )

    def text_to_result_batch(self, text_id_pairs: List[Tuple[str, int]]) -> None:
        query_id_para = [
            (query, upvote_id, self._get_text(result_id))
            for query, result_id in text_id_pairs
            for upvote_id in self._savable_state.documents.reference(
                result_id
            ).upvote_ids
        ]
        teachers.upvote(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            query_id_para=query_id_para,
        )

    def associate(self, source: str, target: str, strength: Strength = Strength.Strong):
        top_k = self._get_associate_top_k(strength)
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_pairs=[(source, target)],
            top_k=top_k,
        )

    def associate_batch(
        self, text_pairs: List[Tuple[str, str]], strength: Strength = Strength.Strong
    ):
        top_k = self._get_associate_top_k(strength)
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_pairs=text_pairs,
            top_k=top_k,
        )

    def _get_associate_top_k(self, strength):
        if strength == Strength.Weak:
            return 3
        elif strength == Strength.Medium:
            return 5
        elif strength == Strength.Strong:
            return 7
        else:
            return 7

    def supervised_train(
        self,
        data: List[Sup],
        learning_rate=0.0001,
        epochs=3,
    ):
        doc_manager = self._savable_state.documents
        query_col = self._savable_state.model.get_query_col()
        self._savable_state.model.get_model().train_on_data_source(
            data_source=SupDataSource(doc_manager, query_col, data),
            learning_rate=learning_rate,
            epochs=epochs,
        )

    def get_associate_samples(self):
        logs = self._savable_state.logger.get_logs()

        associate_logs = logs[logs["action"] == "associate"]
        associate_samples = []
        for _, row in associate_logs.iterrows():
            for source, target in row["args"]["pairs"]:
                associate_samples.append((source, target))

        return associate_samples

    def get_upvote_samples(self):
        logs = self._savable_state.logger.get_logs()

        upvote_associate_samples = []
        upvote_logs = logs[logs["action"] == "upvote"]
        for _, row in upvote_logs.iterrows():
            if "query_id_para" in row["args"]:
                for source, _, target in row["args"]["query_id_para"]:
                    upvote_associate_samples.append((source, target))

        return upvote_associate_samples

    def get_rlhf_samples(self):
        return self.get_associate_samples() + self.get_upvote_samples()

    def retrain(
        self,
        text_pairs: List[Tuple[str, str]] = [],
        learning_rate: float = 0.0001,
        epochs: int = 3,
        strength: Strength = Strength.Strong,
    ):
        doc_manager = self._savable_state.documents

        if not text_pairs:
            text_pairs = self.get_rlhf_samples()

        self._savable_state.model.retrain(
            balancing_data=doc_manager.get_data_source(),
            source_target_pairs=text_pairs,
            n_buckets=self._get_associate_top_k(strength),
            learning_rate=learning_rate,
            epochs=epochs,
        )

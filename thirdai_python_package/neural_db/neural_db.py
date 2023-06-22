from pathlib import Path
from enum import Enum
from typing import Callable, List, Optional
import copy

from .documents import Reference, Document
from .savable_state import State
from .models import Mach
from . import qa, teachers, loggers


Strength = Enum("Strength", ["Weak", "Medium", "Strong"])


class SearchState:
    def __init__(self, query: str, references: List[Reference]):
        self._query = query
        self._references = references

    def len(self):
        return len(self._references)

    def __len__(self):
        return self.len()

    def references(self):
        return self._references


class AnswererState:
    def __init__(self, answerer: qa.QA, context_args: qa.ContextArgs):
        self._answerer = answerer
        self._context_args = context_args

    def answerer(self):
        return self._answerer

    def context_args(self):
        return self._context_args


def no_op(*args, **kwargs):
    pass


class NeuralDB:
    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._savable_state: Optional[State] = None
        self._search_state: Optional[SearchState] = None
        self._answerer_state: Optional[AnswererState] = None

    def from_scratch(self) -> None:
        self._savable_state = State(
            model=Mach(id_col="id", query_col="query"),
            logger=loggers.LoggerList([]),
        )
        self._search_state = None

    def in_session(self) -> bool:
        return self._savable_state is not None

    def ready_to_search(self) -> bool:
        return self.in_session() and self._savable_state.ready()

    def from_checkpoint(
        self, checkpoint_path: Path, on_progress: Callable = no_op, on_error: Callable = no_op
    ):
        try:
            self._savable_state = State.load(checkpoint_path, on_progress)
            if not isinstance(self._savable_state.logger, loggers.LoggerList):
                # TODO(Geordie / Yash): Add DBLogger to LoggerList once ready.
                self._savable_state.logger = loggers.LoggerList(
                    [self._savable_state.logger]
                )
        except Exception as e:
            self._savable_state = None
            on_error(error_msg=e.__str__())

        self._search_state = None

    def clear_session(self) -> None:
        self._savable_state = None
        self._search_state = None

    def sources(self) -> List[str]:
        return self._savable_state.documents.sources()

    def save(self, save_to: Path, on_progress: Callable = no_op) -> None:
        return self._savable_state.save(save_to, on_progress)

    def insert(
        self,
        sources: List[Document],
        on_progress: Callable = no_op,
        on_success: Callable = no_op,
        on_error: Callable = no_op,
        on_irrecoverable_error: Callable = no_op,
    ) -> None:
        documents_copy = copy.deepcopy(self._savable_state.documents)
        try:
            intro_and_train = self._savable_state.documents.add(sources)
        except Exception as e:
            self._savable_state.documents = documents_copy
            on_error(error_msg=f"Failed to add files. {e.__str__()}")
            return
        
        try:
            self._savable_state.model.index_documents(
                intro_documents=intro_and_train.intro,
                train_documents=intro_and_train.train,
                on_progress=on_progress,
            )
            
            self._savable_state.logger.log(
                session_id=self._user_id,
                action="Train",
                args={"files": intro_and_train.intro.resource_name()},
            )
        
            on_success()

        except Exception as e:
            # If we fail during training here it's hard to guarantee that we
            # recover to a resumable state. E.g. if we're in the middle of
            # introducing new documents, we may be in a weird state where half
            # the documents are introduced while others aren't.
            # At the same time, if we fail here, then there must be something
            # wrong with the model, not how we used it, so it should be very
            # rare and probably not worth saving.
            self.clear_session()
            on_irrecoverable_error(
                error_msg=f"Failed to train model on added files. {e.__str__()}"
            )
        
    def clear_sources(self) -> None:
        self._savable_state.documents.clear()
        self._savable_state.model.forget_documents()

    def search(self, query: str, top_k: int, on_error: Callable = no_op) -> List[Reference]:
        try:
            result_ids = self._savable_state.model.infer_labels(samples=[query], n_results=top_k)[0]
            self._search_state = SearchState(
                query=query,
                references=[
                    self._savable_state.documents.reference(id) for id in result_ids
                ],
            )
            return self._search_state.references()
        except Exception as e:
            on_error(e.__str__())
            return []

    def text_to_result(self, text: str, result_id: int) -> None:
        teachers.upvote(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            query=text,
            liked_passage_id=result_id,
        )

    def associate(self, source: str, target: str, strength: Strength = Strength.Strong):
        if strength == Strength.Weak:
            top_k = 3
        elif strength == Strength.Medium:
            top_k = 5
        elif strength == Strength.Strong:
            top_k = 7
        else:
            top_k = 7
        teachers.associate(
            model=self._savable_state.model,
            logger=self._savable_state.logger,
            user_id=self._user_id,
            text_a=source,
            text_b=target,
            top_k=top_k,
        )

    def set_answerer_state(self, answerer_state: AnswererState):
        self._answerer_state = answerer_state

    def can_answer(self) -> bool:
        return self._answerer_state is not None

    def answer(self, on_error: Callable = no_op):
        num_references = self._answerer_state.context_args().num_references
        references = self._search_state.references()[:num_references]

        # Check if "num_references" is the only context arg defined.
        # If not, we retrieve custom context for each document.
        if len(vars(self._answerer_state.context_args())) == 1:
            answers = [reference.text() for reference in references]
        else:
            answers = [
                self._savable_state.documents.context(
                    reference.id(), self._answerer_state.context_args()["chunk_radius"]
                )
                for reference in references
            ]

        return self._answerer_state.answerer().answer(
            question=self._search_state._query,
            answers=answers,
            on_error=on_error,
        )


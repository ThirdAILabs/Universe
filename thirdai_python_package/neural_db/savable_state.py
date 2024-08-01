import datetime
import os
from pathlib import Path
from typing import Callable, List

from .documents import DocumentManager
from .loggers import Logger
from .models.model_interface import Model
from .trainer.checkpoint_config import CheckpointConfig
from .utils import delete_file, delete_folder, pickle_to, unpickle_from


def default_checkpoint_name():
    return Path(f"checkpoint_{datetime.datetime.now()}.ndb")


class State:
    def __init__(self, model: Model, logger: Logger) -> None:
        self.model = model
        self.logger = logger
        self.documents = DocumentManager(
            id_column=model.get_id_col(),
            strong_column="strong",
            weak_column="weak",
        )

    def ready(self) -> bool:
        return (
            self.model is not None
            and self.logger is not None
            and self.documents is not None
            and self.model.searchable
        )

    @staticmethod
    def model_pkl_path(directory: Path) -> Path:
        return directory / "model.pkl"

    @staticmethod
    def model_meta_path(directory: Path) -> Path:
        return directory / "model"

    @staticmethod
    def logger_pkl_path(directory: Path) -> Path:
        return directory / "logger.pkl"

    @staticmethod
    def logger_meta_path(directory: Path) -> Path:
        return directory / "logger"

    @staticmethod
    def documents_pkl_path(directory: Path) -> Path:
        return directory / "documents.pkl"

    @staticmethod
    def documents_meta_path(directory: Path) -> Path:
        return directory / "documents"

    def save(
        self,
        location=default_checkpoint_name(),
        on_progress: Callable = lambda *args, **kwargs: None,
    ) -> str:
        total_steps = 7

        # make directory
        directory = Path(location)
        os.makedirs(directory)
        on_progress(1 / total_steps)

        # pickle model
        pickle_to(self.model, State.model_pkl_path(directory))
        on_progress(2 / total_steps)
        # save model meta
        os.mkdir(State.model_meta_path(directory))
        self.model.save_meta(State.model_meta_path(directory))
        on_progress(3 / total_steps)

        # pickle logger
        pickle_to(self.logger, State.logger_pkl_path(directory))
        on_progress(4 / total_steps)
        # save logger meta
        os.mkdir(State.logger_meta_path(directory))
        self.logger.save_meta(State.logger_meta_path(directory))
        on_progress(5 / total_steps)

        # pickle documents
        pickle_to(self.documents, State.documents_pkl_path(directory))
        on_progress(6 / total_steps)
        # save documents meta
        os.mkdir(State.documents_meta_path(directory))
        self.documents.save_meta(State.documents_meta_path(directory))
        on_progress(7 / total_steps)

        return str(directory)

    @staticmethod
    def load(
        location: Path, on_progress: Callable = lambda *args, **kwargs: None, **kwargs
    ):
        total_steps = 6

        # load model
        model = unpickle_from(State.model_pkl_path(location))
        on_progress(1 / total_steps)
        model.load_meta(State.model_meta_path(location), **kwargs)
        on_progress(2 / total_steps)

        # load logger
        logger = unpickle_from(State.logger_pkl_path(location))
        on_progress(3 / total_steps)
        logger.load_meta(State.logger_meta_path(location))
        on_progress(4 / total_steps)

        state = State(model=model, logger=logger)

        # load documents
        state.documents = unpickle_from(State.documents_pkl_path(location))
        on_progress(5 / total_steps)
        state.documents.load_meta(State.documents_meta_path(location))
        on_progress(6 / total_steps)

        return state


def load_checkpoint(checkpoint_config: CheckpointConfig):
    try:
        documents, ids, resource_name = unpickle_from(
            checkpoint_config.pickled_documents_ids_resource_name_path
        )
        return documents, ids, resource_name
    except:
        raise Exception(
            "Failed to load"
            f" '{checkpoint_config.pickled_documents_ids_resource_name_path}'."
            " Please verify it's a valid document manager checkpoint and the training is"
            " incomplete."
        )


def make_preinsertion_checkpoint(
    savable_state: State,
    ids: List[str],
    resource_name: str,
    checkpoint_config: CheckpointConfig,
):
    checkpoint_config.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    # saving the state of the document manager
    pickle_to(
        (savable_state.documents, ids, resource_name),
        checkpoint_config.pickled_documents_ids_resource_name_path,
    )


def make_training_checkpoint(savable_state: State, checkpoint_config: CheckpointConfig):
    # removing last trained ndb
    delete_folder(checkpoint_config.ndb_trained_path)
    savable_state.save(location=checkpoint_config.ndb_trained_path)
    delete_file(checkpoint_config.pickled_documents_ids_resource_name_path)

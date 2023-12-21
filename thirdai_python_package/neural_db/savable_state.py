import datetime
import os
from pathlib import Path
from typing import Callable

from .documents import DocumentManager
from .loggers import Logger
from .models import Model
from .utils import pickle_to, unpickle_from, delete_folder, delete_file
from .training_state.checkpoint_config import NDBCheckpointConfig


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

    def model_pkl_path(directory: Path) -> Path:
        return directory / "model.pkl"

    def model_meta_path(directory: Path) -> Path:
        return directory / "model"

    def logger_pkl_path(directory: Path) -> Path:
        return directory / "logger.pkl"

    def logger_meta_path(directory: Path) -> Path:
        return directory / "logger"

    def documents_pkl_path(directory: Path) -> Path:
        return directory / "documents.pkl"

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
    def load(location: Path, on_progress: Callable = lambda *args, **kwargs: None):
        total_steps = 6

        # load model
        model = unpickle_from(State.model_pkl_path(location))
        on_progress(1 / total_steps)
        model.load_meta(State.model_meta_path(location))
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


def checkpoint_state_and_ids(
    savable_state: State, ids, resource_name, checkpoint_config: NDBCheckpointConfig
):
    savable_state.save(checkpoint_config.ndb_checkpoint_path)
    pickle_to((ids, resource_name), checkpoint_config.pickled_ids_resource_name_path)


def load_checkpoint_state_ids_from_config(checkpoint_config: NDBCheckpointConfig):
    state = State.load(checkpoint_config.ndb_checkpoint_path)
    ids, resource_name = unpickle_from(checkpoint_config.pickled_ids_resource_name_path)
    return state, ids, resource_name


def delete_checkpoint_state_and_ids(
    checkpoint_config: NDBCheckpointConfig, ignore_errors=True
):
    delete_folder(checkpoint_config.ndb_checkpoint_path, ignore_errors=ignore_errors)
    delete_file(
        checkpoint_config.pickled_ids_resource_name_path, ignore_errors=ignore_errors
    )
